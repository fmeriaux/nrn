//! End-to-end coverage of the `train` command's lifecycle and error paths:
//! architecture validation, continuing from a saved model, the resume
//! checkpoint-selection / trim logic, run-directory overwrite handling, and a
//! fatal (unrecovered) divergence. The happy paths and plotting live in
//! `train_plot.rs`.

use assert_cmd::Command;
use predicates::str::contains;
use std::fs;
use std::path::Path;

/// Runs `nrn` with the given args from `dir`.
fn nrn(dir: &Path, args: &[&str]) -> assert_cmd::assert::Assert {
    Command::cargo_bin("nrn")
        .unwrap()
        .current_dir(dir)
        .args(args)
        .assert()
}

/// Generates a small 2-class ring dataset in `dir` over the `[0, 10]` feature
/// range, returning its filename.
fn synth_ring(dir: &Path, seed: &str, samples: &str) -> String {
    nrn(
        dir,
        &[
            "synth",
            "--min",
            "0",
            "--max",
            "10",
            "--seed",
            seed,
            "--distribution",
            "ring",
            "--clusters",
            "2",
            "--samples",
            samples,
        ],
    )
    .success();
    format!("ring-seed{seed}-c2-f2-n{samples}")
}

fn checkpoint_count(run_dir: &Path) -> usize {
    fs::read_dir(run_dir)
        .map(|rd| {
            rd.filter_map(Result::ok)
                .filter(|e| {
                    e.file_name()
                        .to_str()
                        .map(|n| n.starts_with("checkpoint-"))
                        .unwrap_or(false)
                        && e.path().is_dir()
                })
                .count()
        })
        .unwrap_or(0)
}

#[test]
fn rejects_zero_neuron_hidden_layer() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let ds = synth_ring(dir, "1", "20");

    nrn(
        dir,
        &["train", "start", &ds, "--epochs", "2", "--layers", "4,0"],
    )
    .failure()
    .stderr(contains("at least one output unit"));
}

#[test]
fn continues_from_saved_model_without_checkpoints() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let ds = synth_ring(dir, "5", "20");

    // First run saves the model-<ds>/ directory holding model.safetensors.
    nrn(
        dir,
        &[
            "train",
            "start",
            "--seed",
            "42",
            &ds,
            "--epochs",
            "2",
            "--checkpoint-interval",
            "1",
            "--no-clip",
        ],
    )
    .success();
    assert!(
        dir.join(format!("model-{ds}"))
            .join("model.safetensors")
            .exists()
    );

    // Continue from that model with checkpoints disabled (interval 0 → no recorder).
    nrn(
        dir,
        &[
            "train",
            "start",
            &ds,
            "--model",
            &format!("model-{ds}"),
            "--epochs",
            "2",
            "--checkpoint-interval",
            "0",
            "--no-clip",
        ],
    )
    .success();
}

#[test]
fn scaling_persists_a_sidecar_and_survives_resume() {
    // --scale flows the fitted scaler through the whole run: a preprocessor.json
    // sidecar beside the model, the same sidecar recorded at the run root, and
    // resume reusing it (the run must still complete).
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let ds = synth_ring(dir, "11", "20");
    let run_arg = format!("training-model-{ds}");

    nrn(
        dir,
        &[
            "train",
            "start",
            "--seed",
            "42",
            &ds,
            "--scale",
            "min-max",
            "--epochs",
            "2",
            "--checkpoint-interval",
            "1",
            "--no-clip",
        ],
    )
    .success();

    // Sidecar written beside the model weights, and the run's own preprocessor.json.
    assert!(
        dir.join(format!("model-{ds}"))
            .join("preprocessor.json")
            .exists()
    );
    let run_scaler = fs::read_to_string(dir.join(&run_arg).join("preprocessor.json")).unwrap();
    assert!(run_scaler.contains("MinMax"));

    // Resume reuses the scaler from the run's preprocessor.json (no refit) and completes.
    nrn(dir, &["train", "resume", &run_arg, "--epochs", "1"]).success();
}

#[test]
fn overwrite_required_to_replace_existing_run() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let ds = synth_ring(dir, "6", "20");

    let start = |extra: &[&str]| {
        let mut args = vec![
            "train",
            "start",
            "--seed",
            "42",
            &ds,
            "--epochs",
            "2",
            "--checkpoint-interval",
            "1",
            "--no-clip",
        ];
        args.extend_from_slice(extra);
        nrn(dir, &args)
    };

    start(&[]).success();
    // Second start onto the same run dir must be refused with a remediation hint.
    start(&[]).failure().stderr(contains("use --overwrite"));
    // ...and accepted once --overwrite is given.
    start(&["--overwrite"]).success();
}

#[test]
fn resume_from_epoch_trims_later_checkpoints() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let ds = synth_ring(dir, "7", "20");
    let run_arg = format!("training-model-{ds}");
    let run_dir = dir.join(&run_arg);

    // interval=1, 4 epochs → initial + 4 = 5 checkpoints.
    nrn(
        dir,
        &[
            "train",
            "start",
            "--seed",
            "42",
            &ds,
            "--epochs",
            "4",
            "--checkpoint-interval",
            "1",
            "--no-clip",
        ],
    )
    .success();
    assert_eq!(checkpoint_count(&run_dir), 5);

    // Resuming from epoch 0 rewinds the trajectory: the later checkpoints are
    // trimmed before training forward again.
    nrn(
        dir,
        &["train", "resume", &run_arg, "--from", "0", "--epochs", "3"],
    )
    .success()
    .stderr(contains("Removed"));
}

#[test]
fn resume_with_no_checkpoints_fails() {
    // A run directory whose meta.json survives but whose checkpoints were all
    // removed (corrupted / manually cleaned run) cannot be resumed.
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let ds = synth_ring(dir, "9", "20");
    let run_arg = format!("training-model-{ds}");
    let run_dir = dir.join(&run_arg);

    nrn(
        dir,
        &[
            "train",
            "start",
            "--seed",
            "42",
            &ds,
            "--epochs",
            "2",
            "--checkpoint-interval",
            "1",
            "--no-clip",
        ],
    )
    .success();

    // Wipe every checkpoint directory, leaving meta.json behind.
    for entry in fs::read_dir(&run_dir).unwrap().filter_map(Result::ok) {
        if entry
            .file_name()
            .to_str()
            .is_some_and(|n| n.starts_with("checkpoint-"))
        {
            fs::remove_dir_all(entry.path()).unwrap();
        }
    }

    nrn(dir, &["train", "resume", &run_arg, "--epochs", "3"])
        .failure()
        .stderr(contains("no checkpoints found"));
}

#[test]
fn resume_from_unknown_epoch_fails() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let ds = synth_ring(dir, "8", "20");
    let run_arg = format!("training-model-{ds}");

    nrn(
        dir,
        &[
            "train",
            "start",
            "--seed",
            "42",
            &ds,
            "--epochs",
            "2",
            "--checkpoint-interval",
            "1",
            "--no-clip",
        ],
    )
    .success();

    nrn(
        dir,
        &[
            "train", "resume", &run_arg, "--from", "999", "--epochs", "3",
        ],
    )
    .failure()
    .stderr(contains("no checkpoint recorded at epoch 999"));
}

#[test]
fn fatal_divergence_without_early_stopping_errors() {
    // An enormous lr + no-clip overflows the weights to ±inf. Without early stopping
    // there is no best model to recover, so the run must fail with the divergence error.
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let ds = synth_ring(dir, "42", "40");

    nrn(
        dir,
        &[
            "train",
            "start",
            "--seed",
            "42",
            &ds,
            "--epochs",
            "50",
            "--checkpoint-interval",
            "1000",
            "--no-clip",
            "--lr",
            "1e38",
            "--layers",
            "4,4",
        ],
    )
    .failure()
    .stderr(contains("model diverged at epoch"));
}
