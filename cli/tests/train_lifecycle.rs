//! End-to-end coverage of the `train` command's lifecycle and error paths:
//! architecture validation, continuing from a saved model, the resume
//! checkpoint-selection / trim logic, run-directory overwrite handling, and a
//! fatal (unrecovered) divergence. The happy paths and plotting live in
//! `train_plot.rs`.

mod common;

use common::{checkpoint_count, run, synth_ring};
use predicates::str::contains;
use std::fs;

#[test]
fn rejects_zero_neuron_hidden_layer() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let ds = synth_ring(dir, "1", "20");

    run(
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
    run(
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
    run(
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

    run(
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
    run(dir, &["train", "resume", &run_arg, "--epochs", "1"]).success();
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
        run(dir, &args)
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
    run(
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
    run(
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

    run(
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

    run(dir, &["train", "resume", &run_arg, "--epochs", "3"])
        .failure()
        .stderr(contains("no checkpoints found"));
}

#[test]
fn resume_from_unknown_epoch_fails() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let ds = synth_ring(dir, "8", "20");
    let run_arg = format!("training-model-{ds}");

    run(
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

    run(
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

    run(
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
