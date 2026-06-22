//! End-to-end coverage of the `train`/`plot` happy paths: a full run produces a
//! run directory and checkpoints, and `plot` covers its formats — `plot run`
//! (console, still PNG, animated GIF, the two-checkpoint minimum) and `plot
//! dataset` (console and image). The recap reflects resume overrides. Also
//! exercises early-stopping checkpoint flushing and optimizer-state restoration
//! on resume. The lifecycle and error paths live in `train_lifecycle.rs`.

use assert_cmd::Command;
use predicates::str::contains;
use std::fs;
use std::path::Path;

fn checkpoint_count(run_dir: &std::path::Path) -> usize {
    fs::read_dir(run_dir)
        .map(|rd| {
            rd.filter_map(Result::ok)
                .filter(|e| {
                    let name = e.file_name();
                    name.to_str()
                        .map(|n| n.starts_with("checkpoint-"))
                        .unwrap_or(false)
                        && e.path().is_dir()
                })
                .count()
        })
        .unwrap_or(0)
}

/// Runs `nrn` with the given args from `dir`.
fn nrn(dir: &std::path::Path, args: &[&str]) -> assert_cmd::assert::Assert {
    Command::cargo_bin("nrn")
        .unwrap()
        .current_dir(dir)
        .args(args)
        .assert()
}

#[test]
fn train_creates_run_dir_and_plot_generates_png_and_gif() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();

    // Generate a 2-class, 2-feature ring dataset (20 samples for speed).
    nrn(
        dir,
        &[
            "synth",
            "--min",
            "0",
            "--max",
            "10",
            "--seed",
            "42",
            "--distribution",
            "ring",
            "--clusters",
            "2",
            "--samples",
            "20",
        ],
    )
    .success();

    // Dataset filename produced by synth: ring-seed42-c2-f2-n20
    let ds_name = "ring-seed42-c2-f2-n20";

    // Train with checkpoints every 5 epochs (20 epochs total → ≥ 5 checkpoints).
    nrn(
        dir,
        &[
            "train",
            "start",
            "--seed",
            "42",
            ds_name,
            "--epochs",
            "20",
            "--checkpoint-interval",
            "5",
            "--no-clip",
        ],
    )
    .success();

    // Training run directory must exist.
    let run_dir = dir.join(format!("training-model-{ds_name}"));
    assert!(run_dir.is_dir(), "expected training run dir at {run_dir:?}");

    // Must have more than 2 checkpoints (guard from load_history).
    let count = checkpoint_count(&run_dir);
    assert!(count > 2, "expected >2 checkpoints, got {count}");

    let run_arg = format!("training-model-{ds_name}");

    // `plot run --format image --animate` emits the training curves (PNG) and,
    // for the 2D dataset recorded in the run's meta.json, the decision boundary
    // animation (GIF). Artifacts land beside the run directory.
    nrn(
        dir,
        &["plot", "run", &run_arg, "--format", "image", "--animate"],
    )
    .success();

    let png = dir.join(format!("curves-training-model-{ds_name}.png"));
    assert!(png.exists(), "expected curves PNG at {png:?}");

    let gif = dir.join(format!("boundary-training-model-{ds_name}.gif"));
    assert!(gif.exists(), "expected boundary GIF at {gif:?}");
}

#[test]
fn plot_run_still_image_emits_two_pngs_and_no_gif() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();

    nrn(
        dir,
        &[
            "synth",
            "--min",
            "0",
            "--max",
            "10",
            "--seed",
            "42",
            "--distribution",
            "ring",
            "--clusters",
            "2",
            "--samples",
            "20",
        ],
    )
    .success();

    let ds_name = "ring-seed42-c2-f2-n20";
    nrn(
        dir,
        &[
            "train",
            "start",
            "--seed",
            "42",
            ds_name,
            "--epochs",
            "20",
            "--checkpoint-interval",
            "5",
            "--no-clip",
        ],
    )
    .success();

    let run_arg = format!("training-model-{ds_name}");

    // No --animate: a still boundary PNG, no GIF.
    nrn(dir, &["plot", "run", &run_arg, "--format", "image"]).success();

    assert!(
        dir.join(format!("curves-training-model-{ds_name}.png"))
            .exists(),
        "expected a curves PNG"
    );
    assert!(
        dir.join(format!("boundary-training-model-{ds_name}.png"))
            .exists(),
        "expected a still boundary PNG"
    );
    assert!(
        !dir.join(format!("boundary-training-model-{ds_name}.gif"))
            .exists(),
        "a still plot must not emit a GIF"
    );
}

#[test]
fn plot_run_console_prints_curves_and_boundary_without_files() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();

    nrn(
        dir,
        &[
            "synth",
            "--min",
            "0",
            "--max",
            "10",
            "--seed",
            "42",
            "--distribution",
            "ring",
            "--clusters",
            "2",
            "--samples",
            "20",
        ],
    )
    .success();

    let ds_name = "ring-seed42-c2-f2-n20";
    nrn(
        dir,
        &[
            "train",
            "start",
            "--seed",
            "42",
            ds_name,
            "--epochs",
            "20",
            "--checkpoint-interval",
            "5",
            "--no-clip",
        ],
    )
    .success();

    let run_arg = format!("training-model-{ds_name}");

    let out = Command::cargo_bin("nrn")
        .unwrap()
        .current_dir(dir)
        .args(["plot", "run", &run_arg])
        .output()
        .unwrap();
    assert!(out.status.success());

    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("Training Loss Over Epochs"),
        "expected curves in stdout: {stdout}"
    );
    assert!(
        stdout.contains("Scatter Plot of Dataset Features"),
        "expected the boundary scatter in stdout: {stdout}"
    );
    assert!(
        !dir.join(format!("curves-training-model-{ds_name}.png"))
            .exists(),
        "the console format must not write files"
    );
}

#[test]
fn plot_dataset_renders_console_and_image() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();

    nrn(
        dir,
        &[
            "synth",
            "--min",
            "0",
            "--max",
            "10",
            "--seed",
            "42",
            "--distribution",
            "ring",
            "--clusters",
            "2",
            "--samples",
            "20",
        ],
    )
    .success();

    let ds_name = "ring-seed42-c2-f2-n20";

    // Console: an inline scatter on stdout, no file.
    let out = Command::cargo_bin("nrn")
        .unwrap()
        .current_dir(dir)
        .args(["plot", "dataset", ds_name])
        .output()
        .unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("Scatter Plot of Dataset Features"),
        "expected an inline scatter on stdout: {stdout}"
    );

    // Image: a PNG beside the dataset.
    nrn(dir, &["plot", "dataset", ds_name, "--format", "image"]).success();
    assert!(
        dir.join(format!("{ds_name}.png")).exists(),
        "expected a scatter PNG beside the dataset"
    );
}

#[test]
fn plot_run_skips_the_boundary_silently_for_non_2d_data() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();

    // A 3-feature dataset: the curves still plot, but the decision boundary
    // (a 2D-only bonus) cannot.
    nrn(
        dir,
        &[
            "synth",
            "--min",
            "0",
            "--max",
            "10",
            "--seed",
            "42",
            "--distribution",
            "ring",
            "--clusters",
            "2",
            "--features",
            "3",
            "--samples",
            "30",
        ],
    )
    .success();

    let ds_name = "ring-seed42-c2-f3-n30";
    nrn(
        dir,
        &[
            "train",
            "start",
            "--seed",
            "42",
            ds_name,
            "--epochs",
            "20",
            "--checkpoint-interval",
            "5",
            "--no-clip",
        ],
    )
    .success();

    let run_arg = format!("training-model-{ds_name}");

    // Without --animate: curves only, and no nag about the unavailable boundary.
    let out = Command::cargo_bin("nrn")
        .unwrap()
        .current_dir(dir)
        .args(["plot", "run", &run_arg])
        .output()
        .unwrap();
    assert!(out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !stderr.contains("two-feature"),
        "a still plot must not warn about the boundary: {stderr}"
    );

    // With --animate: the explicit request can't be honored, so it warns.
    let out = Command::cargo_bin("nrn")
        .unwrap()
        .current_dir(dir)
        .args(["plot", "run", &run_arg, "--animate"])
        .output()
        .unwrap();
    assert!(out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("two-feature dataset"),
        "an explicit --animate on non-2D data should warn: {stderr}"
    );
}

#[test]
fn plot_run_succeeds_at_the_two_checkpoint_minimum() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();

    nrn(
        dir,
        &[
            "synth",
            "--min",
            "0",
            "--max",
            "10",
            "--seed",
            "7",
            "--distribution",
            "ring",
            "--clusters",
            "2",
            "--samples",
            "20",
        ],
    )
    .success();

    let ds_name = "ring-seed7-c2-f2-n20";

    // interval=100 with epochs=2 → the initial and final checkpoints only: the
    // two-point minimum a run plot needs.
    nrn(
        dir,
        &[
            "train",
            "start",
            "--seed",
            "42",
            ds_name,
            "--epochs",
            "2",
            "--checkpoint-interval",
            "100",
        ],
    )
    .success();

    let run_arg = format!("training-model-{ds_name}");
    nrn(dir, &["plot", "run", &run_arg])
        .success()
        .stdout(contains("Training Loss Over Epochs"));
}

#[test]
fn early_stopping_writes_final_checkpoint() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();

    // 40 samples → 4 validation samples with default val_ratio=0.1.
    nrn(
        dir,
        &[
            "synth",
            "--min",
            "0",
            "--max",
            "10",
            "--seed",
            "99",
            "--distribution",
            "ring",
            "--clusters",
            "2",
            "--samples",
            "40",
        ],
    )
    .success();

    let ds_name = "ring-seed99-c2-f2-n40";

    // interval=1000 >> epochs=30: only initial + final-epoch checkpoint would be
    // written without early stopping. With patience=1 this fires quickly and
    // the fix must write a checkpoint at the stopped epoch.
    let out = Command::cargo_bin("nrn")
        .unwrap()
        .current_dir(dir)
        .args([
            "train",
            "start",
            "--seed",
            "42",
            ds_name,
            "--epochs",
            "30",
            "--checkpoint-interval",
            "1000",
            "--early-stopping",
            "1",
            "--no-clip",
        ])
        .output()
        .unwrap();
    assert!(out.status.success());

    let stdout = String::from_utf8_lossy(&out.stdout);
    let run_dir = dir.join(format!("training-model-{ds_name}"));
    let count = checkpoint_count(&run_dir);

    if stdout.contains("Early stopping triggered") {
        assert!(
            count >= 2,
            "early stopping fired but no final checkpoint was written (got {count} checkpoints)"
        );
    } else {
        assert!(count >= 1, "expected at least 1 checkpoint, got {count}");
    }
}

#[test]
fn no_duplicate_checkpoint_when_early_stop_fires_on_interval_boundary() {
    // When interval=1, every epoch is an interval boundary.
    // The bug writes one checkpoint via the interval block AND a second via the
    // early-stopping flush in the same iteration → duplicate entry.
    //
    // SGD + lr=0.5 + patience=1 reliably triggers early stopping at epoch 3
    // (validated with seed=42, 100 samples, ring distribution).
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();

    nrn(
        dir,
        &[
            "synth",
            "--min",
            "0",
            "--max",
            "10",
            "--seed",
            "42",
            "--distribution",
            "ring",
            "--clusters",
            "2",
            "--samples",
            "100",
        ],
    )
    .success();

    let ds_name = "ring-seed42-c2-f2-n100";

    let out = Command::cargo_bin("nrn")
        .unwrap()
        .current_dir(dir)
        .args([
            "train",
            "start",
            "--seed",
            "42",
            ds_name,
            "--epochs",
            "20",
            "--checkpoint-interval",
            "1",
            "--early-stopping",
            "1",
            "--optimizer",
            "sgd",
            "--lr",
            "0.5",
        ])
        .output()
        .unwrap();
    assert!(out.status.success());

    let stdout = String::from_utf8_lossy(&out.stdout);

    // Find "Early stopping triggered at epoch N" and parse N.
    let stop_epoch = stdout.lines().find_map(|l| {
        l.contains("Early stopping triggered at epoch")
            .then(|| l.split_whitespace().last()?.parse::<usize>().ok())?
    });

    if let Some(epoch) = stop_epoch {
        let run_dir = dir.join(format!("training-model-{ds_name}"));
        let count = checkpoint_count(&run_dir);
        // Correct: initial (1) + one per completed epoch = epoch + 1.
        // Bug:     initial + epoch + duplicate from early-stop flush = epoch + 2.
        assert_eq!(
            count,
            epoch + 1,
            "expected {} checkpoints (initial + {epoch} interval), got {count}",
            epoch + 1
        );
    }
}

fn model_exists(dir: &Path, ds_name: &str) -> bool {
    dir.join(format!("model-{ds_name}"))
        .join("model.safetensors")
        .exists()
}

#[test]
fn start_prints_recap_without_overrides() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();

    nrn(
        dir,
        &[
            "synth",
            "--min",
            "0",
            "--max",
            "10",
            "--seed",
            "1",
            "--distribution",
            "ring",
            "--clusters",
            "2",
            "--samples",
            "20",
        ],
    )
    .success();

    let ds_name = "ring-seed1-c2-f2-n20";

    let out = Command::cargo_bin("nrn")
        .unwrap()
        .current_dir(dir)
        .args([
            "train",
            "start",
            "--seed",
            "42",
            ds_name,
            "--epochs",
            "2",
            "--checkpoint-interval",
            "1",
            "--no-clip",
        ])
        .output()
        .unwrap();
    assert!(out.status.success());

    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("TRAINING HYPERPARAMETERS"),
        "expected hyperparameters recap in stdout: {stdout}"
    );
    assert!(
        !stdout.contains("was"),
        "a fresh run must not show any override marker: {stdout}"
    );
}

#[test]
fn resume_with_lr_override_shows_marker_on_optimizer_line() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();

    nrn(
        dir,
        &[
            "synth",
            "--min",
            "0",
            "--max",
            "10",
            "--seed",
            "2",
            "--distribution",
            "ring",
            "--clusters",
            "2",
            "--samples",
            "20",
        ],
    )
    .success();

    let ds_name = "ring-seed2-c2-f2-n20";

    nrn(
        dir,
        &[
            "train",
            "start",
            "--seed",
            "42",
            ds_name,
            "--epochs",
            "2",
            "--checkpoint-interval",
            "1",
            "--no-clip",
            "--lr",
            "0.001",
        ],
    )
    .success();

    let run_arg = format!("training-model-{ds_name}");

    let out = Command::cargo_bin("nrn")
        .unwrap()
        .current_dir(dir)
        .args(["train", "resume", &run_arg, "--epochs", "3", "--lr", "0.01"])
        .output()
        .unwrap();
    assert!(out.status.success());

    let stdout = String::from_utf8_lossy(&out.stdout);
    let optimizer_line = stdout
        .lines()
        .find(|line| line.contains("Optimizer"))
        .unwrap_or_default();
    assert!(
        optimizer_line.contains("was") && optimizer_line.contains("0.001"),
        "expected the Optimizer recap line to mark the previous lr: {optimizer_line}"
    );
}

#[test]
fn resume_restores_adam_optimizer_state() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();

    nrn(
        dir,
        &[
            "synth",
            "--min",
            "0",
            "--max",
            "10",
            "--seed",
            "2",
            "--distribution",
            "ring",
            "--clusters",
            "2",
            "--samples",
            "20",
        ],
    )
    .success();

    let ds_name = "ring-seed2-c2-f2-n20";

    nrn(
        dir,
        &[
            "train",
            "start",
            "--seed",
            "42",
            ds_name,
            "--epochs",
            "2",
            "--checkpoint-interval",
            "1",
            "--no-clip",
            "--lr",
            "0.001",
        ],
    )
    .success();

    let run_arg = format!("training-model-{ds_name}");

    nrn(dir, &["train", "resume", &run_arg, "--epochs", "3"])
        .success()
        .stdout(contains("Restored Adam optimizer state"));
}

#[test]
fn resume_rejects_val_ratio_override() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();

    nrn(
        dir,
        &[
            "synth",
            "--min",
            "0",
            "--max",
            "10",
            "--seed",
            "3",
            "--distribution",
            "ring",
            "--clusters",
            "2",
            "--samples",
            "20",
        ],
    )
    .success();

    let ds_name = "ring-seed3-c2-f2-n20";

    nrn(
        dir,
        &[
            "train",
            "start",
            "--seed",
            "42",
            ds_name,
            "--epochs",
            "2",
            "--checkpoint-interval",
            "1",
            "--no-clip",
        ],
    )
    .success();

    let run_arg = format!("training-model-{ds_name}");

    nrn(dir, &["train", "resume", &run_arg, "--val-ratio", "0.2"])
        .failure()
        .stderr(contains("unexpected argument"));
}

#[test]
fn divergence_with_early_stopping_recovers_best_model() {
    // lr=5.0 + no-clip causes divergence at epoch 2-3 for any He-initialized model.
    // With --early-stopping + restore_best_model (default), the best model seen before
    // divergence must be saved instead of erroring out.
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();

    nrn(
        dir,
        &[
            "synth",
            "--min",
            "0",
            "--max",
            "10",
            "--seed",
            "42",
            "--distribution",
            "ring",
            "--clusters",
            "2",
            "--samples",
            "40",
        ],
    )
    .success();

    let ds_name = "ring-seed42-c2-f2-n40";
    let out = Command::cargo_bin("nrn")
        .unwrap()
        .current_dir(dir)
        .args([
            "train",
            "start",
            "--seed",
            "42",
            ds_name,
            "--epochs",
            "50",
            "--no-clip",
            "--lr",
            "5.0",
            "--early-stopping",
            "20",
            "--checkpoint-interval",
            "1000",
            "--layers",
            "4,4",
        ])
        .output()
        .unwrap();

    let stderr = String::from_utf8_lossy(&out.stderr);

    // With --early-stopping + restore_best_model (default), the model must always be
    // saved: recovery kicks in on divergence, normal save on successful training.
    assert!(
        model_exists(dir, ds_name),
        "model should be saved whether training converges or recovers from divergence\nstderr: {stderr}"
    );
    if stderr.contains("diverged") {
        assert!(
            stderr.contains("recovered"),
            "divergence with early stopping should print a recovery warning\nstderr: {stderr}"
        );
    }
}

#[test]
fn synth_warns_on_uneven_clusters_and_plots_scatter() {
    // 21 samples across 2 clusters → 1 dropped (uneven division warning, stderr).
    // The dataset id reflects the *generated* count, and a 2-feature dataset
    // gets an inline scatter preview printed to stdout.
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();

    let out = Command::cargo_bin("nrn")
        .unwrap()
        .current_dir(dir)
        .args([
            "synth",
            "--min",
            "0",
            "--max",
            "10",
            "--seed",
            "1",
            "--distribution",
            "ring",
            "--clusters",
            "2",
            "--samples",
            "21",
        ])
        .output()
        .unwrap();
    assert!(out.status.success());

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("1 dropped"),
        "expected an uneven-cluster warning on stderr: {stderr}"
    );

    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("Scatter Plot of Dataset Features"),
        "expected an inline scatter preview on stdout: {stdout}"
    );

    let ds_name = "ring-seed1-c2-f2-n20";
    assert!(
        dir.join(format!("{ds_name}.safetensors")).exists(),
        "expected dataset named after the generated count"
    );
}

#[test]
fn start_with_auto_layers_infers_architecture() {
    // --auto-layers derives the hidden layers from the dataset shape instead of
    // taking explicit --layers; the recap reports the inferred architecture.
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();

    nrn(
        dir,
        &[
            "synth",
            "--min",
            "0",
            "--max",
            "10",
            "--seed",
            "1",
            "--distribution",
            "ring",
            "--clusters",
            "2",
            "--samples",
            "20",
        ],
    )
    .success();

    let ds_name = "ring-seed1-c2-f2-n20";

    let out = Command::cargo_bin("nrn")
        .unwrap()
        .current_dir(dir)
        .args([
            "train",
            "start",
            "--seed",
            "42",
            ds_name,
            "--epochs",
            "2",
            "--checkpoint-interval",
            "0",
            "--no-clip",
            "--auto-layers",
        ])
        .output()
        .unwrap();
    assert!(out.status.success());

    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("relu") && stdout.contains("sigmoid"),
        "expected an inferred hidden-layer architecture in the recap: {stdout}"
    );
    assert!(model_exists(dir, ds_name));
}

#[test]
fn resume_restores_stateful_scheduler() {
    // A run trained with a stateful scheduler (step decay) persists scheduler
    // state in its checkpoints; resuming must reinstate it and say so. The
    // default constant scheduler is stateless, so this path is otherwise unhit.
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();

    nrn(
        dir,
        &[
            "synth",
            "--min",
            "0",
            "--max",
            "10",
            "--seed",
            "5",
            "--distribution",
            "ring",
            "--clusters",
            "2",
            "--samples",
            "20",
        ],
    )
    .success();

    let ds_name = "ring-seed5-c2-f2-n20";

    nrn(
        dir,
        &[
            "train",
            "start",
            "--seed",
            "42",
            ds_name,
            "--epochs",
            "3",
            "--checkpoint-interval",
            "1",
            "--no-clip",
            "--scheduler",
            "step",
            "--steps",
            "2",
        ],
    )
    .success();

    let run_arg = format!("training-model-{ds_name}");

    nrn(dir, &["train", "resume", &run_arg, "--epochs", "2"])
        .success()
        .stdout(contains("Restored Step Decay scheduler state"));
}

#[test]
fn resume_with_checkpoints_disabled_writes_no_new_checkpoints() {
    // Resuming with --checkpoint-interval 0 trains forward without a recorder, so
    // the checkpoint trajectory is left untouched.
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();

    nrn(
        dir,
        &[
            "synth",
            "--min",
            "0",
            "--max",
            "10",
            "--seed",
            "6",
            "--distribution",
            "ring",
            "--clusters",
            "2",
            "--samples",
            "20",
        ],
    )
    .success();

    let ds_name = "ring-seed6-c2-f2-n20";
    let run_arg = format!("training-model-{ds_name}");
    let run_dir = dir.join(&run_arg);

    nrn(
        dir,
        &[
            "train",
            "start",
            "--seed",
            "42",
            ds_name,
            "--epochs",
            "2",
            "--checkpoint-interval",
            "1",
            "--no-clip",
        ],
    )
    .success();
    let before = checkpoint_count(&run_dir);

    nrn(
        dir,
        &[
            "train",
            "resume",
            &run_arg,
            "--epochs",
            "2",
            "--checkpoint-interval",
            "0",
        ],
    )
    .success();

    assert_eq!(
        checkpoint_count(&run_dir),
        before,
        "resuming with checkpoints disabled must not add or remove checkpoints"
    );
}
