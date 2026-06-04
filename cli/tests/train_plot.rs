use assert_cmd::Command;
use predicates::str::contains;
use std::fs;
use std::path::Path;

fn snapshot_count(history_dir: &std::path::Path) -> usize {
    fs::read_dir(history_dir)
        .map(|rd| {
            rd.filter_map(Result::ok)
                .filter(|e| {
                    let name = e.file_name();
                    name.to_str()
                        .map(|n| n.starts_with("snapshot-"))
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
fn train_creates_history_dir_and_plot_generates_png_and_gif() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();

    // Generate a 2-class, 2-feature ring dataset (20 samples for speed).
    nrn(
        dir,
        &[
            "synth",
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

    // Dataset filename produced by synth: ring-c2-f2-n20-seed42
    let ds_name = "ring-c2-f2-n20-seed42";

    // Train with checkpoints every 5 epochs (20 epochs total → ≥ 5 snapshots).
    nrn(
        dir,
        &[
            "train",
            "start",
            ds_name,
            "--epochs",
            "20",
            "--checkpoint-interval",
            "5",
            "--no-clip",
        ],
    )
    .success();

    // Training history directory must exist.
    let history_dir = dir.join(format!("training-model-{ds_name}"));
    assert!(
        history_dir.is_dir(),
        "expected training history dir at {history_dir:?}"
    );

    // Must have more than 2 snapshot files (guard from load_history).
    let count = snapshot_count(&history_dir);
    assert!(count > 2, "expected >2 snapshots, got {count}");

    let history_arg = format!("training-model-{ds_name}");

    // Plot training curves → PNG.
    nrn(dir, &["plot", &history_arg]).success();

    let png = dir.join(format!("training-model-{ds_name}.png"));
    assert!(png.exists(), "expected PNG at {png:?}");

    // Plot with decision boundary → GIF.
    nrn(dir, &["plot", &history_arg, "--dataset", ds_name]).success();

    let gif = dir.join(format!("training-model-{ds_name}.gif"));
    assert!(gif.exists(), "expected GIF at {gif:?}");
}

#[test]
fn load_history_rejects_too_few_snapshots() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();

    // interval=100 with epochs=2 → only the initial snapshot is written.
    nrn(
        dir,
        &[
            "synth",
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

    let ds_name = "ring-c2-f2-n20-seed7";

    nrn(
        dir,
        &[
            "train",
            "start",
            ds_name,
            "--epochs",
            "2",
            "--checkpoint-interval",
            "100",
            "--no-clip",
        ],
    )
    .success();

    let history_arg = format!("training-model-{ds_name}");
    nrn(dir, &["plot", &history_arg])
        .failure()
        .stderr(contains("more than two snapshots"));
}

#[test]
fn early_stopping_writes_final_snapshot() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();

    // 40 samples → 4 validation samples with default val_ratio=0.1.
    nrn(
        dir,
        &[
            "synth",
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

    let ds_name = "ring-c2-f2-n40-seed99";

    // interval=1000 >> epochs=30: only initial + final-epoch snapshot would be
    // written without early stopping. With patience=1 this fires quickly and
    // the fix must write a snapshot at the stopped epoch.
    let out = Command::cargo_bin("nrn")
        .unwrap()
        .current_dir(dir)
        .args([
            "train",
            "start",
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
    let history_dir = dir.join(format!("training-model-{ds_name}"));
    let count = snapshot_count(&history_dir);

    if stdout.contains("Early stopping triggered") {
        assert!(
            count >= 2,
            "early stopping fired but no final snapshot was written (got {count} snapshots)"
        );
    } else {
        assert!(count >= 1, "expected at least 1 snapshot, got {count}");
    }
}

#[test]
fn no_duplicate_snapshot_when_early_stop_fires_on_interval_boundary() {
    // When interval=1, every epoch is an interval boundary.
    // The bug writes one snapshot via the interval block AND a second via the
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

    let ds_name = "ring-c2-f2-n100-seed42";

    let out = Command::cargo_bin("nrn")
        .unwrap()
        .current_dir(dir)
        .args([
            "train",
            "start",
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
        let history_dir = dir.join(format!("training-model-{ds_name}"));
        let count = snapshot_count(&history_dir);
        // Correct: initial (1) + one per completed epoch = epoch + 1.
        // Bug:     initial + epoch + duplicate from early-stop flush = epoch + 2.
        assert_eq!(
            count,
            epoch + 1,
            "expected {} snapshots (initial + {epoch} interval), got {count}",
            epoch + 1
        );
    }
}

fn model_exists(dir: &Path, ds_name: &str) -> bool {
    dir.join(format!("model-{ds_name}.safetensors")).exists()
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

    let ds_name = "ring-c2-f2-n40-seed42";
    let out = Command::cargo_bin("nrn")
        .unwrap()
        .current_dir(dir)
        .args([
            "train",
            "start",
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
