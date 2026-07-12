//! End-to-end coverage of `plot activations`: the console node list and the
//! node-link image, both from an instance read on stdin. As a pure figure the
//! command renders the forward pass only — the ranked decision stays with
//! `predict`, so it must not appear here.
//!
//! Fixtures are minted through the library into a temp dir created *under* the
//! crate directory, so both this process and the `nrn` subprocess resolve them
//! within the path-safety boundary (paths outside the cwd are rejected).

use assert_cmd::Command;
use nrn::activations::RELU;
use nrn::model::{LayerPlan, NeuralNetwork, NeuronLayerSpec, Predictor};
use nrn::task::Task;
use predicates::prelude::PredicateBooleanExt;
use predicates::str::contains;
use std::path::Path;
use tempfile::TempDir;

/// A temp dir under the crate directory (within the path-safety boundary).
fn workspace() -> TempDir {
    TempDir::new_in(".").unwrap()
}

/// Writes a binary two-input predictor to `dir/model`.
fn write_predictor(dir: &Path) {
    let specs = NeuronLayerSpec::plan(LayerPlan::Explicit(vec![4]), 2, &*RELU).unwrap();
    let network = NeuralNetwork::initialization(2, &specs, 7);
    Predictor::new(network, Task::Binary, None)
        .save(dir.join("model"))
        .unwrap();
}

/// A fresh `nrn` invocation rooted at `dir`.
fn nrn(dir: &Path) -> Command {
    let mut cmd = Command::cargo_bin("nrn").unwrap();
    cmd.current_dir(dir);
    cmd
}

#[test]
fn console_prints_the_diagram_without_the_ranked_decision() {
    let tmp = workspace();
    write_predictor(tmp.path());

    nrn(tmp.path())
        .args(["plot", "activations", "model"])
        .write_stdin("0.3\n0.7\n")
        .assert()
        .success()
        .stdout(contains("Input (2 features)"))
        // A figure, not a report: the classification ranking belongs to `predict`.
        .stdout(contains("CLASSIFICATION").not());
}

#[test]
fn image_format_writes_a_png() {
    let tmp = workspace();
    write_predictor(tmp.path());

    nrn(tmp.path())
        .args([
            "plot",
            "activations",
            "model",
            "--format",
            "image",
            "--output",
            "diagram",
        ])
        .write_stdin("0.3\n0.7\n")
        .assert()
        .success();

    assert!(tmp.path().join("diagram.png").exists());
}
