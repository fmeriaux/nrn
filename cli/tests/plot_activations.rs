//! End-to-end coverage of `plot activations`: the console node list and the
//! node-link image, both from an instance read on stdin. As a pure figure the
//! command renders the forward pass only — the ranked decision stays with
//! `predict`, so it must not appear here.
//!
//! Fixtures are minted through the library into a temp dir created *under* the
//! crate directory, so both this process and the `nrn` subprocess resolve them
//! within the path-safety boundary (paths outside the cwd are rejected).

mod common;

use common::{nrn, workspace, write_predictor};
use predicates::prelude::PredicateBooleanExt;
use predicates::str::contains;

#[test]
fn console_prints_the_diagram_without_the_ranked_decision() {
    let tmp = workspace();
    write_predictor(tmp.path(), None);

    nrn(tmp.path())
        .args(["plot", "activations", "model"])
        .write_stdin("0.3\n0.7\n")
        .assert()
        .success()
        .stdout(contains("Input (2 features)"))
        // A figure, not a report: the ranked prediction belongs to `predict`.
        .stdout(contains("PREDICTION").not());
}

#[test]
fn image_format_writes_a_png() {
    let tmp = workspace();
    write_predictor(tmp.path(), None);

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
