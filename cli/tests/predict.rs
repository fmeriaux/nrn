//! End-to-end coverage of the `predict` command: ranking output from stdin and
//! from an instance file, plus the interactive-input guards (premature EOF, an
//! unparseable line) and the instance/model dimension check.
//!
//! Fixtures are minted through the library into a temp dir created *under* the
//! crate directory, so both this process and the `nrn` subprocess resolve them
//! within the path-safety boundary (paths outside the cwd are rejected).

mod common;

use common::{nrn, two_feature_network, workspace, write_predictor};
use ndarray::{Array1, array};
use nrn::data::Instance;
use nrn::data::scalers::{MinMaxScaler, ScalerMethod};
use nrn::model::{Labels, ModelConfig, Predictor};
use nrn::task::Task;
use predicates::prelude::PredicateBooleanExt;
use predicates::str::contains;
use std::path::Path;

/// Writes a labeled 2-class predictor (`cat`/`dog`) to `dir/model`.
fn write_labeled_predictor(dir: &Path) {
    let labels = Labels::new(vec!["cat".to_string(), "dog".to_string()]);
    Predictor::new(
        two_feature_network(),
        ModelConfig::new(Task::Binary, Some(labels)).unwrap(),
        None,
    )
    .save(dir.join("model"))
    .unwrap();
}

/// Writes a single-output regression predictor to `dir/model`.
fn write_regression_predictor(dir: &Path) {
    Predictor::new(
        two_feature_network(),
        ModelConfig::new(
            Task::Regression {
                target_shape: vec![1],
            },
            None,
        )
        .unwrap(),
        None,
    )
    .save(dir.join("model"))
    .unwrap();
}

/// A min-max scaler fitted over two features, for the scaler-present path.
fn two_feature_scaler() -> ScalerMethod {
    ScalerMethod::MinMax(MinMaxScaler::default().fit(array![[0.0, 0.0], [1.0, 1.0]].view()))
}

#[test]
fn ranks_classes_from_stdin() {
    let tmp = workspace();
    write_predictor(tmp.path(), None);

    nrn(tmp.path())
        .args(["predict", "model"])
        .write_stdin("0.3\n0.7\n")
        .assert()
        .success()
        .stdout(contains("PREDICTION"))
        .stdout(contains("Class 0"))
        .stdout(contains("Class 1"));
}

#[test]
fn activations_flag_prints_the_diagram_above_the_prediction() {
    let tmp = workspace();
    write_predictor(tmp.path(), None);

    nrn(tmp.path())
        .args(["predict", "model", "--activations"])
        .write_stdin("0.3\n0.7\n")
        .assert()
        .success()
        // The forward-pass diagram,
        .stdout(contains("Input (2 features)"))
        // then the ranked prediction with the winning class arrow-marked.
        .stdout(contains("PREDICTION"))
        .stdout(contains('\u{25c0}'));
}

#[test]
fn errors_on_premature_end_of_input() {
    let tmp = workspace();
    write_predictor(tmp.path(), None);

    nrn(tmp.path())
        .args(["predict", "model"])
        .write_stdin("0.3\n")
        .assert()
        .failure()
        .stderr(contains("unexpected end of input"));
}

#[test]
fn recovers_after_an_unparseable_line() {
    let tmp = workspace();
    write_predictor(tmp.path(), None);

    nrn(tmp.path())
        .args(["predict", "model"])
        .write_stdin("oops\n0.3\n0.7\n")
        .assert()
        .success()
        .stderr(contains("invalid float literal"))
        .stdout(contains("PREDICTION"));
}

#[test]
fn reads_an_instance_from_a_file() {
    let tmp = workspace();
    write_predictor(tmp.path(), None);
    Instance::new(array![0.3, 0.7])
        .save(tmp.path().join("sample"))
        .unwrap();

    nrn(tmp.path())
        .args(["predict", "model", "--instance", "sample"])
        .assert()
        .success()
        .stdout(contains("INSTANCE LOADED"))
        .stdout(contains("PREDICTION"));
}

#[test]
fn errors_when_instance_dimension_mismatches_model() {
    let tmp = workspace();
    write_predictor(tmp.path(), None);
    Instance::new(Array1::zeros(3))
        .save(tmp.path().join("sample"))
        .unwrap();

    nrn(tmp.path())
        .args(["predict", "model", "--instance", "sample"])
        .assert()
        .failure()
        .stderr(contains("expects [2]"));
}

#[test]
fn loads_and_applies_a_scaler_sidecar() {
    let tmp = workspace();
    write_predictor(tmp.path(), Some(two_feature_scaler()));

    nrn(tmp.path())
        .args(["predict", "model"])
        .write_stdin("0.3\n0.7\n")
        .assert()
        .success()
        .stdout(contains("PREDICTOR LOADED"))
        .stdout(contains("min-max"))
        .stdout(contains("PREDICTION"));
}

#[test]
fn errors_when_the_model_is_missing() {
    let tmp = workspace();

    nrn(tmp.path())
        .args(["predict", "model"])
        .assert()
        .failure();
}

#[test]
fn errors_when_the_input_file_is_missing() {
    let tmp = workspace();
    write_predictor(tmp.path(), None);

    nrn(tmp.path())
        .args(["predict", "model", "--instance", "missing"])
        .assert()
        .failure();
}

#[test]
fn names_classes_from_the_models_labels() {
    let tmp = workspace();
    write_labeled_predictor(tmp.path());

    nrn(tmp.path())
        .args(["predict", "model"])
        .write_stdin("0.3\n0.7\n")
        .assert()
        .success()
        .stdout(contains("cat"))
        .stdout(contains("dog"))
        .stdout(contains("Class").not());
}

#[test]
fn renders_a_regression_predictors_output() {
    let tmp = workspace();
    write_regression_predictor(tmp.path());

    nrn(tmp.path())
        .args(["predict", "model"])
        .write_stdin("0.3\n0.7\n")
        .assert()
        .success()
        .stdout(contains("PREDICTION"))
        .stdout(contains("Output 0"));
}
