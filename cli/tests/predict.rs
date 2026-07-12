//! End-to-end coverage of the `predict` command: ranking output from stdin and
//! from an instance file, plus the interactive-input guards (premature EOF, an
//! unparseable line) and the instance/model dimension check.
//!
//! Fixtures are minted through the library into a temp dir created *under* the
//! crate directory, so both this process and the `nrn` subprocess resolve them
//! within the path-safety boundary (paths outside the cwd are rejected).

use assert_cmd::Command;
use ndarray::{Array1, array};
use nrn::activations::RELU;
use nrn::data::Instance;
use nrn::data::scalers::{MinMaxScaler, ScalerMethod};
use nrn::model::{LayerPlan, NeuralNetwork, NeuronLayerSpec, Predictor};
use nrn::task::Task;
use predicates::str::contains;
use std::path::Path;
use tempfile::TempDir;

/// A temp dir under the crate directory (within the path-safety boundary).
fn workspace() -> TempDir {
    TempDir::new_in(".").unwrap()
}

/// Writes a binary `n_features`-input predictor to `dir/model`, with an optional
/// fitted scaler sidecar.
fn write_predictor(dir: &Path, n_features: usize, scaler: Option<ScalerMethod>) {
    let specs = NeuronLayerSpec::plan(LayerPlan::Explicit(vec![4]), 2, &*RELU).unwrap();
    let network = NeuralNetwork::initialization(n_features, &specs, 7);
    Predictor::new(network, Task::Binary, scaler)
        .save(dir.join("model"))
        .unwrap();
}

/// A min-max scaler fitted over two features, for the scaler-present path.
fn two_feature_scaler() -> ScalerMethod {
    ScalerMethod::MinMax(MinMaxScaler::default().fit(array![[0.0, 0.0], [1.0, 1.0]].view()))
}

/// A fresh `nrn` invocation rooted at `dir`.
fn nrn(dir: &Path) -> Command {
    let mut cmd = Command::cargo_bin("nrn").unwrap();
    cmd.current_dir(dir);
    cmd
}

#[test]
fn ranks_classes_from_stdin() {
    let tmp = workspace();
    write_predictor(tmp.path(), 2, None);

    nrn(tmp.path())
        .args(["predict", "model"])
        .write_stdin("0.3\n0.7\n")
        .assert()
        .success()
        .stdout(contains("CLASSIFICATION"))
        .stdout(contains("Class 0"))
        .stdout(contains("Class 1"));
}

#[test]
fn activations_flag_prints_the_diagram_above_the_classification() {
    let tmp = workspace();
    write_predictor(tmp.path(), 2, None);

    nrn(tmp.path())
        .args(["predict", "model", "--activations"])
        .write_stdin("0.3\n0.7\n")
        .assert()
        .success()
        // The forward-pass diagram,
        .stdout(contains("Input (2 features)"))
        // then the ranked classification with the winning class arrow-marked.
        .stdout(contains("CLASSIFICATION"))
        .stdout(contains('\u{25c0}'));
}

#[test]
fn errors_on_premature_end_of_input() {
    let tmp = workspace();
    write_predictor(tmp.path(), 2, None);

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
    write_predictor(tmp.path(), 2, None);

    nrn(tmp.path())
        .args(["predict", "model"])
        .write_stdin("oops\n0.3\n0.7\n")
        .assert()
        .success()
        .stderr(contains("invalid float literal"))
        .stdout(contains("CLASSIFICATION"));
}

#[test]
fn reads_an_instance_from_a_file() {
    let tmp = workspace();
    write_predictor(tmp.path(), 2, None);
    Instance::new(array![0.3, 0.7])
        .save(tmp.path().join("sample"))
        .unwrap();

    nrn(tmp.path())
        .args(["predict", "model", "--instance", "sample"])
        .assert()
        .success()
        .stdout(contains("INSTANCE LOADED"))
        .stdout(contains("CLASSIFICATION"));
}

#[test]
fn errors_when_instance_dimension_mismatches_model() {
    let tmp = workspace();
    write_predictor(tmp.path(), 2, None);
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
    write_predictor(tmp.path(), 2, Some(two_feature_scaler()));

    nrn(tmp.path())
        .args(["predict", "model"])
        .write_stdin("0.3\n0.7\n")
        .assert()
        .success()
        .stdout(contains("PREDICTOR LOADED"))
        .stdout(contains("min-max"))
        .stdout(contains("CLASSIFICATION"));
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
    write_predictor(tmp.path(), 2, None);

    nrn(tmp.path())
        .args(["predict", "model", "--instance", "missing"])
        .assert()
        .failure();
}
