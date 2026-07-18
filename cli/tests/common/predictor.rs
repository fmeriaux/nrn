//! A plain binary predictor for tests that don't care about its weights.

use nrn::activations::{IDENTITY, RELU};
use nrn::data::scalers::ScalerMethod;
use nrn::model::{ModelConfig, NetworkConfig, NeuralNetwork, Predictor};
use nrn::task::Task;
use std::path::Path;

/// A 2-input network (RELU hidden layer, identity output), for tests that don't care about
/// specific weights.
pub fn two_feature_network() -> NeuralNetwork {
    let config = NetworkConfig::builder(vec![2])
        .dense(4, &RELU)
        .dense(1, &IDENTITY)
        .build();
    NeuralNetwork::from_config(config, 7).unwrap()
}

/// Writes a binary predictor to `dir/model`, with an optional fitted scaler sidecar.
pub fn write_predictor(dir: &Path, scaler: Option<ScalerMethod>) {
    Predictor::new(
        two_feature_network(),
        ModelConfig::new(Task::Binary, None).unwrap(),
        scaler,
    )
    .save(dir.join("model"))
    .unwrap();
}
