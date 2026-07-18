//! Helpers shared across `core`'s own unit tests (not visible outside the crate).

use crate::activations::{IDENTITY, RELU};
use crate::data::scalers::ScalerMethod;
use crate::layers::Dense;
use crate::model::{ModelConfig, NetworkConfig, NeuralNetwork, Predictor};
use crate::task::Task;
use ndarray::{Array2, array};

/// A batch of 5 samples of 3 features apiece, values counting up from `0.0` in steps of `0.1`.
pub(crate) fn sample_batch() -> Array2<f32> {
    Array2::from_shape_fn((3, 5), |(i, j)| (i * 5 + j) as f32 * 0.1)
}

impl Dense {
    /// A 2-input, 3-unit ReLU hidden layer whose middle unit's negative pre-activation is
    /// clamped to zero — a dead neuron, exercised by activation-diagram tests.
    pub(crate) fn dead_relu_hidden_layer() -> Self {
        Self::new(
            array![[1.0, 0.0], [-1.0, 0.0], [2.0, 0.0]],
            array![0.0, 0.0, 0.0],
            RELU.clone(),
        )
    }
}

impl NeuralNetwork {
    /// A 3-input regression network (4-unit ReLU hidden layer, 1-unit identity output), for
    /// weights/architecture round-trip tests.
    pub(crate) fn hidden_dense_regression() -> Self {
        let config = NetworkConfig::builder(vec![3])
            .dense(4, &RELU)
            .dense(1, &IDENTITY)
            .build();
        Self::from_config(config, 0).unwrap()
    }
}

impl Predictor {
    /// A predictor for `task` with no labels, the shape most unit tests need.
    pub(crate) fn unlabeled(
        network: NeuralNetwork,
        task: Task,
        scaler: Option<ScalerMethod>,
    ) -> Self {
        Self::new(
            network,
            ModelConfig::new(task, None).expect("None labels always satisfy ModelConfig::new"),
            scaler,
        )
    }

    /// A 2-input binary predictor whose logit is `x0` (weights `[1, 0]`, zero bias, identity
    /// output): its decision boundary sits exactly on `x0 == 0`.
    pub(crate) fn binary() -> Self {
        Self::unlabeled(
            NeuralNetwork::single(Dense::new(
                array![[1.0, 0.0]],
                array![0.0],
                IDENTITY.clone(),
            )),
            Task::Binary,
            None,
        )
    }

    /// A 2-input → 3-output multi-class predictor with a linear output and symmetric weights for
    /// two of the classes, so a tie (equal top-two softmax probabilities) occurs along `x0 == 0`.
    pub(crate) fn multiclass() -> Self {
        Self::unlabeled(
            NeuralNetwork::single(Dense::new(
                array![[1.0, 0.0], [-1.0, 0.0], [0.0, 0.0]],
                array![0.0, 0.0, -10.0],
                IDENTITY.clone(),
            )),
            Task::MultiClass { class_count: 3 },
            None,
        )
    }

    /// An unlabeled binary predictor over [`NeuralNetwork::hidden_dense_regression`], the shape
    /// most save/load round-trip tests need.
    pub(crate) fn hidden_dense_regression() -> Self {
        Self::unlabeled(NeuralNetwork::hidden_dense_regression(), Task::Binary, None)
    }
}
