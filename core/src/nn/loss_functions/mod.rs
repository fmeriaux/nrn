mod cross_entropy_loss;

pub use cross_entropy_loss::{CROSS_ENTROPY_LOSS, CrossEntropyLoss};

use ndarray::{Array2, ArrayView2};

pub trait LossFunction: Send + Sync {
    /// Returns a human-readable name for this loss function.
    fn name(&self) -> &'static str;

    /// Computes the loss from the network's outputs and the true targets.
    ///
    /// # Arguments
    ///
    /// * `outputs` - A 2D array of the network's outputs, one column per sample.
    /// * `targets` - A 2D array of the expected (true) values, one column per sample.
    fn compute(&self, outputs: ArrayView2<f32>, targets: ArrayView2<f32>) -> f32;

    /// Computes the gradient of the loss with respect to the network's output, ∂L/∂output —
    /// the array the backward pass propagates back from the output layer.
    ///
    /// # Arguments
    /// * `outputs` - A 2D array of the network's outputs, one column per sample.
    /// * `targets` - A 2D array of the expected (true) values, one column per sample.
    fn gradient(&self, outputs: ArrayView2<f32>, targets: ArrayView2<f32>) -> Array2<f32>;
}
