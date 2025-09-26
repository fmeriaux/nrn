mod cross_entropy_loss;

pub use cross_entropy_loss::CROSS_ENTROPY_LOSS;

use ndarray::{Array2, ArrayView2};

pub trait LossFunction: Send + Sync {
    /// Computes the loss given the predicted and true values.
    ///
    /// # Arguments
    ///
    /// * `predictions` - A 2D array where each row represents a predicted output for a sample.
    /// * `expectations` - A 2D array where each row represents the expected (true) output for a sample.
    fn compute(&self, predictions: ArrayView2<f32>, expectations: ArrayView2<f32>) -> f32;

    /// Computes the gradient of the loss with respect to the predicted values.
    ///
    /// # Arguments
    /// * `predictions` - A 2D array where each row represents a predicted output for a sample.
    /// * `expectations` - A 2D array where each row represents the expected (true) output for a sample.
    fn gradient(&self, predictions: ArrayView2<f32>, expectations: ArrayView2<f32>) -> Array2<f32>;
}
