mod cross_entropy_loss;

pub use cross_entropy_loss::{CROSS_ENTROPY_LOSS, CrossEntropyLoss};

use ndarray::{Array2, ArrayView2};

pub trait LossFunction: Send + Sync {
    /// Returns a human-readable name for this loss function.
    fn name(&self) -> &'static str;

    /// Computes the loss given the predicted and true values.
    ///
    /// # Arguments
    ///
    /// * `predictions` - A 2D array where each row represents a predicted output for a sample.
    /// * `targets` - A 2D array where each row represents the expected (true) output for a sample.
    fn compute(&self, predictions: ArrayView2<f32>, targets: ArrayView2<f32>) -> f32;

    /// Computes the gradient of the loss with respect to the predicted values.
    ///
    /// Assumes `predictions` has already been passed through [`stabilize`](Self::stabilize).
    ///
    /// # Arguments
    /// * `predictions` - A 2D array where each row represents a predicted output for a sample.
    /// * `targets` - A 2D array where each row represents the expected (true) output for a sample.
    fn gradient(&self, predictions: ArrayView2<f32>, targets: ArrayView2<f32>) -> Array2<f32>;

    /// Projects `predictions` into the numerically safe domain that `compute` and
    /// `gradient` assume. The default is the identity; a loss whose domain is singular
    /// (e.g. cross-entropy's `ln` and `p(1 − p)` denominator) overrides it.
    ///
    /// Kept separate from `gradient` so the caller can feed the *same* stabilized values
    /// to the output activation's `vjp`, making their product cancel exactly.
    fn stabilize(&self, predictions: ArrayView2<f32>) -> Array2<f32> {
        predictions.to_owned()
    }
}
