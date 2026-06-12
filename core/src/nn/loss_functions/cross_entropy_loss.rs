//! Cross entropy loss module for binary and multi-class classification.
//!
//! This module defines the `LossFunction` trait and a cross-entropy
//! implementation usable for classification tasks.
//!
//! The loss computes the difference between predicted probabilities
//! and true one-hot labels, providing both the scalar loss and its gradient.

use crate::loss_functions::LossFunction;
use ndarray::{Array2, ArrayView2};
use once_cell::sync::Lazy;
use std::sync::Arc;

pub struct CrossEntropyLoss;

impl CrossEntropyLoss {
    /// Clips the predicted probabilities so that they lie within a safe numerical range
    /// to prevent invalid logarithms during loss computation.
    fn clip_probabilities(probabilities: &ArrayView2<f32>) -> Array2<f32> {
        let epsilon = 1e-15;
        probabilities.mapv(|p| p.clamp(epsilon, 1.0 - epsilon))
    }
}

impl LossFunction for CrossEntropyLoss {
    fn name(&self) -> &'static str {
        "Cross-Entropy"
    }

    /// Computes the scalar loss value given predictions and true labels.
    ///
    /// # Arguments
    /// * `predictions` - 2D array where each row is the predicted probability distribution for a sample.
    /// * `targets` - 2D array where each row is the true one-hot encoded label for a sample.
    ///
    /// # Returns
    /// Average loss over the batch.
    fn compute(&self, predictions: ArrayView2<f32>, targets: ArrayView2<f32>) -> f32 {
        let clipped = Self::clip_probabilities(&predictions);
        let n_samples = predictions.ncols() as f32;
        let log_p = clipped.mapv(|p| p.ln());

        if predictions.nrows() == 1 {
            // Binary (1 sigmoid output): both terms needed since y can be 0
            let log_1_p = clipped.mapv(|p| (1.0 - p).ln());
            -(&targets * &log_p + (1.0 - &targets) * &log_1_p).sum() / n_samples
        } else {
            // Multi-class (softmax): one-hot y is always 1 somewhere, so -(y*log(p)) is complete
            -(&targets * &log_p).sum() / n_samples
        }
    }

    /// Computes ∂L/∂a — the gradient of the loss with respect to the predicted probabilities.
    ///
    /// Expects `predictions` to be clipped to a safe interior of (0, 1) by the caller;
    /// no clipping is applied here so that the caller can pass identical values to the
    /// activation's `vjp`, ensuring exact compositional cancellation.
    ///
    /// # Arguments
    /// * `predictions` - Same as in `compute`.
    /// * `targets` - Same as in `compute`.
    ///
    /// # Returns
    /// Gradient array (`Array2<f32>`) of the same shape as predictions.
    /// Binary CE: `(p - y) / (p * (1 - p))`. Multi-class CE: `-y / p`.
    fn gradient(&self, predictions: ArrayView2<f32>, targets: ArrayView2<f32>) -> Array2<f32> {
        if predictions.nrows() == 1 {
            // Binary CE: ∂L/∂p = (p - y) / (p * (1 - p))
            let p = predictions.to_owned();
            let denom = &p * p.mapv(|x| 1.0 - x);
            (p - targets) / denom
        } else {
            // Multi-class CE: ∂L/∂p_i = -y_i / p_i
            -targets.to_owned() / predictions.to_owned()
        }
    }
}

/// Static instance of the CrossEntropyLoss wrapped in an `Arc` for shared use.
pub static CROSS_ENTROPY_LOSS: Lazy<Arc<CrossEntropyLoss>> =
    Lazy::new(|| Arc::new(CrossEntropyLoss));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_is_cross_entropy() {
        assert_eq!(CrossEntropyLoss.name(), "Cross-Entropy");
    }
}
