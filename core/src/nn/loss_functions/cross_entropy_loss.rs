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
    /// Distance kept from the open ends of `(0, 1)` when clamping probabilities.
    ///
    /// Sized for `f32`: the largest float below `1.0` is `1 − 2⁻²⁴ ≈ 1 − 6e-8`, so an
    /// epsilon any smaller than that makes `1.0 − epsilon` round straight back to `1.0`
    /// and the upper clamp a no-op. A saturated sigmoid output of exactly `1.0` would
    /// then drive the binary-CE gradient's `p(1 − p)` denominator to zero — `NaN`/`inf`.
    pub(crate) const PROBABILITY_EPSILON: f32 = 1e-7;
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
        let clipped = self.stabilize(predictions);
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
    /// Expects `predictions` to be clipped to a safe interior of (0, 1) by the caller
    /// (via [`stabilize`](LossFunction::stabilize)); no clipping is applied here so that
    /// the caller can pass identical values to the activation's `vjp`, ensuring exact
    /// compositional cancellation.
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

    /// Clips the predicted probabilities into the open interval `(0, 1)` so that `ln(p)`
    /// stays finite and the binary gradient's `p(1 − p)` denominator never reaches zero.
    fn stabilize(&self, predictions: ArrayView2<f32>) -> Array2<f32> {
        predictions.mapv(|p| p.clamp(Self::PROBABILITY_EPSILON, 1.0 - Self::PROBABILITY_EPSILON))
    }
}

/// Static instance of the CrossEntropyLoss wrapped in an `Arc` for shared use.
pub static CROSS_ENTROPY_LOSS: Lazy<Arc<CrossEntropyLoss>> =
    Lazy::new(|| Arc::new(CrossEntropyLoss));

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::array;

    #[test]
    fn name_is_cross_entropy() {
        assert_eq!(CrossEntropyLoss.name(), "Cross-Entropy");
    }

    #[test]
    fn clip_keeps_saturated_probabilities_off_the_open_ends() {
        // Regression: a saturated 0.0/1.0 must land strictly inside (0, 1).
        let clipped = CrossEntropyLoss.stabilize(array![[0.0, 1.0]].view());
        assert!(clipped.iter().all(|&p| p > 0.0 && p < 1.0), "{clipped:?}");
    }

    #[test]
    fn saturated_binary_predictions_give_finite_loss_and_gradient() {
        // Correctly classified but saturated: loss ≈ 0 and gradient ≈ 0, not NaN/inf.
        let predictions = array![[1.0, 0.0]];
        let targets = array![[1.0, 0.0]];
        let loss = CrossEntropyLoss.compute(predictions.view(), targets.view());
        assert!(loss.is_finite() && loss >= 0.0, "loss = {loss}");

        // gradient() expects pre-clipped inputs, mirroring the backward pass.
        let safe = CrossEntropyLoss.stabilize(predictions.view());
        let grad = CrossEntropyLoss.gradient(safe.view(), targets.view());
        assert!(grad.iter().all(|v| v.is_finite()), "grad = {grad:?}");
    }
}
