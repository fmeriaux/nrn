//! Cross-entropy loss for binary and multi-class classification, computed from **logits**.
//!
//! - multi-class: softmax + categorical cross-entropy, valued via log-sum-exp;
//! - binary (a single logit): sigmoid + binary cross-entropy, valued via the softplus form.

use crate::activations::{Activation, SIGMOID, SOFTMAX};
use crate::loss_functions::LossFunction;
use ndarray::{Array1, Array2, ArrayView2, Axis, Zip};
use once_cell::sync::Lazy;
use std::sync::Arc;

pub struct CrossEntropyLoss;

/// Log-sum-exp of each column, `ln Σ e^{zᵢ}`, computed stably by subtracting the column max.
fn logsumexp_columns(logits: ArrayView2<f32>) -> Array1<f32> {
    logits.map_axis(Axis(0), |col| {
        let max = col.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = col.iter().map(|&z| (z - max).exp()).sum();
        max + sum.ln()
    })
}

impl LossFunction for CrossEntropyLoss {
    fn name(&self) -> &'static str {
        "Cross-Entropy"
    }

    /// Computes the scalar loss value from logits.
    ///
    /// # Arguments
    /// * `logits` - 2D array `(classes, samples)` of the output layer's logits.
    /// * `targets` - 2D array `(classes, samples)` of true one-hot labels.
    ///
    /// # Returns
    /// Average loss over the batch.
    fn compute(&self, logits: ArrayView2<f32>, targets: ArrayView2<f32>) -> f32 {
        let n_samples = logits.ncols() as f32;

        if logits.nrows() == 1 {
            // Binary sigmoid BCE from the logit, via the stable softplus form:
            // L = max(z, 0) − z·y + ln(1 + e^{−|z|}), finite for any logit.
            let per_element = Zip::from(&logits)
                .and(&targets)
                .map_collect(|&z, &y| z.max(0.0) - z * y + (1.0 + (-z.abs()).exp()).ln());
            per_element.sum() / n_samples
        } else {
            // Multi-class softmax CE from logits, via log-sum-exp: per sample,
            // −Σ yᵢ (zᵢ − logsumexp(z)) = logsumexp(z)·Σy − Σ yᵢ zᵢ.
            let lse = logsumexp_columns(logits);
            let y_sum = targets.sum_axis(Axis(0));
            let yz = (&targets * &logits).sum_axis(Axis(0));
            (&lse * &y_sum - &yz).sum() / n_samples
        }
    }

    /// Computes ∂L/∂z — the gradient of the loss with respect to the logits: `p − y`.
    ///
    /// `p` is `sigmoid(z)` for the binary head and `softmax(z)` for the multi-class head.
    ///
    /// # Arguments
    /// * `logits` - Same as in `compute`.
    /// * `targets` - Same as in `compute`.
    ///
    /// # Returns
    /// Gradient array (`Array2<f32>`) of the same shape as `logits`.
    fn gradient(&self, logits: ArrayView2<f32>, targets: ArrayView2<f32>) -> Array2<f32> {
        let probabilities = if logits.nrows() == 1 {
            SIGMOID.apply(logits)
        } else {
            SOFTMAX.apply(logits)
        };
        probabilities - targets
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
    fn binary_gradient_is_sigmoid_of_logit_minus_target() {
        let logits = array![[2.0, -1.0]];
        let targets = array![[1.0, 0.0]];
        let grad = CrossEntropyLoss.gradient(logits.view(), targets.view());
        assert_eq!(grad, SIGMOID.apply(logits.view()) - &targets);
    }

    #[test]
    fn multiclass_gradient_is_softmax_of_logits_minus_target() {
        let logits = array![[2.0, 0.5], [0.1, -1.0], [-0.3, 1.5]];
        let targets = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]];
        let grad = CrossEntropyLoss.gradient(logits.view(), targets.view());
        assert_eq!(grad, SOFTMAX.apply(logits.view()) - &targets);
    }

    #[test]
    fn multiclass_compute_matches_negative_log_softmax() {
        let logits = array![[2.0, 0.5], [0.1, -1.0], [-0.3, 1.5]];
        let targets = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]];
        let probabilities = SOFTMAX.apply(logits.view());
        let n_samples = logits.ncols() as f32;
        let expected = -(&targets * &probabilities.mapv(f32::ln)).sum() / n_samples;

        let loss = CrossEntropyLoss.compute(logits.view(), targets.view());
        assert!((loss - expected).abs() < 1e-5, "loss {loss} vs {expected}");
    }

    #[test]
    fn saturated_logits_give_finite_loss_and_gradient() {
        let logits = array![[1000.0, -1000.0]];
        let targets = array![[1.0, 0.0]];

        let loss = CrossEntropyLoss.compute(logits.view(), targets.view());
        assert!(loss.is_finite() && loss >= 0.0, "loss = {loss}");

        let grad = CrossEntropyLoss.gradient(logits.view(), targets.view());
        assert!(grad.iter().all(|v| v.is_finite()), "grad = {grad:?}");
    }
}
