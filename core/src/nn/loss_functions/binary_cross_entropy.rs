//! Cross-entropy loss for binary classification, computed from **logits**.
//! Supports N-Dimensional tensors laid out samples-last (the leading axis is a single logit row).
//!
//! The value uses the numerically stable softplus form `max(z, 0) - z·y + ln(1 + e^-|z|)`,
//! and the gradient is `σ(z) − y`.

use crate::activations::{Activation, SIGMOID};
use crate::loss_functions::{LossFunction, Reduction};
use ndarray::{ArrayD, ArrayViewD, Zip};

/// Binary cross-entropy from logits: a single sigmoid output per position, thresholded
/// against a `0.0`/`1.0` target. The loss is element-wise; [`Reduction`] collapses it.
pub struct BinaryCrossEntropy {
    reduction: Reduction,
}

impl BinaryCrossEntropy {
    /// Builds a binary cross-entropy loss with the given [`Reduction`].
    pub fn new(reduction: Reduction) -> Self {
        Self { reduction }
    }
}

impl LossFunction for BinaryCrossEntropy {
    fn name(&self) -> &'static str {
        "Binary-Cross-Entropy"
    }

    fn reduction(&self) -> Reduction {
        self.reduction
    }

    /// Computes the per-element loss from N-Dimensional logits.
    ///
    /// # Arguments
    /// * `inputs` - N-D array view where Axis(0) is a single logit row.
    /// * `targets` - N-D array view of matching shape with true labels (`0.0` or `1.0`).
    fn terms(&self, inputs: ArrayViewD<f32>, targets: ArrayViewD<f32>) -> ArrayD<f32> {
        // Softplus form per element, numerically stable for logits of either sign.
        Zip::from(&inputs)
            .and(&targets)
            .map_collect(|&z, &y| z.max(0.0) - z * y + (1.0 + (-z.abs()).exp()).ln())
    }

    /// Computes ∂L/∂z — the gradient of the loss with respect to the logits: `σ(z) − y`.
    /// Works out-of-the-box for N-Dimensional tensors.
    fn gradient(&self, inputs: ArrayViewD<f32>, targets: ArrayViewD<f32>) -> ArrayD<f32> {
        SIGMOID.apply(inputs) - targets
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Mean-reduced binary cross-entropy, the default used throughout training.
    fn bce() -> BinaryCrossEntropy {
        BinaryCrossEntropy::new(Reduction::Mean)
    }

    /// Element-wise softplus value, the reference each assertion is checked against.
    fn softplus_bce(z: f32, y: f32) -> f32 {
        z.max(0.0) - z * y + (1.0 + (-z.abs()).exp()).ln()
    }

    #[test]
    fn confident_correct_logits_give_near_zero_loss() {
        let inputs = array![[10.0_f32, -10.0]].into_dyn();
        let targets = array![[1.0_f32, 0.0]].into_dyn();
        let loss = bce().compute(inputs.view(), targets.view());
        assert!((0.0..1e-3).contains(&loss), "loss was {loss}");
    }

    #[test]
    fn mean_averages_over_every_position() {
        let inputs = array![[0.5_f32, -0.7, 1.3]].into_dyn();
        let targets = array![[1.0_f32, 0.0, 1.0]].into_dyn();
        let expected =
            (softplus_bce(0.5, 1.0) + softplus_bce(-0.7, 0.0) + softplus_bce(1.3, 1.0)) / 3.0;
        let loss = BinaryCrossEntropy::new(Reduction::Mean).compute(inputs.view(), targets.view());
        assert!((loss - expected).abs() < 1e-6, "loss was {loss}");
    }

    #[test]
    fn sum_totals_every_position() {
        let inputs = array![[0.5_f32, -0.7, 1.3]].into_dyn();
        let targets = array![[1.0_f32, 0.0, 1.0]].into_dyn();
        let expected = softplus_bce(0.5, 1.0) + softplus_bce(-0.7, 0.0) + softplus_bce(1.3, 1.0);
        let loss = BinaryCrossEntropy::new(Reduction::Sum).compute(inputs.view(), targets.view());
        assert!((loss - expected).abs() < 1e-6, "loss was {loss}");
    }

    #[test]
    fn mean_scores_every_spatial_position() {
        // (class=1, height=2, samples=2): four positions averaged.
        let inputs = array![[[0.5_f32, -0.5], [-1.0, 2.0]]].into_dyn();
        let targets = array![[[1.0_f32, 0.0], [1.0, 0.0]]].into_dyn();
        let expected = (softplus_bce(0.5, 1.0)
            + softplus_bce(-0.5, 0.0)
            + softplus_bce(-1.0, 1.0)
            + softplus_bce(2.0, 0.0))
            / 4.0;
        let loss = bce().compute(inputs.view(), targets.view());
        assert!((loss - expected).abs() < 1e-6, "loss was {loss}");
    }

    #[test]
    fn gradient_is_sigmoid_minus_target() {
        let inputs = array![[0.0_f32, 2.0]].into_dyn();
        let targets = array![[1.0_f32, 0.0]].into_dyn();
        let grad = bce().gradient(inputs.view(), targets.view());
        // σ(0)=0.5 → 0.5-1=-0.5 ; σ(2)=0.8808 → 0.8808-0=0.8808
        let (g0, g1) = (grad[[0, 0]], grad[[0, 1]]);
        assert!((g0 - (-0.5)).abs() < 1e-6, "grad was {g0}");
        assert!((g1 - 0.880797).abs() < 1e-5, "grad was {g1}");
    }

    #[test]
    fn name_is_stable() {
        assert_eq!(bce().name(), "Binary-Cross-Entropy");
    }
}
