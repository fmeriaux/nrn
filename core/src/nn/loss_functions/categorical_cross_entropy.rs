//! Cross-entropy loss for multi-class classification, computed from **logits**.
//! Supports N-Dimensional tensors laid out samples-last (the leading axis is always the classes).
//!
//! Softmax + categorical cross-entropy, valued via a max-shifted log-sum-exp for numerical
//! stability; the gradient is `softmax(z) − y`.

use crate::activations::{Activation, SOFTMAX};
use crate::loss_functions::{LossFunction, Reduction};
use ndarray::{ArrayD, ArrayViewD, Axis};

/// Categorical cross-entropy from logits: `k` softmax outputs per position, matched against a
/// one-hot (or soft) target lane. The loss is per-position; [`Reduction`] collapses it.
pub struct CategoricalCrossEntropy {
    reduction: Reduction,
}

impl CategoricalCrossEntropy {
    /// Builds a categorical cross-entropy loss with the given [`Reduction`].
    pub fn new(reduction: Reduction) -> Self {
        Self { reduction }
    }
}

impl LossFunction for CategoricalCrossEntropy {
    fn name(&self) -> &'static str {
        "Categorical-Cross-Entropy"
    }

    fn reduction(&self) -> Reduction {
        self.reduction
    }

    /// Computes the per-position loss from N-Dimensional logits.
    ///
    /// # Arguments
    /// * `inputs` - N-D array view where Axis(0) is `classes`.
    /// * `targets` - N-D array view of matching shape with true labels (one-hot or soft targets).
    ///
    /// # Panics
    /// When `inputs` and `targets` do not have the same shape.
    fn terms(&self, inputs: ArrayViewD<f32>, targets: ArrayViewD<f32>) -> ArrayD<f32> {
        assert_eq!(
            inputs.shape(),
            targets.shape(),
            "Inputs and targets must have the same shape."
        );

        // Per-position log-sum-exp over the class lane, max-shifted for numerical stability.
        let logsumexp = inputs.map_axis(Axis(0), |lane| {
            let max = lane.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            max + lane.iter().map(|&z| (z - max).exp()).sum::<f32>().ln()
        });
        // Categorical cross-entropy per position: logsumexp·Σy − Σ(y·z).
        let sum_y = targets.sum_axis(Axis(0));
        let sum_yz = (&inputs * &targets).sum_axis(Axis(0));
        logsumexp * sum_y - sum_yz
    }

    /// Computes ∂L/∂z — the gradient of the loss with respect to the logits: the per-position
    /// residual `softmax(z) − y`, scaled by the [`Reduction`] over the positions (the class lane
    /// is a single term). Works out-of-the-box for N-Dimensional tensors.
    ///
    /// # Panics
    /// When `inputs` and `targets` do not have the same shape.
    fn gradient(&self, inputs: ArrayViewD<f32>, targets: ArrayViewD<f32>) -> ArrayD<f32> {
        assert_eq!(
            inputs.shape(),
            targets.shape(),
            "Inputs and targets must have the same shape."
        );

        let positions = inputs.len() / inputs.shape()[0];
        let residual = SOFTMAX.apply(inputs) - targets;
        self.reduction.scale_gradient(residual, positions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Mean-reduced categorical cross-entropy, the default used throughout training.
    fn cce() -> CategoricalCrossEntropy {
        CategoricalCrossEntropy::new(Reduction::Mean)
    }

    #[test]
    fn confident_correct_logits_give_near_zero_loss() {
        // Two positions, both peaking on the true class → near-zero loss.
        let inputs = array![[10.0_f32, -10.0], [-10.0, 10.0]].into_dyn();
        let targets = array![[1.0_f32, 0.0], [0.0, 1.0]].into_dyn();
        let loss = cce().compute(inputs.view(), targets.view());
        assert!((0.0..1e-3).contains(&loss), "loss was {loss}");
    }

    #[test]
    fn uniform_logits_give_ln_of_n_classes() {
        // Equal logits → softmax is uniform → per-position loss is ln(k).
        let inputs = array![[0.0_f32], [0.0], [0.0]].into_dyn();
        let targets = array![[1.0_f32], [0.0], [0.0]].into_dyn();
        let loss = cce().compute(inputs.view(), targets.view());
        assert!((loss - 3.0_f32.ln()).abs() < 1e-6, "loss was {loss}");
    }

    #[test]
    fn sum_totals_every_position() {
        // Two uniform 3-class positions: Sum = 2·ln(3), Mean = ln(3).
        let inputs = array![[0.0_f32, 0.0], [0.0, 0.0], [0.0, 0.0]].into_dyn();
        let targets = array![[1.0_f32, 0.0], [0.0, 1.0], [0.0, 0.0]].into_dyn();
        let sum =
            CategoricalCrossEntropy::new(Reduction::Sum).compute(inputs.view(), targets.view());
        let mean = cce().compute(inputs.view(), targets.view());
        assert!((sum - 2.0 * 3.0_f32.ln()).abs() < 1e-6, "sum was {sum}");
        assert!((mean - 3.0_f32.ln()).abs() < 1e-6, "mean was {mean}");
    }

    #[test]
    fn mean_scores_every_spatial_position() {
        // (n_classes=2, height=2, samples=1): two uniform positions → Mean = ln(2).
        let inputs = array![[[0.0_f32], [0.0]], [[0.0], [0.0]]].into_dyn();
        let targets = array![[[1.0_f32], [0.0]], [[0.0], [1.0]]].into_dyn();
        let loss = cce().compute(inputs.view(), targets.view());
        assert!((loss - 2.0_f32.ln()).abs() < 1e-6, "loss was {loss}");
    }

    #[test]
    fn gradient_is_softmax_minus_target() {
        // Uniform logits → softmax 0.5 each; grad = 0.5 - y.
        let inputs = array![[0.0_f32], [0.0]].into_dyn();
        let targets = array![[1.0_f32], [0.0]].into_dyn();
        let grad = cce().gradient(inputs.view(), targets.view());
        let (g0, g1) = (grad[[0, 0]], grad[[1, 0]]);
        assert!((g0 - (-0.5)).abs() < 1e-6, "grad was {g0}");
        assert!((g1 - 0.5).abs() < 1e-6, "grad was {g1}");
    }

    #[test]
    fn mean_gradient_divides_the_residual_by_the_position_count() {
        // Two uniform 2-class positions: residual is softmax(0.5) − y at each, and Mean halves
        // it over the two positions (the class lane counts as a single term).
        let inputs = array![[0.0_f32, 0.0], [0.0, 0.0]].into_dyn();
        let targets = array![[1.0_f32, 0.0], [0.0, 1.0]].into_dyn();
        let grad = cce().gradient(inputs.view(), targets.view());
        // Position 0: (0.5-1, 0.5-0)/2 = (-0.25, 0.25); position 1: (0.5, 0.5-1)/2 = (0.25, -0.25).
        assert!(
            (grad[[0, 0]] - (-0.25)).abs() < 1e-6,
            "grad was {}",
            grad[[0, 0]]
        );
        assert!(
            (grad[[1, 0]] - 0.25).abs() < 1e-6,
            "grad was {}",
            grad[[1, 0]]
        );
        assert!(
            (grad[[0, 1]] - 0.25).abs() < 1e-6,
            "grad was {}",
            grad[[0, 1]]
        );
        assert!(
            (grad[[1, 1]] - (-0.25)).abs() < 1e-6,
            "grad was {}",
            grad[[1, 1]]
        );
    }

    #[test]
    fn name_is_stable() {
        assert_eq!(cce().name(), "Categorical-Cross-Entropy");
    }

    #[test]
    #[should_panic(expected = "must have the same shape")]
    fn terms_panics_on_a_shape_mismatch() {
        // A single class-id row instead of a one-hot target: must panic, not silently broadcast.
        let inputs = array![[0.0_f32, 0.0], [0.0, 0.0], [0.0, 0.0]].into_dyn();
        let targets = array![[1.0_f32, 2.0]].into_dyn();
        let _ = cce().terms(inputs.view(), targets.view());
    }

    #[test]
    #[should_panic(expected = "must have the same shape")]
    fn gradient_panics_on_a_shape_mismatch() {
        let inputs = array![[0.0_f32, 0.0], [0.0, 0.0], [0.0, 0.0]].into_dyn();
        let targets = array![[1.0_f32, 2.0]].into_dyn();
        let _ = cce().gradient(inputs.view(), targets.view());
    }
}
