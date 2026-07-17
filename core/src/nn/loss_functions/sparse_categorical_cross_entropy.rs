//! Sparse categorical cross-entropy loss for multi-class classification, computed from
//! **logits** against direct class ids rather than one-hot targets.
//! Supports N-Dimensional tensors laid out samples-last (the leading axis is always the classes).
//!
//! Softmax + categorical cross-entropy, valued via a max-shifted log-sum-exp for numerical
//! stability; the gradient is `softmax(z) − onehot(y)`, computed without materializing the
//! one-hot row.

use crate::activations::{Activation, SOFTMAX};
use crate::loss_functions::{LossFunction, Reduction};
use ndarray::{ArrayD, ArrayViewD, Axis};

/// Categorical cross-entropy from logits: `k` softmax outputs per position, matched against a
/// single class-id target lane. The loss is per-position; [`Reduction`] collapses it.
pub struct SparseCategoricalCrossEntropy {
    reduction: Reduction,
}

impl SparseCategoricalCrossEntropy {
    /// Builds a sparse categorical cross-entropy loss with the given [`Reduction`].
    pub fn new(reduction: Reduction) -> Self {
        Self { reduction }
    }
}

impl LossFunction for SparseCategoricalCrossEntropy {
    fn name(&self) -> &'static str {
        "Sparse-Categorical-Cross-Entropy"
    }

    fn reduction(&self) -> Reduction {
        self.reduction
    }

    /// Computes the per-position loss from N-Dimensional logits.
    ///
    /// # Arguments
    /// * `inputs` - N-D array view where Axis(0) is `classes`.
    /// * `targets` - N-D array view whose Axis(0) is a single class-id row, matching `inputs`
    ///   on every other axis.
    fn terms(&self, inputs: ArrayViewD<f32>, targets: ArrayViewD<f32>) -> ArrayD<f32> {
        // Per-position log-sum-exp over the class lane, max-shifted for numerical stability.
        let logsumexp = inputs.map_axis(Axis(0), |lane| {
            let max = lane.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            max + lane.iter().map(|&z| (z - max).exp()).sum::<f32>().ln()
        });
        // Categorical cross-entropy per position: logsumexp − the true class's logit.
        let ids = targets.index_axis(Axis(0), 0);
        let true_logit = inputs
            .lanes(Axis(0))
            .into_iter()
            .zip(ids.iter())
            .map(|(lane, &id)| lane[id as usize]);
        logsumexp - ArrayD::from_shape_vec(ids.raw_dim(), true_logit.collect()).unwrap()
    }

    /// Computes ∂L/∂z — the gradient of the loss with respect to the logits: the per-position
    /// residual `softmax(z) − onehot(y)`, scaled by the [`Reduction`] over the positions (the
    /// class lane is a single term). Works out-of-the-box for N-Dimensional tensors.
    fn gradient(&self, inputs: ArrayViewD<f32>, targets: ArrayViewD<f32>) -> ArrayD<f32> {
        let positions = inputs.len() / inputs.shape()[0];
        let mut residual = SOFTMAX.apply(inputs);
        let ids = targets.index_axis(Axis(0), 0);
        for (mut lane, &id) in residual.lanes_mut(Axis(0)).into_iter().zip(ids.iter()) {
            lane[id as usize] -= 1.0;
        }
        self.reduction.scale_gradient(residual, positions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loss_functions::CategoricalCrossEntropy;
    use ndarray::array;

    /// Mean-reduced sparse categorical cross-entropy, the default used throughout training.
    fn sparse_cce() -> SparseCategoricalCrossEntropy {
        SparseCategoricalCrossEntropy::new(Reduction::Mean)
    }

    #[test]
    fn confident_correct_logits_give_near_zero_loss() {
        // Two positions, both peaking on the true class → near-zero loss.
        let inputs = array![[10.0_f32, -10.0], [-10.0, 10.0]].into_dyn();
        let targets = array![[0.0_f32, 1.0]].into_dyn();
        let loss = sparse_cce().compute(inputs.view(), targets.view());
        assert!((0.0..1e-3).contains(&loss), "loss was {loss}");
    }

    #[test]
    fn uniform_logits_give_ln_of_n_classes() {
        // Equal logits → softmax is uniform → per-position loss is ln(k).
        let inputs = array![[0.0_f32], [0.0], [0.0]].into_dyn();
        let targets = array![[0.0_f32]].into_dyn();
        let loss = sparse_cce().compute(inputs.view(), targets.view());
        assert!((loss - 3.0_f32.ln()).abs() < 1e-6, "loss was {loss}");
    }

    #[test]
    fn sum_totals_every_position() {
        // Two uniform 3-class positions: Sum = 2·ln(3), Mean = ln(3).
        let inputs = array![[0.0_f32, 0.0], [0.0, 0.0], [0.0, 0.0]].into_dyn();
        let targets = array![[0.0_f32, 1.0]].into_dyn();
        let sum = SparseCategoricalCrossEntropy::new(Reduction::Sum)
            .compute(inputs.view(), targets.view());
        let mean = sparse_cce().compute(inputs.view(), targets.view());
        assert!((sum - 2.0 * 3.0_f32.ln()).abs() < 1e-6, "sum was {sum}");
        assert!((mean - 3.0_f32.ln()).abs() < 1e-6, "mean was {mean}");
    }

    #[test]
    fn mean_scores_every_spatial_position() {
        // (n_classes=2, height=2, samples=1): two uniform positions → Mean = ln(2).
        let inputs = array![[[0.0_f32], [0.0]], [[0.0], [0.0]]].into_dyn();
        let targets = array![[[0.0_f32], [1.0]]].into_dyn();
        let loss = sparse_cce().compute(inputs.view(), targets.view());
        assert!((loss - 2.0_f32.ln()).abs() < 1e-6, "loss was {loss}");
    }

    #[test]
    fn gradient_is_softmax_minus_onehot() {
        // Uniform logits → softmax 0.5 each; grad = 0.5 - onehot(y).
        let inputs = array![[0.0_f32], [0.0]].into_dyn();
        let targets = array![[0.0_f32]].into_dyn();
        let grad = sparse_cce().gradient(inputs.view(), targets.view());
        let (g0, g1) = (grad[[0, 0]], grad[[1, 0]]);
        assert!((g0 - (-0.5)).abs() < 1e-6, "grad was {g0}");
        assert!((g1 - 0.5).abs() < 1e-6, "grad was {g1}");
    }

    #[test]
    fn mean_gradient_divides_the_residual_by_the_position_count() {
        // Two uniform 2-class positions: residual is softmax(0.5) − onehot(y) at each, and Mean
        // halves it over the two positions (the class lane counts as a single term).
        let inputs = array![[0.0_f32, 0.0], [0.0, 0.0]].into_dyn();
        let targets = array![[0.0_f32, 1.0]].into_dyn();
        let grad = sparse_cce().gradient(inputs.view(), targets.view());
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
    fn matches_one_hot_categorical_cross_entropy_bit_for_bit() {
        let inputs = array![[2.0_f32, -1.0, 0.5], [0.1, 3.0, -2.0], [-1.5, 0.4, 1.2]].into_dyn();
        let ids = array![[0.0_f32, 1.0, 2.0]].into_dyn();
        let one_hot = array![[1.0_f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]].into_dyn();

        let sparse = sparse_cce();
        let dense = CategoricalCrossEntropy::new(Reduction::Mean);

        assert_eq!(
            sparse.compute(inputs.view(), ids.view()),
            dense.compute(inputs.view(), one_hot.view())
        );
        assert_eq!(
            sparse.gradient(inputs.view(), ids.view()),
            dense.gradient(inputs.view(), one_hot.view())
        );
    }

    #[test]
    fn stays_finite_where_dense_one_hot_would_nan_on_a_masked_class() {
        // A -inf logit on a *non-true* class (e.g. a masked-out class) is otherwise harmless:
        // `exp(-inf - max) = 0.0` inside log-sum-exp. But dense one-hot CCE also computes
        // `sum_yz = Σ(inputs · targets)`, and `-inf * 0.0 = NaN` in IEEE 754 poisons that sum
        // even though the target is exactly 0 there. Sparse never multiplies by the target — it
        // indexes the true class's logit directly — so it stays finite.
        let inputs = array![[5.0_f32], [f32::NEG_INFINITY]].into_dyn();
        let ids = array![[0.0_f32]].into_dyn();
        let one_hot = array![[1.0_f32], [0.0]].into_dyn();

        let sparse_loss = sparse_cce().compute(inputs.view(), ids.view());
        let dense_loss =
            CategoricalCrossEntropy::new(Reduction::Mean).compute(inputs.view(), one_hot.view());

        assert!(sparse_loss.is_finite(), "loss was {sparse_loss}");
        assert!(
            dense_loss.is_nan(),
            "expected dense CCE to NaN on -inf * 0.0"
        );
    }

    #[test]
    fn name_is_stable() {
        assert_eq!(sparse_cce().name(), "Sparse-Categorical-Cross-Entropy");
    }
}
