//! Mean squared error loss for regression, computed directly from the network outputs.
//! Supports N-Dimensional tensors laid out samples-last (the leading axis is the output row).
//!
//! The per-term value is the squared residual `(z − y)²`, and the gradient is `2·(z − y)`
//! (following the un-halved PyTorch convention), both scaled by the [`Reduction`].

use crate::loss_functions::{LossFunction, Reduction};
use ndarray::{ArrayD, ArrayViewD, Zip};

/// Mean squared error: the squared residual between each output and its target. The loss
/// is element-wise; [`Reduction`] collapses it.
pub struct MeanSquaredError {
    reduction: Reduction,
}

impl MeanSquaredError {
    /// Builds a mean squared error loss with the given [`Reduction`].
    pub fn new(reduction: Reduction) -> Self {
        Self { reduction }
    }
}

impl LossFunction for MeanSquaredError {
    fn name(&self) -> &'static str {
        "Mean-Squared-Error"
    }

    fn reduction(&self) -> Reduction {
        self.reduction
    }

    /// Computes the per-element squared residual from N-Dimensional outputs.
    ///
    /// # Arguments
    /// * `inputs` - N-D array view of the network's outputs.
    /// * `targets` - N-D array view of matching shape with the true values.
    fn terms(&self, inputs: ArrayViewD<f32>, targets: ArrayViewD<f32>) -> ArrayD<f32> {
        Zip::from(&inputs)
            .and(&targets)
            .map_collect(|&z, &y| (z - y).powi(2))
    }

    /// Computes ∂L/∂z — the gradient of the loss with respect to the outputs: the per-element
    /// residual `2·(z − y)`, scaled by the [`Reduction`]. Works out-of-the-box for N-D tensors.
    fn gradient(&self, inputs: ArrayViewD<f32>, targets: ArrayViewD<f32>) -> ArrayD<f32> {
        let n_terms = inputs.len();
        let residual = Zip::from(&inputs)
            .and(&targets)
            .map_collect(|&z, &y| 2.0 * (z - y));
        self.reduction.scale_gradient(residual, n_terms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Mean-reduced mean squared error, the default used throughout training.
    fn mse() -> MeanSquaredError {
        MeanSquaredError::new(Reduction::Mean)
    }

    #[test]
    fn exact_predictions_give_zero_loss() {
        let inputs = array![[1.0_f32, -2.0, 3.5]].into_dyn();
        let targets = array![[1.0_f32, -2.0, 3.5]].into_dyn();
        let loss = mse().compute(inputs.view(), targets.view());
        assert_eq!(loss, 0.0);
    }

    #[test]
    fn mean_averages_the_squared_residuals() {
        let inputs = array![[1.0_f32, 4.0, -1.0]].into_dyn();
        let targets = array![[0.0_f32, 2.0, 1.0]].into_dyn();
        // Residuals 1, 2, -2 → squared 1, 4, 4 → mean 3.
        let loss = mse().compute(inputs.view(), targets.view());
        assert!((loss - 3.0).abs() < 1e-6, "loss was {loss}");
    }

    #[test]
    fn sum_totals_the_squared_residuals() {
        let inputs = array![[1.0_f32, 4.0, -1.0]].into_dyn();
        let targets = array![[0.0_f32, 2.0, 1.0]].into_dyn();
        let loss = MeanSquaredError::new(Reduction::Sum).compute(inputs.view(), targets.view());
        assert!((loss - 9.0).abs() < 1e-6, "loss was {loss}");
    }

    #[test]
    fn mean_scores_every_spatial_position() {
        // (output=1, height=2, samples=2): four positions averaged.
        let inputs = array![[[1.0_f32, 0.0], [2.0, -1.0]]].into_dyn();
        let targets = array![[[0.0_f32, 0.0], [0.0, 0.0]]].into_dyn();
        // Squared residuals 1, 0, 4, 1 → mean 1.5.
        let loss = mse().compute(inputs.view(), targets.view());
        assert!((loss - 1.5).abs() < 1e-6, "loss was {loss}");
    }

    #[test]
    fn summed_gradient_is_twice_the_residual() {
        let inputs = array![[1.0_f32, 4.0]].into_dyn();
        let targets = array![[0.0_f32, 2.0]].into_dyn();
        let grad = MeanSquaredError::new(Reduction::Sum).gradient(inputs.view(), targets.view());
        // 2·(1-0)=2 ; 2·(4-2)=4
        assert_eq!(grad[[0, 0]], 2.0);
        assert_eq!(grad[[0, 1]], 4.0);
    }

    #[test]
    fn mean_gradient_divides_the_residual_by_the_term_count() {
        let inputs = array![[1.0_f32, 4.0]].into_dyn();
        let targets = array![[0.0_f32, 2.0]].into_dyn();
        let grad = mse().gradient(inputs.view(), targets.view());
        // The summed gradient (2, 4) halved over the two terms.
        assert_eq!(grad[[0, 0]], 1.0);
        assert_eq!(grad[[0, 1]], 2.0);
    }

    #[test]
    fn name_is_stable() {
        assert_eq!(mse().name(), "Mean-Squared-Error");
    }
}
