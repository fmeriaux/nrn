//! Sigmoid activation function implementation.
//!
//! This module provides the `Sigmoid` struct, which implements the `Activation` trait for the sigmoid activation function.
//! The sigmoid function maps input values to the (0, 1) range, making it suitable for binary classification tasks and as an output activation in neural networks.
//! It is defined as `f(x) = 1 / (1 + exp(-x))`. The sigmoid function is smooth and differentiable, but can suffer from vanishing gradients for large input values.

use crate::activations::{Activation, ActivationProvider};
use crate::initializations::{Initialization, XAVIER_UNIFORM};
use ndarray::{ArrayD, ArrayViewD, Zip};
use once_cell::sync::Lazy;
use std::sync::Arc;

#[derive(Debug)]
pub struct Sigmoid;

impl Activation for Sigmoid {
    /// Returns the canonical name of the activation function.
    fn name(&self) -> &'static str {
        "sigmoid"
    }

    /// Applies the sigmoid function element-wise, in place.
    fn apply_inplace(&self, input: &mut ArrayD<f32>) {
        input.mapv_inplace(|x| 1.0 / (1.0 + (-x).exp()));
    }

    /// Computes ∂L/∂z = upstream ⊙ a(1 − a).
    fn vjp(&self, upstream: ArrayViewD<f32>, activations: ArrayViewD<f32>) -> ArrayD<f32> {
        Zip::from(&upstream)
            .and(&activations)
            .map_collect(|&u, &s| u * s * (1.0 - s))
    }

    /// Provides the recommended initialization for layers using sigmoid.
    ///
    /// Xavier initialization is commonly used with sigmoid to maintain stable gradients.
    fn initialization(&self) -> Arc<dyn Initialization> {
        XAVIER_UNIFORM.clone()
    }
}

/// Static instance of the Sigmoid activation wrapped in an `Arc` for shared use.
pub static SIGMOID: Lazy<Arc<Sigmoid>> = Lazy::new(|| Arc::new(Sigmoid));
inventory::submit!(ActivationProvider(|| SIGMOID.clone()));

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn apply_at_zero_is_half() {
        let input = array![[0.0]].into_dyn();
        let result = SIGMOID.apply(input.view());
        assert!((result[[0, 0]] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn apply_large_positive_approaches_one() {
        let input = array![[100.0]].into_dyn();
        let result = SIGMOID.apply(input.view());
        assert!((result[[0, 0]] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn apply_large_negative_approaches_zero() {
        let input = array![[-100.0]].into_dyn();
        let result = SIGMOID.apply(input.view());
        assert!(result[[0, 0]] < 1e-5);
    }

    #[test]
    fn all_outputs_strictly_in_zero_one() {
        let input = array![[-10.0, -1.0, 0.0, 1.0, 10.0]].into_dyn();
        let result = SIGMOID.apply(input.view());
        for &v in result.iter() {
            assert!(v > 0.0 && v < 1.0, "Value {} not in (0, 1)", v);
        }
    }

    #[test]
    fn vjp_at_half_scales_by_quarter() {
        // a=0.5 → a(1-a)=0.25, upstream=1.0 → vjp=0.25
        let result = SIGMOID.vjp(
            array![[1.0]].view().into_dyn(),
            array![[0.5]].view().into_dyn(),
        );
        assert!((result[[0, 0]] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn vjp_scales_upstream_by_local_derivative() {
        // upstream=2.0, a=0.5 → vjp = 2.0 * 0.25 = 0.5
        let result = SIGMOID.vjp(
            array![[2.0]].view().into_dyn(),
            array![[0.5]].view().into_dyn(),
        );
        assert!((result[[0, 0]] - 0.5).abs() < 1e-6);
    }
}
