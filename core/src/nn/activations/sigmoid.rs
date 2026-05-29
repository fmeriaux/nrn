//! Sigmoid activation function implementation.
//!
//! This module provides the `Sigmoid` struct, which implements the `Activation` trait for the sigmoid activation function.
//! The sigmoid function maps input values to the (0, 1) range, making it suitable for binary classification tasks and as an output activation in neural networks.
//! It is defined as `f(x) = 1 / (1 + exp(-x))`. The sigmoid function is smooth and differentiable, but can suffer from vanishing gradients for large input values.

use crate::activations::{Activation, ActivationProvider};
use crate::initializations::{Initialization, XAVIER_UNIFORM};
use ndarray::{Array2, ArrayView2};
use once_cell::sync::Lazy;
use std::sync::Arc;

pub struct Sigmoid;

impl Activation for Sigmoid {
    /// Returns the canonical name of the activation function.
    fn name(&self) -> &'static str {
        "sigmoid"
    }

    /// Applies the sigmoid function element-wise to the input matrix.
    fn apply(&self, input: ArrayView2<f32>) -> Array2<f32> {
        input.mapv(|x| 1.0 / (1.0 + (-x).exp()))
    }

    /// Computes the derivative of the sigmoid function for backpropagation.
    ///
    /// This is used to propagate gradients during training.
    fn derivative(&self, activations: ArrayView2<f32>) -> Array2<f32> {
        activations.mapv(|s| s * (1.0 - s))
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
        let input = array![[0.0]];
        let result = SIGMOID.apply(input.view());
        assert!((result[[0, 0]] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn apply_large_positive_approaches_one() {
        let input = array![[100.0]];
        let result = SIGMOID.apply(input.view());
        assert!((result[[0, 0]] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn apply_large_negative_approaches_zero() {
        let input = array![[-100.0]];
        let result = SIGMOID.apply(input.view());
        assert!(result[[0, 0]] < 1e-5);
    }

    #[test]
    fn all_outputs_strictly_in_zero_one() {
        let input = array![[-10.0, -1.0, 0.0, 1.0, 10.0]];
        let result = SIGMOID.apply(input.view());
        for &v in result.iter() {
            assert!(v > 0.0 && v < 1.0, "Value {} not in (0, 1)", v);
        }
    }

    #[test]
    fn derivative_at_half_is_quarter() {
        // sigma'(x) = sigma(x) * (1 - sigma(x)), at sigma(x)=0.5 -> 0.5 * 0.5 = 0.25
        let activations = array![[0.5]];
        let d = SIGMOID.derivative(activations.view());
        assert!((d[[0, 0]] - 0.25).abs() < 1e-6);
    }
}
