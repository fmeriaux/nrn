//! ReLU activation function implementation.
//!
//! This module provides the `ReLU` struct, which implements the `Activation` trait for the Rectified Linear Unit activation function.
//! ReLU is widely used in neural networks for its simplicity and effectiveness in introducing non-linearity.
//! It outputs the input directly if it is positive; otherwise, it outputs zero. This helps mitigate the vanishing gradient problem and accelerates convergence during training.

use crate::activations::{Activation, ActivationProvider};
use crate::initializations::{HE_UNIFORM, Initialization};
use ndarray::{Array2, ArrayView2};
use once_cell::sync::Lazy;
use std::sync::Arc;

pub struct ReLU;

impl Activation for ReLU {
    /// Returns the canonical name of the activation function.
    fn name(&self) -> &'static str {
        "relu"
    }

    /// Applies the ReLU function element-wise to the input matrix.
    fn apply(&self, input: ArrayView2<f32>) -> Array2<f32> {
        input.mapv(|x| x.max(0.0))
    }

    /// Computes ∂L/∂z = upstream ⊙ 1[a > 0].
    fn vjp(&self, upstream: ArrayView2<f32>, activations: ArrayView2<f32>) -> Array2<f32> {
        upstream.to_owned() * activations.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 })
    }

    /// Provides the recommended initialization for layers using ReLU.
    ///
    /// He initialization is commonly used with ReLU to maintain variance in deep networks.
    fn initialization(&self) -> Arc<dyn Initialization> {
        HE_UNIFORM.clone()
    }
}

/// Static instance of the ReLU activation wrapped in an `Arc` for shared use.
pub static RELU: Lazy<Arc<ReLU>> = Lazy::new(|| Arc::new(ReLU));
inventory::submit!(ActivationProvider(|| RELU.clone()));

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn apply_passes_positive_values_unchanged() {
        let input = array![[1.0, 2.0], [3.0, 4.0]];
        let result = RELU.apply(input.view());
        assert_eq!(result, input);
    }

    #[test]
    fn apply_zeros_negative_values() {
        let input = array![[-1.0, -2.0], [-3.0, -0.5]];
        let result = RELU.apply(input.view());
        assert_eq!(result, array![[0.0, 0.0], [0.0, 0.0]]);
    }

    #[test]
    fn apply_mixed_values() {
        let input = array![[-1.0, 2.0], [0.0, -3.0]];
        let result = RELU.apply(input.view());
        assert_eq!(result, array![[0.0, 2.0], [0.0, 0.0]]);
    }

    #[test]
    fn vjp_passes_upstream_for_positive_activations() {
        let upstream = array![[1.0, 2.0], [3.0, 4.0]];
        let activations = array![[0.5, 1.0], [2.0, 0.1]];
        assert_eq!(RELU.vjp(upstream.view(), activations.view()), upstream);
    }

    #[test]
    fn vjp_blocks_upstream_for_nonpositive_activations() {
        let upstream = array![[1.0, 2.0], [3.0, 4.0]];
        let activations = array![[0.0, -1.0], [-2.0, 0.0]];
        assert_eq!(
            RELU.vjp(upstream.view(), activations.view()),
            array![[0.0, 0.0], [0.0, 0.0]]
        );
    }

    #[test]
    fn vjp_mixed_activations() {
        let upstream = array![[2.0, 3.0]];
        let activations = array![[1.0, -1.0]];
        assert_eq!(RELU.vjp(upstream.view(), activations.view()), array![[2.0, 0.0]]);
    }
}
