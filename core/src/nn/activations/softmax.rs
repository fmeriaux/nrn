//! Softmax activation function implementation.
//!
//! This module provides the `Softmax` struct, which implements the `Activation` trait for the softmax activation function.
//! Softmax converts raw scores (logits) into probabilities, making it suitable for multi-class classification tasks in neural networks.
//! It operates over the leading axis (the classes), ensuring each lane is positive and sums to 1.
//! The implementation includes a numerical stability trick by subtracting the maximum value before exponentiation, which prevents overflow and improves reliability.
//! The derivative is implemented for use with cross-entropy loss, returning the gradient needed for backpropagation.

use crate::activations::{Activation, ActivationProvider};
use crate::initializations::{Initialization, XAVIER_UNIFORM};
use ndarray::{ArrayD, ArrayViewD, Axis};
use once_cell::sync::Lazy;
use std::sync::Arc;

#[derive(Debug)]
pub struct Softmax;

impl Activation for Softmax {
    /// Returns the canonical name of the activation function.
    fn name(&self) -> &'static str {
        "softmax"
    }

    /// Applies the softmax function over the leading axis (the classes).
    ///
    /// For numerical stability, the max of each lane is subtracted before exponentiation.
    /// Each lane is then normalized to sum to 1, forming a probability distribution.
    fn apply(&self, input: ArrayViewD<f32>) -> ArrayD<f32> {
        let mut result = input.to_owned();

        for mut lane in result.lanes_mut(Axis(0)) {
            let max = lane.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            lane.mapv_inplace(|x| (x - max).exp());
            let sum = lane.sum();
            lane /= sum;
        }

        result
    }

    /// Computes ∂L/∂z = a ⊙ (upstream − ⟨a, upstream⟩) per lane.
    fn vjp(&self, upstream: ArrayViewD<f32>, activations: ArrayViewD<f32>) -> ArrayD<f32> {
        let dot = (&activations * &upstream)
            .sum_axis(Axis(0))
            .insert_axis(Axis(0));
        activations.to_owned() * (upstream.to_owned() - dot)
    }

    /// Provides the recommended initialization for layers using softmax.
    ///
    /// Xavier initialization is commonly used with softmax to maintain stable gradients.
    fn initialization(&self) -> Arc<dyn Initialization> {
        XAVIER_UNIFORM.clone()
    }
}

/// Static instance of the Softmax activation wrapped in an `Arc` for shared use.
pub static SOFTMAX: Lazy<Arc<Softmax>> = Lazy::new(|| Arc::new(Softmax));
inventory::submit!(ActivationProvider(|| SOFTMAX.clone()));

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Axis, array};

    #[test]
    fn output_sums_to_one_per_lane() {
        // shape: (3 classes, 2 samples)
        let input = array![[1.0, 2.0], [2.0, 1.0], [3.0, 3.0]].into_dyn();
        let result = SOFTMAX.apply(input.view());
        for lane in result.lanes(Axis(0)) {
            let sum: f32 = lane.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "lane sum {} != 1.0", sum);
        }
    }

    #[test]
    fn all_outputs_positive() {
        let input = array![[-5.0, 0.0], [0.0, 1.0], [5.0, -1.0]].into_dyn();
        let result = SOFTMAX.apply(input.view());
        for &v in result.iter() {
            assert!(v > 0.0, "Value {} is not positive", v);
        }
    }

    #[test]
    fn max_input_gets_highest_probability() {
        // Max input at row 2 -> highest probability at row 2
        let input = array![[1.0], [2.0], [10.0]].into_dyn();
        let result = SOFTMAX.apply(input.view());
        let lane = result.index_axis(Axis(1), 0);
        let max_idx = lane
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(max_idx, 2);
    }

    #[test]
    fn numerically_stable_with_large_inputs() {
        let input = array![[1000.0], [1001.0], [1002.0]].into_dyn();
        let result = SOFTMAX.apply(input.view());
        for &v in result.iter() {
            assert!(v.is_finite(), "Value {} is not finite", v);
        }
    }

    #[test]
    fn vjp_output_sums_to_zero_per_lane() {
        // Property: softmax VJP always produces zero-sum lanes
        let activations = array![[0.7, 0.3], [0.2, 0.3], [0.1, 0.4]].into_dyn();
        let upstream = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]].into_dyn();
        let result = SOFTMAX.vjp(upstream.view(), activations.view());
        for lane in result.lanes(Axis(0)) {
            let sum: f32 = lane.iter().sum();
            assert!(sum.abs() < 1e-5, "lane sum {sum} != 0");
        }
    }

    #[test]
    fn vjp_known_values() {
        // a=[0.7, 0.2, 0.1], u=[1, 0, 0]: dot=0.7, vjp=a*(u-dot)=[0.21, -0.14, -0.07]
        let activations = array![[0.7], [0.2], [0.1]].into_dyn();
        let upstream = array![[1.0], [0.0], [0.0]].into_dyn();
        let result = SOFTMAX.vjp(upstream.view(), activations.view());
        assert!((result[[0, 0]] - 0.21).abs() < 1e-6);
        assert!((result[[1, 0]] - (-0.14)).abs() < 1e-6);
        assert!((result[[2, 0]] - (-0.07)).abs() < 1e-6);
    }
}
