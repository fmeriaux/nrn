//! Softmax activation function implementation.
//!
//! This module provides the `Softmax` struct, which implements the `Activation` trait for the softmax activation function.
//! Softmax converts raw scores (logits) into probabilities, making it suitable for multi-class classification tasks in neural networks.
//! It operates on each column (sample) of the input, ensuring the output values are positive and sum to 1.
//! The implementation includes a numerical stability trick by subtracting the maximum value before exponentiation, which prevents overflow and improves reliability.
//! The derivative is implemented for use with cross-entropy loss, returning the gradient needed for backpropagation.

use crate::activations::{Activation, ActivationProvider};
use crate::initializations::{Initialization, XAVIER_UNIFORM};
use ndarray::{Array2, ArrayView2};
use once_cell::sync::Lazy;
use std::sync::Arc;

pub struct Softmax;

impl Activation for Softmax {
    /// Returns the canonical name of the activation function.
    fn name(&self) -> &'static str {
        "softmax"
    }

    /// Applies the softmax function to each column of the input matrix.
    ///
    /// For numerical stability, the maximum value in each column is subtracted before exponentiation.
    /// The result is normalized so that each column sums to 1, representing a probability distribution.
    ///
    /// # Panics
    /// Panics if the input is not a 2D array.
    fn apply(&self, input: ArrayView2<f32>) -> Array2<f32> {
        assert_eq!(
            input.ndim(),
            2,
            "Softmax apply: input must be 2D, got shape {:?}",
            input.shape()
        );
        let mut result = input.to_owned();

        for mut col in result.columns_mut() {
            // Numerical stability, subtract max value from each element
            let max_val = *col.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
            col.mapv_inplace(|x| (x - max_val).exp());

            // Normalize to get probabilities
            col /= col.sum();
        }

        result
    }

    /// Computes the derivative of the softmax function for backpropagation with cross-entropy loss.
    ///
    /// # Panics
    /// Will panic if the shapes of `activations` and `targets` do not match.
    fn derivative(&self, activations: ArrayView2<f32>, targets: ArrayView2<f32>) -> Array2<f32> {
        assert_eq!(
            activations.shape(),
            targets.shape(),
            "Softmax derivative: activations and targets must have the same shape, got {:?} and {:?}",
            activations.shape(),
            targets.shape()
        );
        let batch_size = activations.ncols() as f32;
        (activations.to_owned() - targets) / batch_size
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
