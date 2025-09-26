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

    /// Computes the derivative of the ReLU function for backpropagation.
    ///
    /// The derivative is 1 for positive input values and 0 otherwise.
    /// This property allows gradients to flow only through activated neurons.
    fn derivative(&self, activations: ArrayView2<f32>, _: ArrayView2<f32>) -> Array2<f32> {
        activations.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 })
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
