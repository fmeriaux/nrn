//! Optimizer module.
//!
//! This module defines the `Optimizer` trait for neural network optimization algorithms.
//! It enables extensibility by allowing new optimizers (such as SGD, Adam, RMSProp, etc.) to be implemented
//! and used interchangeably in the training process. Each optimizer provides a unified interface for updating
//! the weights and biases of a layer based on computed gradients.
//!
//! Example usage:
//! ```rust
//! use crate::optimizers::{Optimizer, StochasticGradientDescent};
//! let mut optimizer = StochasticGradientDescent::new(0.01);
//! optimizer.update(&mut layer, &gradients);
//! ```

mod sgd;

pub use sgd::StochasticGradientDescent;

use crate::model::NeuronLayer;
use crate::training::Gradients;

pub trait Optimizer {
    /// Updates the weights and biases of a layer using the provided gradients.
    ///
    /// # Arguments
    /// * `layer` - The layer to update.
    /// * `gradients` - The gradients computed during backpropagation for this layer.
    fn update(&mut self, layer: &mut NeuronLayer, gradients: &Gradients);
}