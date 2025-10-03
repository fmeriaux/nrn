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

mod adam;
mod sgd;

pub use adam::*;
pub use sgd::*;

use crate::model::NeuronLayer;
use crate::training::Gradients;

pub trait Optimizer {
    /// Updates the weights and biases of a layer using the provided gradients.
    ///
    /// # Arguments
    /// * `layer_index` - The index of the layer being updated, inside the neural network.
    /// * `layer` - The layer whose weights and biases are to be updated.
    /// * `gradients` - The gradients computed during backpropagation for this layer.
    fn update(&mut self, layer_index: usize, layer: &mut NeuronLayer, gradients: &Gradients);

    /// Performs any necessary state updates after each training step.
    /// This is useful for optimizers that maintain internal state, such as Adam.
    /// The default implementation does nothing.
    fn step(&mut self) {}
}
