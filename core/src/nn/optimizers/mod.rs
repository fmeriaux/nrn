//! Optimizer module.
//!
//! This module defines the `Optimizer` trait for neural network optimization algorithms.
//! It enables extensibility by allowing new optimizers (such as SGD, Adam, RMSProp, etc.) to be implemented
//! and used interchangeably in the training process. Each optimizer provides a unified interface for updating
//! the weights and biases of a layer based on computed gradients.
//!

mod adam;
mod sgd;

pub use adam::*;
pub use sgd::*;

use crate::gradients::Gradients;
use crate::learning_rate::LearningRate;
use crate::model::NeuronLayer;
use ndarray::ArrayD;
use std::collections::HashMap;
use std::io::Result;

/// Optimizer-agnostic snapshot of internal state (e.g. Adam's moment
/// estimates), shaped like a model: named tensors of arbitrary rank plus
/// scalar metadata. Encoding to/from safetensors lives in `io::optimizer`;
/// this type stays free of serde/safetensors.
pub struct OptimizerState {
    pub tensors: Vec<(String, ArrayD<f32>)>,
    pub metadata: HashMap<String, String>,
}

pub trait Optimizer {
    /// Returns a human-readable name for this optimizer.
    fn name(&self) -> &'static str;

    /// Returns the optimizer's current learning rate.
    fn learning_rate(&self) -> LearningRate;

    /// Sets the learning rate for the optimizer.
    /// This allows dynamic adjustment of the learning rate during training.
    fn set_learning_rate(&mut self, learning_rate: LearningRate);

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

    /// Returns a snapshot of this optimizer's internal state for checkpointing,
    /// or `None` for stateless optimizers (e.g. SGD).
    fn save_state(&self) -> Option<OptimizerState> {
        None
    }

    /// Restores internal state previously returned by [`save_state`](Optimizer::save_state).
    /// The default implementation ignores `state` (stateless optimizers).
    fn load_state(&mut self, _state: &OptimizerState) -> Result<()> {
        Ok(())
    }
}
