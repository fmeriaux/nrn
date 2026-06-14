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
use std::fmt;

/// Optimizer-agnostic snapshot of internal state (e.g. Adam's moment
/// estimates), shaped like a model: named tensors of arbitrary rank plus
/// scalar metadata. Serialization lives behind the `io` feature; this type
/// stays free of serde/safetensors.
pub struct OptimizerState {
    pub tensors: Vec<(String, ArrayD<f32>)>,
    pub metadata: HashMap<String, String>,
}

/// Returned by [`Optimizer::restore`] when an [`OptimizerState`] is missing
/// or malformed for the optimizer being restored.
#[derive(Debug)]
pub enum OptimizerStateError {
    /// A required metadata entry was missing from the state.
    MissingMetadata(String),
    /// A metadata entry could not be parsed into the expected type.
    InvalidMetadata { key: String },
    /// A tensor was present but did not have the expected rank.
    WrongRank { tensor: String, expected: usize },
    /// A required tensor was missing from the state.
    MissingTensor(String),
}

impl fmt::Display for OptimizerStateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OptimizerStateError::MissingMetadata(key) => {
                write!(f, "optimizer state is missing `{key}`")
            }
            OptimizerStateError::InvalidMetadata { key } => {
                write!(f, "optimizer state has an invalid `{key}`")
            }
            OptimizerStateError::WrongRank { tensor, expected } => {
                write!(f, "tensor `{tensor}` is not rank {expected}")
            }
            OptimizerStateError::MissingTensor(name) => {
                write!(f, "optimizer state is missing `{name}`")
            }
        }
    }
}

impl std::error::Error for OptimizerStateError {}

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
    fn to_state(&self) -> Option<OptimizerState> {
        None
    }

    /// Restores internal state previously returned by [`to_state`](Optimizer::to_state).
    /// The default implementation ignores `state` (stateless optimizers).
    fn restore(&mut self, _state: &OptimizerState) -> Result<(), OptimizerStateError> {
        Ok(())
    }
}
