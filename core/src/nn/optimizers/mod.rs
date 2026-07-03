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

use crate::gradients::LayerGradients;
use crate::layers::{Layer, Parameter};
use crate::learning_rate::LearningRate;
use ndarray::ArrayD;
use std::collections::HashMap;
use std::fmt;
use std::ops::{Deref, DerefMut};

/// A layer parameter paired with its gradient — the unit an [`Optimizer`] steps.
pub struct ParameterUpdate<'a> {
    /// The parameter to update in place.
    pub parameter: Parameter<'a>,
    /// The gradient of the loss with respect to that parameter.
    pub gradient: &'a ArrayD<f32>,
}

/// A layer's parameters, each paired with its gradient, in parameter order.
pub struct ParameterUpdates<'a>(Vec<ParameterUpdate<'a>>);

impl<'a> ParameterUpdates<'a> {
    /// Pairs each parameter with its gradient.
    /// # Panics
    /// When the number of parameters and gradients differ.
    pub fn new(parameters: Vec<Parameter<'a>>, gradients: &'a LayerGradients) -> Self {
        assert_eq!(
            parameters.len(),
            gradients.len(),
            "each parameter must have exactly one gradient"
        );
        Self(
            parameters
                .into_iter()
                .zip(gradients.iter())
                .map(|(parameter, gradient)| ParameterUpdate {
                    parameter,
                    gradient,
                })
                .collect(),
        )
    }
}

impl<'a> Deref for ParameterUpdates<'a> {
    type Target = [ParameterUpdate<'a>];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for ParameterUpdates<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Optimizer-agnostic snapshot of internal state (e.g. Adam's moment
/// estimates), shaped like a model: named tensors of arbitrary rank plus
/// scalar metadata. Serialization lives behind the `io` feature; this type
/// stays free of serde/safetensors.
#[derive(Debug)]
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

    /// Updates a layer's parameters in place, each from its paired gradient.
    ///
    /// # Arguments
    /// * `layer_index` - The index of the layer being updated, inside the neural network.
    /// * `updates` - The layer's trainable parameters, each paired with its gradient.
    fn update(&mut self, layer_index: usize, updates: &mut ParameterUpdates<'_>);

    /// Updates a layer in place, pairing its parameters with `gradients` before stepping.
    ///
    /// # Arguments
    /// * `layer_index` - The index of the layer being updated, inside the neural network.
    /// * `layer` - The layer whose parameters are updated.
    /// * `gradients` - The gradients computed during backpropagation, in parameter order.
    fn update_layer(
        &mut self,
        layer_index: usize,
        layer: &mut dyn Layer,
        gradients: &LayerGradients,
    ) {
        let mut updates = ParameterUpdates::new(layer.parameters_mut(), gradients);
        self.update(layer_index, &mut updates);
    }

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
