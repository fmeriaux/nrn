//! This module defines the core data structures for representing neural network architectures.
//!
//! It provides types for individual neuron layers ([`NeuronLayer`]) and entire networks ([`NeuronNetwork`]).
//! Each layer contains its weights, biases, and activation function, enabling flexible and modular
//! construction of multi-layer perceptron and similar models.

use crate::core::activations::Activation;
use ndarray::{Array1, Array2};
use std::sync::Arc;

/// Represents a single layer in a neural network, containing weights and biases.
#[derive(Clone)]
pub struct NeuronLayer {
    /// A 2D array where each row corresponds to a neuron and each column corresponds to an input feature.
    pub weights: Array2<f32>,
    /// A 1D array where each element is the bias for the corresponding neuron.
    pub bias: Array1<f32>,
    /// The activation function applied to the output of this layer.
    pub activation: Arc<dyn Activation>,
}

/// Represents a neural network composed of multiple layers of neurons.
#[derive(Clone)]
pub struct NeuronNetwork {
    /// A vector of [`NeuronLayer`] instances, defining the architecture of the network.
    pub layers: Vec<NeuronLayer>,
}

/// Represents the specifications for a neuron layer in a neural network.
pub struct NeuronLayerSpec {
    /// The number of neurons in this layer.
    pub neurons: usize,
    /// The activation function used in this layer.
    pub activation: Arc<dyn Activation>,
}
