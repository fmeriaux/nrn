//! This module defines the core data structures for representing neural network architectures.
//!
//! It provides types for individual neuron layers ([`NeuronLayerSpec`]) and entire networks
//! ([`NeuralNetwork`]), plus the [`Predictor`] that pairs a trained network with its scaler.
//! The submodules are re-exported flat, so every type lives at `crate::model::*`.

mod network;
mod predictor;
mod specs;

pub use network::*;
pub use predictor::*;
pub use specs::*;
