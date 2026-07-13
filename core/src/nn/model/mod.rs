//! This module defines the core data structures for representing neural network architectures.
//!
//! It provides types for individual neuron layers ([`NeuronLayerSpec`]) and entire networks
//! ([`NeuralNetwork`]), plus the [`Predictor`] that pairs a trained network with its scaler.
//! The submodules are re-exported flat, so every type lives at `crate::model::*`.

mod config;
mod network;
mod predictor;

pub use config::*;
pub use network::*;
pub use predictor::*;
