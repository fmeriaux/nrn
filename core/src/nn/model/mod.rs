//! Neural network architecture: a declarative, weight-free description of a network's shape,
//! and the live network of weighted layers it builds.
//!
//! [`NetworkConfig`], assembled fluently through [`NetworkConfigBuilder`], is the architecture
//! alone; [`NeuralNetwork`] is the weighted layers it configures. [`Predictor`] pairs a trained
//! network with the scaler fitted alongside it.
//! The submodules are re-exported flat, so every type lives at `crate::model::*`.

mod config;
mod network;
mod predictor;

pub use config::*;
pub use network::*;
pub use predictor::*;
