//! The [`Predictor`]: a trained network paired with the scaler fitted alongside it, plus the
//! error it raises when an instance's shape does not match.

use crate::data::scalers::{ScalerFeatureMismatch, ScalerMethod};
use crate::model::{InputShapeMismatch, NeuralNetwork};
use std::fmt;

/// A trained [`NeuralNetwork`] paired with the scaler fitted alongside it.
#[derive(Clone, Debug)]
pub struct Predictor {
    /// The trained network.
    pub network: NeuralNetwork,
    /// The scaler applied to raw inputs before prediction, when one is present.
    pub scaler: Option<ScalerMethod>,
}

impl Predictor {
    /// Pairs a network with an optional scaler.
    pub fn new(network: NeuralNetwork, scaler: Option<ScalerMethod>) -> Self {
        Self { network, scaler }
    }
}

/// A [`Predictor`] rejected an instance: either its scaler or its network found the
/// wrong number of features.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PredictionError {
    /// The scaler's fitted feature count did not match the input.
    Scaling(ScalerFeatureMismatch),
    /// The network's input shape did not match the input.
    Network(InputShapeMismatch),
}

impl fmt::Display for PredictionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PredictionError::Scaling(e) => write!(f, "{e}"),
            PredictionError::Network(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for PredictionError {}

impl From<ScalerFeatureMismatch> for PredictionError {
    fn from(error: ScalerFeatureMismatch) -> Self {
        PredictionError::Scaling(error)
    }
}

impl From<InputShapeMismatch> for PredictionError {
    fn from(error: InputShapeMismatch) -> Self {
        PredictionError::Network(error)
    }
}
