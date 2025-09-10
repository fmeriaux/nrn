use std::fmt;
use ndarray::{Array2, Axis};
use crate::core::activation::ActivationMethod::*;

/// Represents the activation methods available for neurons in a neural network.
#[derive(Clone)]
pub enum ActivationMethod {
    Sigmoid,
    Softmax,
    ReLU
}

impl fmt::Display for ActivationMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Sigmoid => write!(f, "sigmoid"),
            Softmax => write!(f, "softmax"),
            ReLU => write!(f, "relu"),
        }
    }
}

impl std::str::FromStr for ActivationMethod {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "sigmoid" => Ok(Sigmoid),
            "softmax" => Ok(Softmax),
            "relu" => Ok(ReLU),
            _ => Err(format!("Unknown activation method: {}", s)),
        }
    }
}


impl ActivationMethod {
    /// Applies the activation function to the input array.
    pub fn apply(&self, output: &Array2<f32>) -> Array2<f32> {
        match self {
            Sigmoid =>
                output.mapv(|val| 1.0 / (1.0 + (-val).exp())),

            Softmax => {
                let max = output.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let exp_output = output.mapv(|val| (val - max).exp());
                let sum = exp_output.sum_axis(Axis(0));
                exp_output / sum
            },
            ReLU =>
                output.mapv(|val| if val > 0.0 { val } else { 0.0 }),
        }
    }
}

