use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use crate::core::activation::ActivationMethod;

/// Represents the activation methods available for neurons in a neural network.
#[derive(Clone)]
pub enum InitializationMethod {
    HeUniform,
    XavierUniform,
}

/// Generates a uniform distribution of weights and biases based on the specified shape.
fn uniform_distribution(shape: (usize, usize), distribution: &Uniform<f32>) -> (Array2<f32>, Array1<f32>) {
    let weights = Array2::random(shape, distribution);
    let biases = Array1::zeros(shape.0);
    (weights, biases)
}

impl InitializationMethod {
    
    /// Derives an initialization method based on the activation method.
    pub fn from_activation(activation: &ActivationMethod) -> Self {
        match activation {
            ActivationMethod::ReLU => InitializationMethod::HeUniform,
            _ => InitializationMethod::XavierUniform,
        }
    }
    
    /// Initializes weights and biases based on the specified method and shape.
    pub fn apply(&self, shape: (usize, usize)) -> (Array2<f32>, Array1<f32>) {
        match self {
            InitializationMethod::HeUniform => {
                let limit = (6.0 / shape.1 as f32).sqrt();
                uniform_distribution(shape, &Uniform::new(-limit, limit))
            },
            InitializationMethod::XavierUniform => {
                let limit = (6.0 / (shape.1 + shape.0) as f32).sqrt();
                uniform_distribution(shape, &Uniform::new(-limit, limit))
            },
        }
    }
}