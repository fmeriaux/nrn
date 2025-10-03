use std::sync::Arc;
use crate::initializations::Initialization;
use crate::initializations::uniform_distribution;
use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Uniform;
use once_cell::sync::Lazy;

/// HeUniform initializer for neural network weights.
///
/// Draws weights from a uniform distribution within [-limit, limit],
/// where limit = sqrt(6 / fan_in). This helps maintain variance
/// in deep networks, especially with ReLU activations.
pub struct HeUniform;

impl Initialization for HeUniform {
    /// Apply He uniform initialization to a weight matrix and bias vector.
    ///
    /// # Arguments
    /// * `shape` - Tuple (fan_out, fan_in) for the weight matrix.
    ///
    /// # Returns
    /// Tuple of (weights, biases) as ndarray arrays.
    fn apply(&self, shape: (usize, usize)) -> (Array2<f32>, Array1<f32>) {
        let limit = (6.0 / shape.1 as f32).sqrt();
        uniform_distribution(shape, &Uniform::new(-limit, limit))
    }
}

/// Global static instance of HeUniform initializer.
pub static HE_UNIFORM: Lazy<Arc<HeUniform>> = Lazy::new(|| Arc::new(HeUniform));
