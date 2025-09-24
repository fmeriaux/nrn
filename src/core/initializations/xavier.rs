use crate::core::initializations::{Initialization, uniform_distribution};
use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Uniform;
use once_cell::sync::Lazy;
use std::sync::Arc;

/// XavierUniform initializer for neural network weights.
///
/// Draws weights from a uniform distribution within [-limit, limit],
/// where limit = sqrt(6 / (fan_in + fan_out)). This helps maintain variance
/// in deep networks, especially with sigmoid or tanh activations.
pub struct XavierUniform;

impl Initialization for XavierUniform {
    /// Apply Xavier uniform initialization to a weight matrix and bias vector.
    ///
    /// # Arguments
    /// * `shape` - Tuple (fan_out, fan_in) for the weight matrix.
    ///
    /// # Returns
    /// Tuple of (weights, biases) as ndarray arrays.
    fn apply(&self, shape: (usize, usize)) -> (Array2<f32>, Array1<f32>) {
        let limit = (6.0 / (shape.1 + shape.0) as f32).sqrt();
        uniform_distribution(shape, &Uniform::new(-limit, limit))
    }
}

/// Global static instance of XavierUniform initializer.
pub static XAVIER_UNIFORM: Lazy<Arc<XavierUniform>> = Lazy::new(|| Arc::new(XavierUniform));
