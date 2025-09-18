mod he;
mod xavier;

pub use he::HeUniform;
pub use xavier::XavierUniform;

use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

/// Represents a strategy for initializing weights and biases in a neural network.
pub trait Initializer: Send + Sync {
    /// Applies the initialization method to generate weights and biases based on the specified shape.
    fn apply(&self, shape: (usize, usize)) -> (Array2<f32>, Array1<f32>);
}

fn uniform_distribution(shape: (usize, usize), distribution: &Uniform<f32>) -> (Array2<f32>, Array1<f32>) {
    let weights = Array2::random(shape, distribution);
    let biases = Array1::zeros(shape.0);
    (weights, biases)
}