mod he;
mod xavier;

pub use he::HE_UNIFORM;
pub use xavier::XAVIER_UNIFORM;

use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

/// Defines a strategy for initializing weights and biases in a neural network.
///
/// Proper initialization is crucial to ensure stable and efficient training,
/// avoiding problems like vanishing or exploding gradients.
pub trait Initializer: Send + Sync {
    /// Generates initial weights and biases according to the initialization method.
    ///
    /// # Arguments
    ///
    /// * `shape` - A tuple `(rows, cols)` specifying the desired shape of the weight matrix.
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// - A 2D array of floats representing initialized weights,
    /// - A 1D array of floats representing initialized biases.
    ///
    /// This allows layers to start training from a well-defined state,
    /// tailored to the activation function or network architecture.
    fn apply(&self, shape: (usize, usize)) -> (Array2<f32>, Array1<f32>);
}

fn uniform_distribution(shape: (usize, usize), distribution: &Uniform<f32>) -> (Array2<f32>, Array1<f32>) {
    let weights = Array2::random(shape, distribution);
    let biases = Array1::zeros(shape.0);
    (weights, biases)
}