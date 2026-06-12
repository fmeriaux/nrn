//! Activation functions module.
//!
//! This module defines the `Activation` trait and provides a registry for activation functions used in neural networks.
//! It enables extensibility by allowing new activation functions to be added without modifying the core logic.
//! Each activation implements a common interface for forward and backward passes, and can be registered for dynamic lookup.
//! Typical activations include ReLU, Sigmoid, and Softmax, but the design supports custom user-defined activations as well.
//!
//! All built-in activations are registered using the `inventory` crate, allowing for easy discovery and use.
//!
mod relu;
mod sigmoid;
mod softmax;

pub use relu::RELU;
pub use sigmoid::SIGMOID;
pub use softmax::SOFTMAX;
use std::sync::Arc;

use crate::initializations::Initialization;
use ndarray::{Array2, ArrayView2};
use std::fmt::Debug;

pub trait Activation: Send + Sync + Debug {
    /// Returns the canonical name of the activation function.
    fn name(&self) -> &'static str;

    /// Applies the activation function element-wise to the input matrix.
    ///
    /// # Arguments
    /// * `input` - A 2D array of pre-activation values (logits) from the linear layer.
    ///
    /// # Returns
    /// A 2D array of the same dimensions containing activated outputs.
    ///
    /// This non-linear transformation enables the network to model complex patterns.
    fn apply(&self, input: ArrayView2<f32>) -> Array2<f32>;

    /// Computes the vector-Jacobian product (VJP) for backpropagation.
    ///
    /// Given the upstream gradient ∂L/∂a (with respect to post-activation values),
    /// returns ∂L/∂z (with respect to pre-activation values), correctly handling
    /// both diagonal (ReLU, Sigmoid) and full (Softmax) Jacobians.
    ///
    /// # Arguments
    /// * `upstream` - Incoming gradient ∂L/∂a, shape `(neurons, samples)`.
    /// * `activations` - Post-activation values from the forward pass, same shape.
    ///
    /// # Returns
    /// ∂L/∂z — gradient with respect to the pre-activation input.
    fn vjp(&self, upstream: ArrayView2<f32>, activations: ArrayView2<f32>) -> Array2<f32>;

    /// Provides an initialization method linked to this activation.
    ///
    /// This can be used to initialize the parameters of layers associated with this activation
    /// (e.g., for certain parametrized activation functions).
    fn initialization(&self) -> Arc<dyn Initialization>;
}

/// Registration struct for activation implementations.
/// This allows dynamic discovery and use of different activation functions
pub struct ActivationProvider(pub fn() -> Arc<dyn Activation>);

inventory::collect!(ActivationProvider);

impl ActivationProvider {
    /// Retrieves an activation implementation by its name.
    ///
    /// # Arguments
    /// * `name` - The canonical name of the activation function to retrieve.
    ///
    /// # Returns
    /// An `Option` containing the `Arc` to the activation if found, or `None` if not found.
    ///
    /// # Example
    /// ```
    /// use nrn::activations::ActivationProvider;
    /// use ndarray::array;
    ///
    /// let relu = ActivationProvider::get_by_name("relu").expect("relu is registered");
    /// assert_eq!(relu.name(), "relu");
    /// let input = array![[1.0, -1.0]];
    /// let output = relu.apply(input.view());
    /// assert_eq!(output[[0, 0]], 1.0); // positive value passes through
    /// assert_eq!(output[[0, 1]], 0.0); // negative value -> zero
    /// ```
    pub fn get_by_name(name: &str) -> Option<Arc<dyn Activation>> {
        for provider in inventory::iter::<ActivationProvider> {
            let activation = provider.0();
            if activation.name() == name {
                return Some(activation.clone());
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_by_name_returns_registered_activation() {
        let relu = ActivationProvider::get_by_name("relu").expect("relu is registered");
        assert_eq!(relu.name(), "relu");
    }

    #[test]
    fn get_by_name_returns_none_for_unknown_activation() {
        assert!(ActivationProvider::get_by_name("not-an-activation").is_none());
    }
}
