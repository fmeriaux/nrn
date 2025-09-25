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

use crate::core::initializations::Initialization;
use ndarray::{Array2, ArrayView2};

pub trait Activation: Send + Sync {
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

    /// Computes the derivative of the activation function for backpropagation.
    ///
    /// # Arguments
    ///
    /// * `activations` - A 2D array of activation values, typically the output from the forward pass.
    /// * `targets` - An optional 2D array of target values used for specific activations like softmax,
    ///               where the derivative depends on the expected outputs.
    ///
    /// # Returns
    ///
    /// A 2D array containing the gradient of the loss with respect to the input of the activation function.
    /// For element-wise activation functions like sigmoid or ReLU, this is computed element-wise.
    /// For activations like softmax combined with cross-entropy loss, the derivative may use `targets`
    /// to compute the combined gradient directly.
    ///
    /// # Explanation
    ///
    /// The returned derivative matrix represents the local gradient needed for backpropagation.
    /// In typical element-wise activations, this is the element-wise derivative matrix that multiplies the
    /// gradient propagated from the next layer.
    /// In the case of softmax (or other complex activations), the derivative may be computed differently
    /// using the targets, reflecting the special form of the gradient when combined with loss functions.
    fn derivative(&self, activations: ArrayView2<f32>, targets: ArrayView2<f32>) -> Array2<f32>;

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
    /// if let Some(activation) = ActivationProvider::get_by_name("relu") {
    ///     // Use the activation
    /// }
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
