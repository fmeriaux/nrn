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

    /// Computes the element-wise derivative of the activation function for backpropagation.
    ///
    /// # Arguments
    ///
    /// * `activations` - A 2D array of post-activation values from the forward pass.
    ///
    /// # Returns
    ///
    /// A 2D array of the same shape containing `dσ/dz` at each position, used to scale
    /// the incoming gradient via element-wise multiplication in the chain rule.
    ///
    /// # Note
    ///
    /// This interface assumes a diagonal Jacobian (element-wise activations like ReLU and
    /// Sigmoid). Activations with a full Jacobian (e.g. Softmax) cannot implement this
    /// correctly; their output-layer gradient is handled separately by the loss function.
    fn derivative(&self, activations: ArrayView2<f32>) -> Array2<f32>;

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
