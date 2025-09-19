mod relu;
mod sigmoid;
mod softmax;

pub use relu::RELU;
pub use sigmoid::SIGMOID;
pub use softmax::SOFTMAX;
use std::collections::HashMap;
use std::sync::Arc;

use crate::core::initialization::Initializer;
use ndarray::Array2;

/// Represents an activation function applicable to neural network layers.
pub trait Activation: Send + Sync {
    /// Returns the canonical name of the activation function.
    fn name(&self) -> &'static str;

    /// Applies the activation function element-wise to the input matrix.
    ///
    /// # Arguments
    ///
    /// * `input` - A 2D array of pre-activation values (logits) from the linear layer.
    ///
    /// # Returns
    ///
    /// A 2D array of the same dimensions containing activated outputs.
    ///
    /// This non-linear transformation enables the network to model complex patterns.
    fn apply(&self, input: &Array2<f32>) -> Array2<f32>;

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
    fn derivative(&self, activations: &Array2<f32>, targets: &Array2<f32>) -> Array2<f32>;

    /// Provides an initializer instance linked to this activation.
    ///
    /// This can be used to initialize the parameters of layers associated with this activation
    /// (e.g., for certain parametrized activation functions).
    fn get_initializer(&self) -> Box<dyn Initializer>;
}

/// A registry to store and manage activation functions.
///
/// It maps activation function names to their respective implementations,
/// allowing easy lookup and reuse throughout the neural network code.
pub struct ActivationRegistry {
    map: HashMap<String, Arc<dyn Activation>>,
}

impl ActivationRegistry {
    /// Creates a new `ActivationRegistry` from an iterable of activations.
    ///
    /// # Arguments
    ///
    /// * `activations` - An iterable collection of activation functions wrapped in `Arc`.
    ///
    /// # Returns
    ///
    /// A new registry mapping activation names to activation instances.
    ///
    pub fn new<I>(activations: I) -> Self
    where
        I: IntoIterator<Item = Arc<dyn Activation>>,
    {
        let map: HashMap<String, Arc<dyn Activation>> = activations
            .into_iter()
            .map(|a| (a.name().to_string(), a))
            .collect();

        ActivationRegistry { map }
    }

    /// Retrieves an activation function from the registry by name.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the activation function to retrieve.
    ///
    /// # Returns
    ///
    /// An `Option` containing a cloned `Arc` to the activation if found, or `None` otherwise.
    ///
    pub fn get(&self, name: &str) -> Option<Arc<dyn Activation>> {
        self.map.get(name).cloned()
    }
}
