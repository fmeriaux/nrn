mod relu;
mod sigmoid;
mod softmax;

pub use relu::ReLU;
pub use sigmoid::Sigmoid;
pub use softmax::Softmax;
use std::collections::HashMap;
use std::sync::Arc;

use crate::core::initialization::Initializer;
use ndarray::Array2;

/// Represents an activation function that can be applied to neural network layers.
pub trait Activation: Send + Sync {
    /// Returns the name of the activation function
    fn name(&self) -> &str;

    /// Applies the activation function to the input
    fn apply(&self, input: &Array2<f32>) -> Array2<f32>;

    /// Returns the associated initializer for the activation function
    fn get_initializer(&self) -> Box<dyn Initializer>;
}

pub struct ActivationRegistry {
    map: HashMap<String, Arc<dyn Activation>>,
}

impl ActivationRegistry {
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

    pub fn get(&self, name: &str) -> Option<Arc<dyn Activation>> {
        self.map.get(name).cloned()
    }
}
