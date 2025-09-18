use crate::core::activations::Activation;
use crate::core::initialization::{HeUniform, Initializer};
use ndarray::Array2;

pub struct ReLU;

impl Activation for ReLU {
    fn name(&self) -> &str {
        "relu"
    }

    fn apply(&self, input: &Array2<f32>) -> Array2<f32> {
        input.mapv(|x| x.max(0.0))
    }

    fn get_initializer(&self) -> Box<dyn Initializer> {
        Box::new(HeUniform)
    }
}
