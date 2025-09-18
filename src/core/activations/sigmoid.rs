use crate::core::activations::Activation;
use crate::core::initialization::{Initializer, XavierUniform};
use ndarray::Array2;

pub struct Sigmoid;

impl Activation for Sigmoid {
    fn name(&self) -> &str {
        "sigmoid"
    }

    fn apply(&self, input: &Array2<f32>) -> Array2<f32> {
        input.mapv(|x| 1.0 / (1.0 + (-x).exp()))
    }

    fn derivative(&self, activations: &Array2<f32>, _: &Array2<f32>) -> Array2<f32> {
        activations.mapv(|s| s * (1.0 - s))
    }

    fn get_initializer(&self) -> Box<dyn Initializer> {
        Box::new(XavierUniform)
    }
}
