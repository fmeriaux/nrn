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

    fn derivative(&self, activations: &Array2<f32>, _: &Array2<f32>) -> Array2<f32> {
        activations.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 })
    }

    fn get_initializer(&self) -> Box<dyn Initializer> {
        Box::new(HeUniform)
    }
}
