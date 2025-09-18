use crate::core::activations::Activation;
use crate::core::initialization::{Initializer, XavierUniform};
use ndarray::{Array2, Axis};

pub struct Softmax;

impl Activation for Softmax {
    fn name(&self) -> &str {
        "softmax"
    }

    fn apply(&self, input: &Array2<f32>) -> Array2<f32> {
        let max = input.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_output = input.mapv(|val| (val - max).exp());
        let sum = exp_output.sum_axis(Axis(0));
        exp_output / sum
    }

    fn get_initializer(&self) -> Box<dyn Initializer> {
        Box::new(XavierUniform)
    }
}
