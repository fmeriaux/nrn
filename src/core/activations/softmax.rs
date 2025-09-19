use crate::core::activations::Activation;
use crate::core::initialization::{Initializer, XAVIER_UNIFORM};
use ndarray::Array2;
use once_cell::sync::Lazy;
use std::sync::Arc;

pub struct Softmax;

impl Activation for Softmax {
    fn name(&self) -> &'static str {
        "softmax"
    }

    fn apply(&self, input: &Array2<f32>) -> Array2<f32> {
        let mut result = input.clone();

        for mut col in result.columns_mut() {
            // Numerical stability, subtract max value from each element
            let max_val = *col.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
            col.mapv_inplace(|x| (x - max_val).exp());

            // Normalize to get probabilities
            col /= col.sum();
        }

        result
    }

    fn derivative(&self, activations: &Array2<f32>, targets: &Array2<f32>) -> Array2<f32> {
        let batch_size = activations.ncols() as f32;
        (activations - targets) / batch_size
    }

    fn get_initializer(&self) -> Arc<dyn Initializer> {
        XAVIER_UNIFORM.clone()
    }
}

pub static SOFTMAX: Lazy<Arc<Softmax>> = Lazy::new(|| Arc::new(Softmax));
