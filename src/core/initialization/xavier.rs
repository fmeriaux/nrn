use crate::core::initialization::{Initializer, uniform_distribution};
use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Uniform;
use once_cell::sync::Lazy;
use std::sync::Arc;

pub struct XavierUniform;

impl Initializer for XavierUniform {
    fn apply(&self, shape: (usize, usize)) -> (Array2<f32>, Array1<f32>) {
        let limit = (6.0 / (shape.1 + shape.0) as f32).sqrt();
        uniform_distribution(shape, &Uniform::new(-limit, limit))
    }
}

pub static XAVIER_UNIFORM: Lazy<Arc<XavierUniform>> = Lazy::new(|| Arc::new(XavierUniform));
