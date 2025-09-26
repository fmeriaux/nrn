//! Stochastic Gradient Descent (SGD) optimizer implementation.
//!
//! This module provides the `StochasticGradientDescent` struct, which implements the `Optimizer` trait
//! for the classic stochastic gradient descent algorithm. SGD updates the weights and biases of a layer
//! by subtracting the product of the learning rate and the computed gradients. It is simple, efficient,
//! and widely used for many machine learning problems.
//!
//! - Does not maintain any internal state (unlike Adam, RMSProp, etc.).
//!
//! # Example
//! ```rust
//! use crate::core::optimizers::{Optimizer, StochasticGradientDescent};
//! let mut optimizer = StochasticGradientDescent::new(0.01);
//! optimizer.update(&mut layer, &gradients);
//! ```


use crate::core::model::NeuronLayer;
use crate::core::optimizers::Optimizer;
use crate::core::training::Gradients;

pub struct StochasticGradientDescent {
    pub learning_rate: f32,
}

impl Optimizer for StochasticGradientDescent {
    fn update(&mut self, layer: &mut NeuronLayer, gradients: &Gradients) {
        let (dw, db) = (&gradients.dw, &gradients.db);
        layer.weights -= &(dw * self.learning_rate);
        layer.bias -= &(db * self.learning_rate);
    }
}

impl StochasticGradientDescent {
    pub fn new(learning_rate: f32) -> Self {
        assert!(
            learning_rate > 0.0,
            "Learning rate must be greater than zero."
        );
        StochasticGradientDescent { learning_rate }
    }
}

impl Default for StochasticGradientDescent {
    fn default() -> Self {
        Self::new(0.001)
    }
}
