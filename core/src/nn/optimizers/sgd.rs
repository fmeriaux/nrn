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
//! use crate::optimizers::{Optimizer, StochasticGradientDescent};
//! let mut optimizer = StochasticGradientDescent::new(0.01);
//! optimizer.update(&mut layer, &gradients);
//! ```

use crate::model::NeuronLayer;
use crate::optimizers::Optimizer;
use crate::training::{Gradients, LearningRate};

pub struct StochasticGradientDescent {
    pub learning_rate: LearningRate,
}

impl StochasticGradientDescent {
    pub fn new(learning_rate: LearningRate) -> Self {
        StochasticGradientDescent { learning_rate }
    }
}

impl Optimizer for StochasticGradientDescent {
    fn set_learning_rate(&mut self, learning_rate: LearningRate) {
        self.learning_rate = learning_rate;
    }

    fn update(&mut self, _: usize, layer: &mut NeuronLayer, gradients: &Gradients) {
        let (dw, db) = (&gradients.dw, &gradients.db);
        layer.weights -= &(dw * self.learning_rate.value());
        layer.biases -= &(db * self.learning_rate.value());
    }
}
