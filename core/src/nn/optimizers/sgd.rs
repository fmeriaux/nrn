//! Stochastic Gradient Descent (SGD) optimizer implementation.
//!
//! This module provides the `StochasticGradientDescent` struct, which implements the `Optimizer` trait
//! for the classic stochastic gradient descent algorithm. SGD updates the weights and biases of a layer
//! by subtracting the product of the learning rate and the computed gradients. It is simple, efficient,
//! and widely used for many machine learning problems.
//!
//! - Does not maintain any internal state (unlike Adam, RMSProp, etc.).
//!

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::RELU;
    use ndarray::{Array1, Array2, array};

    fn layer(weights: Array2<f32>, biases: Array1<f32>) -> NeuronLayer {
        NeuronLayer {
            weights,
            biases,
            activation: RELU.clone(),
        }
    }

    #[test]
    fn sgd_subtracts_scaled_gradient() {
        let mut opt = StochasticGradientDescent::new(LearningRate::new(0.1));
        let mut l = layer(array![[1.0, 2.0]], array![1.0]);
        let grads = Gradients {
            dw: array![[0.5, 1.0]],
            db: array![2.0],
        };
        opt.update(0, &mut l, &grads);
        // params -= learning_rate * gradient
        assert!((l.weights[[0, 0]] - 0.95).abs() < 1e-6);
        assert!((l.weights[[0, 1]] - 1.9).abs() < 1e-6);
        assert!((l.biases[0] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn sgd_zero_gradient_leaves_params_unchanged() {
        let mut opt = StochasticGradientDescent::new(LearningRate::new(0.5));
        let mut l = layer(array![[1.0, -2.0]], array![3.0]);
        let grads = Gradients {
            dw: Array2::zeros((1, 2)),
            db: Array1::zeros(1),
        };
        opt.update(0, &mut l, &grads);
        assert_eq!(l.weights, array![[1.0, -2.0]]);
        assert_eq!(l.biases, array![3.0]);
    }
}
