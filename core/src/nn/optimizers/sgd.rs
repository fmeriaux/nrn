//! Stochastic Gradient Descent (SGD) optimizer implementation.
//!
//! This module provides the `StochasticGradientDescent` struct, which implements the `Optimizer` trait
//! for the classic stochastic gradient descent algorithm. SGD updates the weights and biases of a layer
//! by subtracting the product of the learning rate and the computed gradients. It is simple, efficient,
//! and widely used for many machine learning problems.
//!
//! - Does not maintain any internal state (unlike Adam, RMSProp, etc.).
//!

use crate::gradients::Gradients;
use crate::learning_rate::LearningRate;
use crate::model::NeuronLayer;
use crate::optimizers::Optimizer;
use crate::weight_decay::WeightDecay;

pub struct StochasticGradientDescent {
    learning_rate: LearningRate,
    weight_decay: WeightDecay,
}

impl StochasticGradientDescent {
    pub fn new(learning_rate: LearningRate, weight_decay: WeightDecay) -> Self {
        StochasticGradientDescent {
            learning_rate,
            weight_decay,
        }
    }
}

impl Optimizer for StochasticGradientDescent {
    fn name(&self) -> &'static str {
        "Stochastic Gradient Descent (SGD)"
    }

    fn learning_rate(&self) -> LearningRate {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, learning_rate: LearningRate) {
        self.learning_rate = learning_rate;
    }

    fn update(&mut self, _: usize, layer: &mut NeuronLayer, gradients: &Gradients) {
        let (dw, db) = (&gradients.dw, &gradients.db);
        // L2 weight decay: shrink the weights before the gradient step. Biases are
        // not decayed.
        if self.weight_decay.is_active() {
            layer.weights *= 1.0 - self.learning_rate.value() * self.weight_decay.value();
        }
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
        let mut opt = StochasticGradientDescent::new(0.1.try_into().unwrap(), WeightDecay::ZERO);
        assert_eq!(opt.name(), "Stochastic Gradient Descent (SGD)");
        assert_eq!(opt.learning_rate().value(), 0.1);
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
        let mut opt = StochasticGradientDescent::new(0.5.try_into().unwrap(), WeightDecay::ZERO);
        let mut l = layer(array![[1.0, -2.0]], array![3.0]);
        let grads = Gradients {
            dw: Array2::zeros((1, 2)),
            db: Array1::zeros(1),
        };
        opt.update(0, &mut l, &grads);
        assert_eq!(l.weights, array![[1.0, -2.0]]);
        assert_eq!(l.biases, array![3.0]);
    }

    #[test]
    fn set_learning_rate_changes_the_update_magnitude() {
        let mut opt = StochasticGradientDescent::new(0.1.try_into().unwrap(), WeightDecay::ZERO);
        opt.set_learning_rate(LearningRate::new(1.0).unwrap());

        let mut l = layer(array![[1.0]], array![0.0]);
        let grads = Gradients {
            dw: array![[0.5]],
            db: array![0.0],
        };
        opt.update(0, &mut l, &grads);
        // With learning rate 1.0 the full gradient is subtracted: 1.0 - 1.0 * 0.5.
        assert!((l.weights[[0, 0]] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn weight_decay_shrinks_weights_and_spares_biases() {
        let lr = 0.1;
        let wd = WeightDecay::new(0.5).unwrap();
        let mut opt = StochasticGradientDescent::new(lr.try_into().unwrap(), wd);
        let mut l = layer(array![[2.0]], array![3.0]);
        let grads = Gradients {
            dw: array![[1.0]],
            db: array![1.0],
        };
        opt.update(0, &mut l, &grads);
        // w = w*(1 - lr*wd) - lr*dw = 2*0.95 - 0.1*1 = 1.8; bias = 3 - 0.1*1 = 2.9 (no decay).
        assert!((l.weights[[0, 0]] - 1.8).abs() < 1e-6);
        assert!((l.biases[0] - 2.9).abs() < 1e-6);
    }

    #[test]
    fn step_is_a_noop_for_stateless_sgd() {
        // SGD keeps no internal state, so it relies on the trait's default `step`.
        let mut opt = StochasticGradientDescent::new(0.1.try_into().unwrap(), WeightDecay::ZERO);
        opt.step();
        assert_eq!(opt.learning_rate().value(), 0.1);
    }
}
