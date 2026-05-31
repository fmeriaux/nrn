use crate::model::NeuronLayer;
use crate::optimizers::Optimizer;
use crate::training::{Gradients, LearningRate};
use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Adam optimizer state, maintaining first and second moment estimates.
struct AdamState {
    /// First moment estimates for weights, averaging past gradients.
    m_weights: Array2<f32>,
    /// Second moment estimates for weights, variances of past gradients.
    v_weights: Array2<f32>,
    /// First moment estimates for biases, averaging past gradients.
    m_biases: Array1<f32>,
    /// Second moment estimates for biases, variances of past gradients.
    v_biases: Array1<f32>,
}

/// Provides the Adam optimization algorithm for updating neural network weights and biases.
pub struct Adam {
    /// Learning rate for the optimizer.
    learning_rate: LearningRate,
    /// Exponential decay rate for the first moment estimates.
    beta1: f32,
    /// Exponential decay rate for the second moment estimates.
    beta2: f32,
    /// Small constant for numerical stability.
    epsilon: f32,
    /// Step counter for bias correction.
    time_step: u32,
    /// Internal state for each layer, storing moment estimates.
    states: HashMap<usize, AdamState>,
}

impl Adam {
    /// Creates a new [`Adam`] optimizer with the specified parameters.
    /// # Panics
    /// Will panic if `beta1` or `beta2` are not in (0, 1) or if `epsilon` is not positive.
    pub fn new(learning_rate: LearningRate, beta1: f32, beta2: f32, epsilon: f32) -> Self {
        assert!(beta1 > 0.0 && beta1 < 1.0, "Beta1 must be in (0, 1).");
        assert!(beta2 > 0.0 && beta2 < 1.0, "Beta2 must be in (0, 1).");
        assert!(epsilon > 0.0, "Epsilon must be greater than zero.");

        Adam {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            time_step: 1,
            states: HashMap::new(),
        }
    }

    /// Creates an Adam optimizer with default parameters.
    pub fn with_defaults(learning_rate: LearningRate) -> Self {
        Self::new(learning_rate, 0.9, 0.999, 1e-8)
    }

    fn init_state(&mut self, layer_index: usize, layer: &NeuronLayer) {
        let m_weights = Array2::<f32>::zeros(layer.weights.dim());
        let v_weights = Array2::<f32>::zeros(layer.weights.dim());
        let m_biases = Array1::<f32>::zeros(layer.biases.dim());
        let v_biases = Array1::<f32>::zeros(layer.biases.dim());

        self.states.insert(
            layer_index,
            AdamState {
                m_weights,
                v_weights,
                m_biases,
                v_biases,
            },
        );
    }
}

impl Optimizer for Adam {
    fn set_learning_rate(&mut self, learning_rate: LearningRate) {
        self.learning_rate = learning_rate;
    }

    fn update(&mut self, layer_index: usize, layer: &mut NeuronLayer, gradients: &Gradients) {
        if !self.states.contains_key(&layer_index) {
            self.init_state(layer_index, layer);
        }

        let state = self.states.get_mut(&layer_index).unwrap();
        let (dw, db) = (&gradients.dw, &gradients.db);

        // Update biased first moment estimate
        state.m_weights = &state.m_weights * self.beta1 + dw * (1.0 - self.beta1);
        state.m_biases = &state.m_biases * self.beta1 + db * (1.0 - self.beta1);

        // Update biased second moment estimate
        state.v_weights = &state.v_weights * self.beta2 + &(dw * dw) * (1.0 - self.beta2);
        state.v_biases = &state.v_biases * self.beta2 + &(db * db) * (1.0 - self.beta2);

        let beta1_correction = 1.0 - self.beta1.powi(self.time_step as i32);
        let beta2_correction = 1.0 - self.beta2.powi(self.time_step as i32);

        // Compute bias-corrected first and second moment estimates
        let m_hat_weights = &state.m_weights / beta1_correction;
        let m_hat_biases = &state.m_biases / beta1_correction;
        let v_hat_weights = &state.v_weights / beta2_correction;
        let v_hat_biases = &state.v_biases / beta2_correction;

        // Update weights and biases
        layer.weights -= &(m_hat_weights * self.learning_rate.value()
            / (v_hat_weights.mapv(f32::sqrt) + self.epsilon));
        layer.biases -= &(m_hat_biases * self.learning_rate.value()
            / (v_hat_biases.mapv(f32::sqrt) + self.epsilon));
    }

    fn step(&mut self) {
        self.time_step += 1;
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

    /// Checks a trained weight has converged near zero, reporting the offending
    /// value on failure. Returning a `Result` (rather than asserting inline)
    /// keeps the diagnostic message and lets a unit test exercise the failure
    /// path without panicking the suite.
    fn converged_near_zero(weight: f32, tol: f32) -> Result<(), String> {
        if weight.abs() < tol {
            Ok(())
        } else {
            Err(format!("weight should approach 0, got {weight}"))
        }
    }

    #[test]
    fn converged_near_zero_reports_the_offending_weight() {
        assert!(converged_near_zero(0.1, 0.5).is_ok());
        assert_eq!(
            converged_near_zero(1.5, 0.5).unwrap_err(),
            "weight should approach 0, got 1.5"
        );
    }

    #[test]
    #[should_panic(expected = "Beta1 must be in (0, 1)")]
    fn adam_rejects_invalid_beta1() {
        Adam::new(LearningRate::new(0.01), 1.5, 0.999, 1e-8);
    }

    #[test]
    fn adam_first_step_moves_by_approximately_learning_rate() {
        // On the first update, m_hat = grad and v_hat = grad^2, so the step is
        // lr * grad / |grad| = lr * sign(grad), independent of the gradient magnitude.
        let lr = 0.01;
        let mut opt = Adam::with_defaults(LearningRate::new(lr));
        let mut l = layer(array![[1.0, 1.0]], array![1.0]);
        let grads = Gradients {
            dw: array![[2.0, -3.0]],
            db: array![0.5],
        };
        opt.update(0, &mut l, &grads);
        assert!((l.weights[[0, 0]] - (1.0 - lr)).abs() < 1e-4); // +grad → decrease
        assert!((l.weights[[0, 1]] - (1.0 + lr)).abs() < 1e-4); // -grad → increase
        assert!((l.biases[0] - (1.0 - lr)).abs() < 1e-4);
    }

    #[test]
    fn adam_zero_gradient_leaves_params_unchanged() {
        let mut opt = Adam::with_defaults(LearningRate::new(0.1));
        let mut l = layer(array![[1.0, -2.0]], array![3.0]);
        let grads = Gradients {
            dw: Array2::zeros((1, 2)),
            db: Array1::zeros(1),
        };
        opt.update(0, &mut l, &grads);
        assert!((l.weights[[0, 0]] - 1.0).abs() < 1e-6);
        assert!((l.weights[[0, 1]] - (-2.0)).abs() < 1e-6);
        assert!((l.biases[0] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn adam_minimizes_simple_quadratic() {
        // f(w) = 0.5 * w^2 has gradient w and minimum at 0; Adam should drive w → 0.
        let mut opt = Adam::with_defaults(LearningRate::new(0.1));
        let mut l = layer(array![[5.0]], array![0.0]);
        for _ in 0..200 {
            let w = l.weights[[0, 0]];
            let grads = Gradients {
                dw: array![[w]],
                db: array![0.0],
            };
            opt.update(0, &mut l, &grads);
            opt.step();
        }
        converged_near_zero(l.weights[[0, 0]], 0.5).unwrap();
    }
}
