use std::collections::HashMap;
use crate::model::NeuronLayer;
use crate::optimizers::Optimizer;
use crate::training::Gradients;
use ndarray::{Array1, Array2};

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
    pub learning_rate: f32,
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
    pub fn new(learning_rate: f32, beta1: f32, beta2: f32, epsilon: f32) -> Self {
        assert!(
            learning_rate > 0.0,
            "Learning rate must be greater than zero."
        );
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
    pub fn with_defaults(learning_rate: f32) -> Self {
        Self::new(learning_rate, 0.9, 0.999, 1e-8)
    }

    fn init_state(&mut self, layer_index: usize, layer: &NeuronLayer) {
        let m_weights = Array2::<f32>::zeros(layer.weights.dim());
        let v_weights = Array2::<f32>::zeros(layer.weights.dim());
        let m_biases = Array1::<f32>::zeros(layer.biases.dim());
        let v_biases = Array1::<f32>::zeros(layer.biases.dim());

        self.states.insert(layer_index, AdamState {
            m_weights,
            v_weights,
            m_biases,
            v_biases,
        });
    }
}

impl Optimizer for Adam {
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

        // Compute bias-corrected first moment estimate
        let m_hat_weights = &state.m_weights / (1.0 - self.beta1.powi(self.time_step as i32));
        let m_hat_biases = &state.m_biases / (1.0 - self.beta1.powi(self.time_step as i32));

        // Compute bias-corrected second moment estimate
        let v_hat_weights = &state.v_weights / (1.0 - self.beta2.powi(self.time_step as i32));
        let v_hat_biases = &state.v_biases / (1.0 - self.beta2.powi(self.time_step as i32));

        // Update weights and biases
        layer.weights -= &(m_hat_weights * self.learning_rate / (v_hat_weights.mapv(f32::sqrt) + self.epsilon));
        layer.biases -= &(m_hat_biases * self.learning_rate / (v_hat_biases.mapv(f32::sqrt) + self.epsilon));
    }

    fn step(&mut self) {
        self.time_step += 1;
    }
}