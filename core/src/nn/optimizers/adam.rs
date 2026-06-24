use crate::gradients::Gradients;
use crate::learning_rate::LearningRate;
use crate::model::NeuronLayer;
use crate::optimizers::{Optimizer, OptimizerState, OptimizerStateError};
use crate::weight_decay::WeightDecay;
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
    /// Decoupled weight-decay coefficient (AdamW). Applied to weights only, never
    /// biases.
    weight_decay: WeightDecay,
    /// Step counter for bias correction.
    time_step: u32,
    /// Internal state for each layer, storing moment estimates.
    states: HashMap<usize, AdamState>,
}

impl Adam {
    /// Creates a new [`Adam`] optimizer with the specified parameters. A non-zero
    /// `weight_decay` applies decoupled decay (AdamW), directly to the weights
    /// rather than through the moment estimates.
    /// # Panics
    /// Will panic if `beta1` or `beta2` are not in (0, 1) or if `epsilon` is not positive.
    pub fn new(
        learning_rate: LearningRate,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: WeightDecay,
    ) -> Self {
        assert!(beta1 > 0.0 && beta1 < 1.0, "Beta1 must be in (0, 1).");
        assert!(beta2 > 0.0 && beta2 < 1.0, "Beta2 must be in (0, 1).");
        assert!(epsilon > 0.0, "Epsilon must be greater than zero.");

        Adam {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            time_step: 1,
            states: HashMap::new(),
        }
    }

    /// Creates an Adam optimizer with default moment parameters.
    pub fn with_defaults(learning_rate: LearningRate, weight_decay: WeightDecay) -> Self {
        // 1e-5 instead of the paper's 1e-8: f32 squared gradients near 1e-19 underflow
        // to 0, leaving epsilon as the sole denominator and inflating the effective step.
        Self::new(learning_rate, 0.9, 0.999, 1e-5, weight_decay)
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
    fn name(&self) -> &'static str {
        if self.weight_decay.is_active() {
            "AdamW"
        } else {
            "Adam"
        }
    }

    fn learning_rate(&self) -> LearningRate {
        self.learning_rate
    }

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

        // Decoupled weight decay (AdamW): shrink the weights before the gradient
        // step. Biases are not decayed.
        if self.weight_decay.is_active() {
            layer.weights *= 1.0 - self.learning_rate.value() * self.weight_decay.value();
        }

        // Update weights and biases
        layer.weights -= &(m_hat_weights * self.learning_rate.value()
            / (v_hat_weights.mapv(f32::sqrt) + self.epsilon));
        layer.biases -= &(m_hat_biases * self.learning_rate.value()
            / (v_hat_biases.mapv(f32::sqrt) + self.epsilon));
    }

    fn step(&mut self) {
        self.time_step += 1;
    }

    fn to_state(&self) -> Option<OptimizerState> {
        let mut tensors = Vec::with_capacity(self.states.len() * 4);
        for (layer_index, state) in &self.states {
            tensors.push((
                format!("layer{layer_index}.m_weights"),
                state.m_weights.clone().into_dyn(),
            ));
            tensors.push((
                format!("layer{layer_index}.v_weights"),
                state.v_weights.clone().into_dyn(),
            ));
            tensors.push((
                format!("layer{layer_index}.m_biases"),
                state.m_biases.clone().into_dyn(),
            ));
            tensors.push((
                format!("layer{layer_index}.v_biases"),
                state.v_biases.clone().into_dyn(),
            ));
        }

        let mut metadata = HashMap::new();
        metadata.insert("time_step".to_string(), self.time_step.to_string());

        Some(OptimizerState { tensors, metadata })
    }

    fn restore(&mut self, state: &OptimizerState) -> Result<(), OptimizerStateError> {
        self.time_step = state
            .metadata
            .get("time_step")
            .ok_or_else(|| OptimizerStateError::MissingMetadata("time_step".to_string()))?
            .parse()
            .map_err(|_| OptimizerStateError::InvalidMetadata {
                key: "time_step".to_string(),
            })?;

        let mut partials: HashMap<usize, PartialAdamState> = HashMap::new();

        for (name, array) in &state.tensors {
            let Some((layer, field)) = name.split_once('.') else {
                continue;
            };
            let Some(layer_index) = layer.strip_prefix("layer").and_then(|s| s.parse().ok()) else {
                continue;
            };

            let partial = partials.entry(layer_index).or_default();
            match field {
                "m_weights" => partial.m_weights = Some(to_array2(name, array)?),
                "v_weights" => partial.v_weights = Some(to_array2(name, array)?),
                "m_biases" => partial.m_biases = Some(to_array1(name, array)?),
                "v_biases" => partial.v_biases = Some(to_array1(name, array)?),
                _ => continue,
            }
        }

        for (layer_index, partial) in partials {
            self.states
                .insert(layer_index, partial.into_state(layer_index)?);
        }

        Ok(())
    }
}

/// Per-layer Adam moment tensors collected from an [`OptimizerState`] while
/// loading, before being assembled into an [`AdamState`].
#[derive(Default)]
struct PartialAdamState {
    m_weights: Option<Array2<f32>>,
    v_weights: Option<Array2<f32>>,
    m_biases: Option<Array1<f32>>,
    v_biases: Option<Array1<f32>>,
}

impl PartialAdamState {
    fn into_state(self, layer_index: usize) -> Result<AdamState, OptimizerStateError> {
        let missing =
            |field: &str| OptimizerStateError::MissingTensor(format!("layer{layer_index}.{field}"));

        Ok(AdamState {
            m_weights: self.m_weights.ok_or_else(|| missing("m_weights"))?,
            v_weights: self.v_weights.ok_or_else(|| missing("v_weights"))?,
            m_biases: self.m_biases.ok_or_else(|| missing("m_biases"))?,
            v_biases: self.v_biases.ok_or_else(|| missing("v_biases"))?,
        })
    }
}

fn to_array2(name: &str, array: &ndarray::ArrayD<f32>) -> Result<Array2<f32>, OptimizerStateError> {
    array
        .clone()
        .into_dimensionality()
        .map_err(|_| OptimizerStateError::WrongRank {
            tensor: name.to_string(),
            expected: 2,
        })
}

fn to_array1(name: &str, array: &ndarray::ArrayD<f32>) -> Result<Array1<f32>, OptimizerStateError> {
    array
        .clone()
        .into_dimensionality()
        .map_err(|_| OptimizerStateError::WrongRank {
            tensor: name.to_string(),
            expected: 1,
        })
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
    fn adam_default_epsilon_is_stable_for_f32() {
        // 1e-8 (paper default, f64) is too small for f32: squared gradients near 1e-19
        // underflow to 0, and the floor epsilon alone drives the step, causing divergence.
        // 1e-5 keeps the denominator above the f32 subnormal range.
        let opt = Adam::with_defaults(0.001.try_into().unwrap(), WeightDecay::ZERO);
        assert_eq!(opt.epsilon, 1e-5);
    }

    #[test]
    #[should_panic(expected = "Beta1 must be in (0, 1)")]
    fn adam_rejects_invalid_beta1() {
        Adam::new(
            LearningRate::new(0.01).unwrap(),
            1.5,
            0.999,
            1e-8,
            WeightDecay::ZERO,
        );
    }

    #[test]
    fn adam_first_step_moves_by_approximately_learning_rate() {
        // On the first update, m_hat = grad and v_hat = grad^2, so the step is
        // lr * grad / |grad| = lr * sign(grad), independent of the gradient magnitude.
        let lr = 0.01;
        let mut opt = Adam::with_defaults(lr.try_into().unwrap(), WeightDecay::ZERO);
        assert_eq!(opt.name(), "Adam");
        assert_eq!(opt.learning_rate().value(), lr);
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
        let mut opt = Adam::with_defaults(0.1.try_into().unwrap(), WeightDecay::ZERO);
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
        let mut opt = Adam::with_defaults(0.1.try_into().unwrap(), WeightDecay::ZERO);
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
        let w = l.weights[[0, 0]];
        assert!(w.abs() < 0.5, "weight should approach 0, got {w}");
    }

    #[test]
    fn weight_decay_renames_optimizer_to_adamw() {
        let plain = Adam::with_defaults(0.01.try_into().unwrap(), WeightDecay::ZERO);
        assert_eq!(plain.name(), "Adam");
        let decayed =
            Adam::with_defaults(0.01.try_into().unwrap(), WeightDecay::new(0.01).unwrap());
        assert_eq!(decayed.name(), "AdamW");
    }

    #[test]
    fn weight_decay_shrinks_weights_under_zero_gradient() {
        // With no gradient, plain Adam leaves the weights put; AdamW's decoupled
        // decay still pulls them toward zero, while leaving the bias untouched.
        let lr = 0.1;
        let wd = WeightDecay::new(0.1).unwrap();
        let mut opt = Adam::with_defaults(lr.try_into().unwrap(), wd);
        let mut l = layer(array![[2.0, -4.0]], array![3.0]);
        let grads = Gradients {
            dw: Array2::zeros((1, 2)),
            db: Array1::zeros(1),
        };
        opt.update(0, &mut l, &grads);

        // Each weight is scaled by (1 - lr*wd) = 0.99; the bias is not decayed.
        assert!((l.weights[[0, 0]] - 2.0 * 0.99).abs() < 1e-5);
        assert!((l.weights[[0, 1]] - (-4.0) * 0.99).abs() < 1e-5);
        assert!((l.biases[0] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn to_state_restore_roundtrip_preserves_moments_and_time_step() {
        let mut opt = Adam::with_defaults(0.01.try_into().unwrap(), WeightDecay::ZERO);
        let mut l = layer(array![[1.0, 1.0]], array![1.0]);
        let grads = Gradients {
            dw: array![[2.0, -3.0]],
            db: array![0.5],
        };
        opt.update(0, &mut l, &grads);
        opt.step();

        let state = opt.to_state().unwrap();

        let mut restored = Adam::with_defaults(0.01.try_into().unwrap(), WeightDecay::ZERO);
        restored.restore(&state).unwrap();

        // Applying the same update from the restored optimizer must reproduce
        // the same step as continuing with the original optimizer.
        let mut from_original = l.clone();
        let mut from_restored = l.clone();
        opt.update(0, &mut from_original, &grads);
        restored.update(0, &mut from_restored, &grads);

        assert_eq!(from_original.weights, from_restored.weights);
        assert_eq!(from_original.biases, from_restored.biases);
    }

    /// Metadata carrying a valid `time_step`, the precondition for parsing tensors.
    fn metadata_with_time_step(time_step: &str) -> HashMap<String, String> {
        HashMap::from([("time_step".to_string(), time_step.to_string())])
    }

    #[test]
    fn restore_rejects_missing_time_step() {
        let mut opt = Adam::with_defaults(0.01.try_into().unwrap(), WeightDecay::ZERO);
        let state = OptimizerState {
            tensors: Vec::new(),
            metadata: HashMap::new(),
        };

        assert_eq!(
            opt.restore(&state).unwrap_err().to_string(),
            "optimizer state is missing `time_step`"
        );
    }

    #[test]
    fn restore_rejects_unparseable_time_step() {
        let mut opt = Adam::with_defaults(0.01.try_into().unwrap(), WeightDecay::ZERO);
        let state = OptimizerState {
            tensors: Vec::new(),
            metadata: metadata_with_time_step("not-a-number"),
        };

        assert_eq!(
            opt.restore(&state).unwrap_err().to_string(),
            "optimizer state has an invalid `time_step`"
        );
    }

    #[test]
    fn restore_skips_unrecognized_tensor_names() {
        let mut opt = Adam::with_defaults(0.01.try_into().unwrap(), WeightDecay::ZERO);
        let state = OptimizerState {
            tensors: vec![
                // A complete, valid layer-0 state...
                (
                    "layer0.m_weights".to_string(),
                    Array2::<f32>::zeros((1, 1)).into_dyn(),
                ),
                (
                    "layer0.v_weights".to_string(),
                    Array2::<f32>::zeros((1, 1)).into_dyn(),
                ),
                (
                    "layer0.m_biases".to_string(),
                    Array1::<f32>::zeros(1).into_dyn(),
                ),
                (
                    "layer0.v_biases".to_string(),
                    Array1::<f32>::zeros(1).into_dyn(),
                ),
                // ...alongside entries that are each skipped: no `.` separator, an
                // unparseable layer index, and an unknown field.
                ("nodot".to_string(), Array1::<f32>::zeros(1).into_dyn()),
                (
                    "layerX.m_weights".to_string(),
                    Array2::<f32>::zeros((1, 1)).into_dyn(),
                ),
                (
                    "layer0.unknown".to_string(),
                    Array2::<f32>::zeros((1, 1)).into_dyn(),
                ),
            ],
            metadata: metadata_with_time_step("5"),
        };

        assert!(opt.restore(&state).is_ok());
    }

    /// A correctly-shaped tensor for an Adam moment field (rank 2 for weights,
    /// rank 1 for biases).
    fn tensor_for(field: &str) -> ndarray::ArrayD<f32> {
        if field.ends_with("weights") {
            Array2::<f32>::zeros((1, 1)).into_dyn()
        } else {
            Array1::<f32>::zeros(1).into_dyn()
        }
    }

    #[test]
    fn restore_reports_each_missing_layer_tensor() {
        const FIELDS: [&str; 4] = ["m_weights", "v_weights", "m_biases", "v_biases"];

        // Dropping any one field of an otherwise-complete layer is reported by name.
        for missing in FIELDS {
            let mut opt = Adam::with_defaults(0.01.try_into().unwrap(), WeightDecay::ZERO);
            let state = OptimizerState {
                tensors: FIELDS
                    .iter()
                    .filter(|f| **f != missing)
                    .map(|f| (format!("layer0.{f}"), tensor_for(f)))
                    .collect(),
                metadata: metadata_with_time_step("1"),
            };

            assert_eq!(
                opt.restore(&state).unwrap_err().to_string(),
                format!("optimizer state is missing `layer0.{missing}`")
            );
        }
    }

    #[test]
    fn restore_reports_wrong_rank_for_each_tensor() {
        // Each field is given the opposite rank to the one it expects.
        for (field, expected_rank) in [
            ("m_weights", 2),
            ("v_weights", 2),
            ("m_biases", 1),
            ("v_biases", 1),
        ] {
            let mut opt = Adam::with_defaults(0.01.try_into().unwrap(), WeightDecay::ZERO);
            let wrong = if expected_rank == 2 {
                Array1::<f32>::zeros(1).into_dyn()
            } else {
                Array2::<f32>::zeros((1, 1)).into_dyn()
            };
            let state = OptimizerState {
                tensors: vec![(format!("layer0.{field}"), wrong)],
                metadata: metadata_with_time_step("1"),
            };

            assert_eq!(
                opt.restore(&state).unwrap_err().to_string(),
                format!("tensor `layer0.{field}` is not rank {expected_rank}")
            );
        }
    }
}
