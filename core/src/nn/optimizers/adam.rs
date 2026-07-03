use crate::learning_rate::LearningRate;
use crate::optimizers::{Optimizer, OptimizerState, OptimizerStateError, ParameterUpdates};
use crate::weight_decay::WeightDecay;
use ndarray::{ArrayD, Zip};
use std::collections::{BTreeMap, HashMap};

/// Adam moment estimates for a single parameter: the first moment (average of past
/// gradients) and the second moment (average of their squares).
struct AdamMoments {
    /// First moment estimate, averaging past gradients.
    m: ArrayD<f32>,
    /// Second moment estimate, the variance of past gradients.
    v: ArrayD<f32>,
}

/// Which moment estimate a state tensor name refers to.
enum MomentField {
    M,
    V,
}

/// Provides the Adam optimization algorithm for updating neural network parameters.
pub struct Adam {
    /// Learning rate for the optimizer.
    learning_rate: LearningRate,
    /// Exponential decay rate for the first moment estimates.
    beta1: f32,
    /// Exponential decay rate for the second moment estimates.
    beta2: f32,
    /// Small constant for numerical stability.
    epsilon: f32,
    /// Decoupled weight-decay coefficient (AdamW). Applied to decaying parameters only.
    weight_decay: WeightDecay,
    /// Step counter for bias correction.
    time_step: u32,
    /// Internal state for each layer: one [`AdamMoments`] per parameter, in parameter order.
    states: HashMap<usize, Vec<AdamMoments>>,
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

    fn update(&mut self, layer_index: usize, updates: &mut ParameterUpdates<'_>) {
        let lr = self.learning_rate.value();
        let (beta1, beta2, epsilon, weight_decay) =
            (self.beta1, self.beta2, self.epsilon, self.weight_decay);
        let beta1_correction = 1.0 - beta1.powi(self.time_step as i32);
        let beta2_correction = 1.0 - beta2.powi(self.time_step as i32);

        let moments = self.states.entry(layer_index).or_insert_with(|| {
            updates
                .iter()
                .map(|update| AdamMoments {
                    m: ArrayD::zeros(update.parameter.value.raw_dim()),
                    v: ArrayD::zeros(update.parameter.value.raw_dim()),
                })
                .collect()
        });

        // A restored state must describe exactly this layer's parameters; a shorter or
        // longer moment list means the checkpoint does not belong to this model, which
        // would otherwise leave some parameters silently un-updated.
        assert_eq!(
            moments.len(),
            updates.len(),
            "Adam state for layer {layer_index} has {} parameters, but the layer has {}",
            moments.len(),
            updates.len(),
        );

        for (update, moment) in updates.iter_mut().zip(moments.iter_mut()) {
            assert_eq!(
                moment.m.shape(),
                update.parameter.value.shape(),
                "Adam moment shape does not match parameter shape for layer {layer_index}",
            );
            // Decoupled weight decay (AdamW): shrink decaying parameters before the
            // gradient step. A factor of `1.0` leaves non-decaying parameters untouched.
            let decay = if weight_decay.is_active() && update.parameter.decays {
                1.0 - lr * weight_decay.value()
            } else {
                1.0
            };

            // Fuse the moment updates, bias correction, decay, and parameter step into a
            // single in-place pass. On tiny parameters this pays the array traversal once
            // instead of allocating a temporary per sub-expression.
            Zip::from(update.parameter.value.view_mut())
                .and(&mut moment.m)
                .and(&mut moment.v)
                .and(update.gradient)
                .for_each(|param, m, v, &g| {
                    // Biased first and second moment estimates.
                    *m = *m * beta1 + g * (1.0 - beta1);
                    *v = *v * beta2 + g * g * (1.0 - beta2);
                    // Bias-corrected estimates.
                    let m_hat = *m / beta1_correction;
                    let v_hat = *v / beta2_correction;
                    *param = *param * decay - m_hat * lr / (v_hat.sqrt() + epsilon);
                });
        }
    }

    fn step(&mut self) {
        self.time_step += 1;
    }

    fn to_state(&self) -> Option<OptimizerState> {
        let mut tensors = Vec::with_capacity(self.states.len() * 2);
        for (layer_index, moments) in &self.states {
            for (param_index, moment) in moments.iter().enumerate() {
                tensors.push((
                    format!("layer{layer_index}.param{param_index}.m"),
                    moment.m.clone(),
                ));
                tensors.push((
                    format!("layer{layer_index}.param{param_index}.v"),
                    moment.v.clone(),
                ));
            }
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

        // Group tensors by layer, then by parameter index (kept sorted for a stable
        // parameter order) before assembling the per-parameter moments.
        let mut partials: HashMap<usize, BTreeMap<usize, PartialMoments>> = HashMap::new();

        for (name, array) in &state.tensors {
            let Some((layer_index, param_index, field)) = parse_moment_name(name) else {
                continue;
            };
            let partial = partials
                .entry(layer_index)
                .or_default()
                .entry(param_index)
                .or_default();
            match field {
                MomentField::M => partial.m = Some(array.clone()),
                MomentField::V => partial.v = Some(array.clone()),
            }
        }

        for (layer_index, params) in partials {
            let mut moments = Vec::with_capacity(params.len());
            for (param_index, partial) in params {
                moments.push(partial.into_moments(layer_index, param_index)?);
            }
            self.states.insert(layer_index, moments);
        }

        Ok(())
    }
}

/// Parses a moment tensor name of the form `layer{L}.param{J}.{m|v}` into its layer
/// index, parameter index, and field. Returns `None` for any other shape.
fn parse_moment_name(name: &str) -> Option<(usize, usize, MomentField)> {
    let mut parts = name.split('.');
    let layer_index = parts.next()?.strip_prefix("layer")?.parse().ok()?;
    let param_index = parts.next()?.strip_prefix("param")?.parse().ok()?;
    let field = match parts.next()? {
        "m" => MomentField::M,
        "v" => MomentField::V,
        _ => return None,
    };
    if parts.next().is_some() {
        return None;
    }
    Some((layer_index, param_index, field))
}

/// Per-parameter Adam moment tensors collected from an [`OptimizerState`] while loading,
/// before being assembled into an [`AdamMoments`].
#[derive(Default)]
struct PartialMoments {
    m: Option<ArrayD<f32>>,
    v: Option<ArrayD<f32>>,
}

impl PartialMoments {
    fn into_moments(
        self,
        layer_index: usize,
        param_index: usize,
    ) -> Result<AdamMoments, OptimizerStateError> {
        let missing = |field: &str| {
            OptimizerStateError::MissingTensor(format!(
                "layer{layer_index}.param{param_index}.{field}"
            ))
        };

        Ok(AdamMoments {
            m: self.m.ok_or_else(|| missing("m"))?,
            v: self.v.ok_or_else(|| missing("v"))?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::RELU;
    use crate::gradients::LayerGradients;
    use crate::layers::Dense;
    use ndarray::{Array1, Array2, array};

    fn layer(weights: Array2<f32>, biases: Array1<f32>) -> Dense {
        Dense::new(weights, biases, RELU.clone())
    }

    /// Layer gradients for a dense layer: a rank-2 weight gradient and a rank-1 bias gradient.
    fn grads(dw: Array2<f32>, db: Array1<f32>) -> LayerGradients {
        LayerGradients(vec![dw.into_dyn(), db.into_dyn()])
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
        opt.update_layer(0, &mut l, &grads(array![[2.0, -3.0]], array![0.5]));
        assert!((l.weights()[[0, 0]] - (1.0 - lr)).abs() < 1e-4); // +grad → decrease
        assert!((l.weights()[[0, 1]] - (1.0 + lr)).abs() < 1e-4); // -grad → increase
        assert!((l.biases()[0] - (1.0 - lr)).abs() < 1e-4);
    }

    #[test]
    fn adam_zero_gradient_leaves_params_unchanged() {
        let mut opt = Adam::with_defaults(0.1.try_into().unwrap(), WeightDecay::ZERO);
        let mut l = layer(array![[1.0, -2.0]], array![3.0]);
        opt.update_layer(0, &mut l, &grads(Array2::zeros((1, 2)), Array1::zeros(1)));
        assert!((l.weights()[[0, 0]] - 1.0).abs() < 1e-6);
        assert!((l.weights()[[0, 1]] - (-2.0)).abs() < 1e-6);
        assert!((l.biases()[0] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn adam_minimizes_simple_quadratic() {
        // f(w) = 0.5 * w^2 has gradient w and minimum at 0; Adam should drive w → 0.
        let mut opt = Adam::with_defaults(0.1.try_into().unwrap(), WeightDecay::ZERO);
        let mut l = layer(array![[5.0]], array![0.0]);
        for _ in 0..200 {
            let w = l.weights()[[0, 0]];
            opt.update_layer(0, &mut l, &grads(array![[w]], array![0.0]));
            opt.step();
        }
        let w = l.weights()[[0, 0]];
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
        opt.update_layer(0, &mut l, &grads(Array2::zeros((1, 2)), Array1::zeros(1)));

        // Each weight is scaled by (1 - lr*wd) = 0.99; the bias is not decayed.
        assert!((l.weights()[[0, 0]] - 2.0 * 0.99).abs() < 1e-5);
        assert!((l.weights()[[0, 1]] - (-4.0) * 0.99).abs() < 1e-5);
        assert!((l.biases()[0] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn to_state_restore_roundtrip_preserves_moments_and_time_step() {
        let mut opt = Adam::with_defaults(0.01.try_into().unwrap(), WeightDecay::ZERO);
        let mut l = layer(array![[1.0, 1.0]], array![1.0]);
        let gradients = grads(array![[2.0, -3.0]], array![0.5]);
        opt.update_layer(0, &mut l, &gradients);
        opt.step();

        let state = opt.to_state().unwrap();

        let mut restored = Adam::with_defaults(0.01.try_into().unwrap(), WeightDecay::ZERO);
        restored.restore(&state).unwrap();

        // Applying the same update from the restored optimizer must reproduce
        // the same step as continuing with the original optimizer.
        let mut from_original = l.clone();
        let mut from_restored = l.clone();
        opt.update_layer(0, &mut from_original, &gradients);
        restored.update_layer(0, &mut from_restored, &gradients);

        assert_eq!(from_original.weights(), from_restored.weights());
        assert_eq!(from_original.biases(), from_restored.biases());
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
                // A complete, valid layer-0 state (two parameters)...
                (
                    "layer0.param0.m".to_string(),
                    Array2::<f32>::zeros((1, 1)).into_dyn(),
                ),
                (
                    "layer0.param0.v".to_string(),
                    Array2::<f32>::zeros((1, 1)).into_dyn(),
                ),
                (
                    "layer0.param1.m".to_string(),
                    Array1::<f32>::zeros(1).into_dyn(),
                ),
                (
                    "layer0.param1.v".to_string(),
                    Array1::<f32>::zeros(1).into_dyn(),
                ),
                // ...alongside entries that are each skipped: no `.` separator, an
                // unparseable layer index, and an unknown field.
                ("nodot".to_string(), Array1::<f32>::zeros(1).into_dyn()),
                (
                    "layerX.param0.m".to_string(),
                    Array2::<f32>::zeros((1, 1)).into_dyn(),
                ),
                (
                    "layer0.param0.unknown".to_string(),
                    Array2::<f32>::zeros((1, 1)).into_dyn(),
                ),
            ],
            metadata: metadata_with_time_step("5"),
        };

        assert!(opt.restore(&state).is_ok());
    }

    #[test]
    fn restore_reports_each_missing_parameter_tensor() {
        const FIELDS: [&str; 2] = ["m", "v"];

        // Dropping either moment of an otherwise-complete parameter is reported by name.
        for missing in FIELDS {
            let mut opt = Adam::with_defaults(0.01.try_into().unwrap(), WeightDecay::ZERO);
            let state = OptimizerState {
                tensors: FIELDS
                    .iter()
                    .filter(|f| **f != missing)
                    .map(|f| {
                        (
                            format!("layer0.param0.{f}"),
                            Array2::<f32>::zeros((1, 1)).into_dyn(),
                        )
                    })
                    .collect(),
                metadata: metadata_with_time_step("1"),
            };

            assert_eq!(
                opt.restore(&state).unwrap_err().to_string(),
                format!("optimizer state is missing `layer0.param0.{missing}`")
            );
        }
    }

    #[test]
    #[should_panic(expected = "has 1 parameters, but the layer has 2")]
    fn update_rejects_restored_state_that_omits_a_parameter() {
        // A checkpoint describing only the first parameter of layer 0 (the bias is absent).
        let state = OptimizerState {
            tensors: vec![
                (
                    "layer0.param0.m".to_string(),
                    Array2::<f32>::zeros((1, 2)).into_dyn(),
                ),
                (
                    "layer0.param0.v".to_string(),
                    Array2::<f32>::zeros((1, 2)).into_dyn(),
                ),
            ],
            metadata: metadata_with_time_step("5"),
        };

        let mut opt = Adam::with_defaults(0.01.try_into().unwrap(), WeightDecay::ZERO);
        opt.restore(&state).unwrap();

        // The dense layer has two parameters (weights and bias); the restored state has one,
        // so stepping it must fail loudly instead of silently skipping the bias.
        let mut l = layer(array![[1.0, 1.0]], array![1.0]);
        opt.update_layer(0, &mut l, &grads(array![[0.1, 0.1]], array![0.1]));
    }
}
