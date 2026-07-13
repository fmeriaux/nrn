//! Layer and network configuration: the declarative description of a network's architecture.
//! [`NetworkConfig`] bundles the per-sample input shape with an ordered stack of
//! [`LayerConfig`]s, one per layer kind.

use crate::activations::{Activation, IDENTITY};
use crate::layers::{Conv2d, Dense, Flatten, Layer};
use crate::tensors::{TensorError, Tensors};
use ndarray::{Ix2, Ix4};
use ndarray_rand::rand::RngCore;
use std::fmt;
use std::sync::Arc;

/// Represents the specifications for a neuron layer in a neural network.
#[derive(Debug)]
pub struct NeuronLayerSpec {
    /// The number of neurons in this layer.
    pub neurons: usize,
    /// The activation function used in this layer.
    pub activation: Arc<dyn Activation>,
}

/// How the hidden-layer architecture of a network is chosen.
#[derive(Debug, Clone)]
pub enum LayerPlan {
    /// Explicit hidden-layer neuron counts (empty = single-layer perceptron).
    Explicit(Vec<usize>),
    /// Infer the hidden layers from the dataset shape.
    Auto {
        /// Number of input features.
        n_features: usize,
        /// Number of training samples.
        n_samples: usize,
    },
}

/// Error returned when an explicit [`LayerPlan`] is invalid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LayerPlanError {
    /// A hidden layer was given zero neurons.
    ZeroNeuronLayer,
}

impl fmt::Display for LayerPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LayerPlanError::ZeroNeuronLayer => {
                write!(f, "each hidden layer must have at least one neuron")
            }
        }
    }
}

impl std::error::Error for LayerPlanError {}

impl NeuronLayerSpec {
    /// Creates specifications for multiple hidden layers with the same activation function.
    /// # Arguments
    /// - `neurons`: An iterator over the number of neurons for each hidden layer.
    /// - `activation`: The activation function to be used for all hidden layers.
    /// # Returns
    /// A vector of `NeuronLayerSpec` instances, one for each hidden layer.
    pub(crate) fn hidden<A: Activation + 'static>(
        neurons: impl IntoIterator<Item = usize>,
        activation: &Arc<A>,
    ) -> Vec<Self> {
        neurons
            .into_iter()
            .map(|n| NeuronLayerSpec {
                neurons: n,
                activation: activation.clone(),
            })
            .collect()
    }

    /// Creates an output layer specification based on the number of classes.
    ///
    /// The output layer is **linear** (it emits logits): softmax/sigmoid is folded into the
    /// cross-entropy loss during training and reapplied at inference by
    /// [`ClassifierActivations`](crate::classification::ClassifierActivations). The width encodes
    /// the task — 1 logit for binary, `n_classes` logits for multi-class.
    ///
    /// # Panics
    /// When `n_classes` is less than or equal to 1. This is a precondition the caller
    /// must guarantee; for data-derived class counts, validate the dataset up front with
    /// [`crate::data::Dataset::validate`], which reports the error instead of panicking.
    /// # Arguments
    /// - `n_classes`: The number of classes for the output layer.
    ///
    pub fn output_for(n_classes: usize) -> Self {
        assert!(
            n_classes > 1,
            "Number of classes must be greater than 1, got {}",
            n_classes
        );
        let neurons = if n_classes == 2 { 1 } else { n_classes };
        NeuronLayerSpec {
            neurons,
            activation: IDENTITY.clone(),
        }
    }

    /// Creates a full network specification including hidden layers and an output layer.
    /// # Arguments
    /// - `hidden_neurons`: An iterator over the number of neurons for each hidden layer.
    /// - `hidden_activation`: The activation function to be used for all hidden layers.
    /// - `n_classes`: The number of classes for the output layer.
    /// # Returns
    /// A vector of `NeuronLayerSpec` instances, including hidden layers and the output layer.
    pub(crate) fn network_for<A: Activation + 'static>(
        hidden_neurons: impl IntoIterator<Item = usize>,
        hidden_activation: &Arc<A>,
        n_classes: usize,
    ) -> Vec<Self> {
        let mut specs = Self::hidden(hidden_neurons, hidden_activation);
        specs.push(Self::output_for(n_classes));
        specs
    }

    /// Resolves a [`LayerPlan`] into a full network specification.
    ///
    /// The single public entry point for turning an architecture choice into layer
    /// specs: it dispatches to the internal builders and is the only place that
    /// validates the plan, rejecting an explicit layer with zero neurons.
    pub fn plan<A: Activation + 'static>(
        plan: LayerPlan,
        n_classes: usize,
        hidden_activation: &Arc<A>,
    ) -> Result<Vec<Self>, LayerPlanError> {
        match plan {
            LayerPlan::Auto {
                n_features,
                n_samples,
            } => Ok(Self::infer_from(
                n_features,
                n_classes,
                n_samples,
                hidden_activation,
            )),
            LayerPlan::Explicit(layers) => {
                if layers.contains(&0) {
                    return Err(LayerPlanError::ZeroNeuronLayer);
                }
                Ok(Self::network_for(layers, hidden_activation, n_classes))
            }
        }
    }

    /// Infers a suitable network architecture based on dataset characteristics.
    /// The architecture is determined by a complexity score derived from the number of features,
    /// classes, and samples in the dataset.
    /// # Panics
    /// - When `n_features` is less than or equal to zero.
    /// - When `n_classes` is less than or equal to one.
    /// - When `n_samples` is less than or equal to zero.
    ///
    /// These are preconditions the caller must guarantee. For data-derived values,
    /// validate the dataset up front with [`crate::data::Dataset::validate`], which
    /// reports these conditions as errors instead of panicking.
    pub(crate) fn infer_from<A: Activation + 'static>(
        n_features: usize,
        n_classes: usize,
        n_samples: usize,
        hidden_activation: &Arc<A>,
    ) -> Vec<Self> {
        assert!(
            n_features > 0,
            "Number of features must be greater than zero."
        );
        assert!(n_classes > 1, "Number of classes must be greater than one.");
        assert!(
            n_samples > 0,
            "Number of samples must be greater than zero."
        );
        // Complexity score combines features, classes, and samples to guide architecture decisions
        let complexity_score = ((n_features as f64) * (n_classes as f64) / (n_samples as f64)).ln();

        // Determine the number of hidden layers and neurons based on the complexity score
        let (n_layers, n_neurons) = match complexity_score {
            f64::NEG_INFINITY..=-3.0 => (1, n_features * 2),
            -3.0..=-1.0 => (2, n_features),
            -1.0..=0.0 => (3, n_features * 2),
            _ => (3, n_features * 3),
        };

        let mut hidden_layers = Vec::with_capacity(n_layers);
        let mut current_neurons = n_neurons.clamp(16, 512);

        for _layer in 0..n_layers {
            hidden_layers.push(current_neurons);
            current_neurons = (current_neurons / 2).max(n_classes * 2);
        }

        Self::network_for(hidden_layers, hidden_activation, n_classes)
    }
}

/// A network's architecture with no weights: the per-sample input shape and the ordered stack
/// of layer configs — the declarative counterpart to a [`NeuralNetwork`](crate::model::NeuralNetwork).
#[derive(Debug, Clone)]
pub struct NetworkConfig {
    /// The per-sample input shape, sample axis excluded.
    pub input_shape: Vec<usize>,
    /// The layers in order, from input to output.
    pub layers: Vec<LayerConfig>,
}

/// A typed, weight-free description of one layer: its kind and the hyperparameters needed to
/// build it, the concrete input shape excepted (that is threaded from the network's input).
///
/// It spans every [`Layer`] kind and is the declarative counterpart to a layer's weights:
/// combined with weights — freshly [initialized](LayerConfig::initialization) or supplied as
/// [named tensors](LayerConfig::from_tensors) — a config yields a live layer.
#[derive(Clone, Debug)]
pub enum LayerConfig {
    /// A fully connected [`Dense`] layer.
    Dense {
        /// The number of neurons, i.e. output features.
        neurons: usize,
        /// The activation applied to the layer's output.
        activation: Arc<dyn Activation>,
    },
    /// A 2D convolution [`Conv2d`] layer.
    Conv2d {
        /// The number of filters, i.e. output channels.
        out_channels: usize,
        /// The kernel's spatial size `(height, width)`.
        kernel: (usize, usize),
        /// The stride applied on both spatial axes.
        stride: usize,
        /// The zero-padding added on both spatial axes.
        padding: usize,
        /// The activation applied to the layer's output.
        activation: Arc<dyn Activation>,
    },
    /// A reshaping [`Flatten`] layer.
    Flatten,
}

impl LayerConfig {
    /// Builds a fresh layer at `input_shape`, drawing its weights from `rng`.
    pub fn initialization(
        &self,
        input_shape: &[usize],
        rng: &mut dyn RngCore,
    ) -> Result<Box<dyn Layer>, LayerConfigError> {
        Ok(match self {
            LayerConfig::Dense {
                neurons,
                activation,
            } => {
                let inputs = flat_input(input_shape)?;
                let (weights, biases) = activation.initialization().apply((*neurons, inputs), rng);
                Box::new(Dense::new(weights, biases, activation.clone()))
            }
            LayerConfig::Conv2d {
                out_channels,
                kernel,
                stride,
                padding,
                activation,
            } => {
                let input = conv_input(input_shape, *kernel, *padding)?;
                Box::new(Conv2d::initialization(
                    input,
                    *out_channels,
                    *kernel,
                    *stride,
                    *padding,
                    activation.clone(),
                    rng,
                ))
            }
            LayerConfig::Flatten => Box::new(Flatten::new(input_shape.to_vec())),
        })
    }

    /// Builds a layer at `input_shape`, taking its weights from the given named `tensors`.
    pub fn from_tensors(
        &self,
        input_shape: &[usize],
        mut tensors: Tensors,
    ) -> Result<Box<dyn Layer>, LayerConfigError> {
        Ok(match self {
            LayerConfig::Dense { activation, .. } => {
                flat_input(input_shape)?;
                let weights = tensors
                    .take_weight::<Ix2>()
                    .map_err(LayerConfigError::Tensor)?;
                let biases = tensors.take_bias().map_err(LayerConfigError::Tensor)?;
                Box::new(Dense::new(weights, biases, activation.clone()))
            }
            LayerConfig::Conv2d {
                kernel,
                stride,
                padding,
                activation,
                ..
            } => {
                let input = conv_input(input_shape, *kernel, *padding)?;
                let kernels = tensors
                    .take_weight::<Ix4>()
                    .map_err(LayerConfigError::Tensor)?;
                let biases = tensors.take_bias().map_err(LayerConfigError::Tensor)?;
                Box::new(Conv2d::new(
                    kernels,
                    biases,
                    input,
                    *stride,
                    *padding,
                    activation.clone(),
                ))
            }
            LayerConfig::Flatten => Box::new(Flatten::new(input_shape.to_vec())),
        })
    }
}

/// Validates that a dense layer's input is flat (rank-1), returning its feature count.
fn flat_input(input_shape: &[usize]) -> Result<usize, LayerConfigError> {
    match *input_shape {
        [inputs] => Ok(inputs),
        _ => Err(LayerConfigError::UnexpectedInputRank {
            expected: 1,
            got: input_shape.to_vec(),
        }),
    }
}

/// Validates that a convolution's input is rank-3 and admits the kernel, returning its
/// `(channels, height, width)`.
fn conv_input(
    input_shape: &[usize],
    kernel: (usize, usize),
    padding: usize,
) -> Result<(usize, usize, usize), LayerConfigError> {
    let &[channels, height, width] = input_shape else {
        return Err(LayerConfigError::UnexpectedInputRank {
            expected: 3,
            got: input_shape.to_vec(),
        });
    };
    let (kh, kw) = kernel;
    if height + 2 * padding < kh || width + 2 * padding < kw {
        return Err(LayerConfigError::WindowDoesNotFit {
            input: (height, width),
            window: kernel,
            padding,
        });
    }
    Ok((channels, height, width))
}

/// Error returned when a [`LayerConfig`] cannot be resolved into a layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayerConfigError {
    /// The layer was given an input of the wrong rank — a dense head needs a flat (rank-1)
    /// input, a convolution a rank-3 `(channels, height, width)` one; insert a [`Flatten`]
    /// where the rank must drop.
    UnexpectedInputRank {
        /// The input rank the layer requires.
        expected: usize,
        /// The per-sample input shape the layer received.
        got: Vec<usize>,
    },
    /// The sliding window (a convolution kernel, a pooling extent), with padding, does not fit
    /// the input's spatial extent — no valid output position exists.
    WindowDoesNotFit {
        /// The input's spatial extent `(height, width)`.
        input: (usize, usize),
        /// The window's spatial size `(height, width)`.
        window: (usize, usize),
        /// The zero-padding added on both spatial axes.
        padding: usize,
    },
    /// A tensor required to build a layer's weights was absent or had the wrong rank.
    Tensor(TensorError),
}

impl fmt::Display for LayerConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LayerConfigError::UnexpectedInputRank { expected, got } => write!(
                f,
                "layer expects a rank-{expected} input, got shape {got:?}"
            ),
            LayerConfigError::WindowDoesNotFit {
                input,
                window,
                padding,
            } => write!(
                f,
                "window {window:?} with padding {padding} does not fit input {input:?}"
            ),
            LayerConfigError::Tensor(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for LayerConfigError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::RELU;

    // infer_from branches — complexity score = ln(n_features * n_classes / n_samples)
    // Branch thresholds: <= -3.0 | -3.0..-1.0 | -1.0..0.0 | > 0.0
    // Each branch selects a different number of hidden layers (1, 2, 3, 3).

    #[test]
    fn infer_from_low_complexity_gives_one_hidden_layer() {
        // ln(2 * 2 / 1000) ≈ -5.5 → 1 hidden layer
        let specs = NeuronLayerSpec::infer_from(2, 2, 1000, &RELU);
        assert_eq!(specs.len(), 2, "expected 1 hidden + 1 output spec");
    }

    #[test]
    fn infer_from_medium_complexity_gives_two_hidden_layers() {
        // ln(2 * 2 / 20) ≈ -1.6 → 2 hidden layers
        let specs = NeuronLayerSpec::infer_from(2, 2, 20, &RELU);
        assert_eq!(specs.len(), 3, "expected 2 hidden + 1 output specs");
    }

    #[test]
    fn infer_from_moderate_complexity_gives_three_hidden_layers() {
        // ln(2 * 3 / 8) ≈ -0.29 → 3 hidden layers
        let specs = NeuronLayerSpec::infer_from(2, 3, 8, &RELU);
        assert_eq!(specs.len(), 4, "expected 3 hidden + 1 output specs");
    }

    #[test]
    fn infer_from_high_complexity_gives_three_hidden_layers() {
        // ln(5 * 5 / 10) ≈ 0.92 → 3 hidden layers (widest variant)
        let specs = NeuronLayerSpec::infer_from(5, 5, 10, &RELU);
        assert_eq!(specs.len(), 4, "expected 3 hidden + 1 output specs");
        // First hidden layer uses n_features * 3 neurons (clamped to [16, 512])
        assert_eq!(
            specs[0].neurons, 16,
            "first layer should use at least 16 neurons"
        );
    }

    #[test]
    fn plan_explicit_matches_network_for() {
        // Explicit plan uses the given layers verbatim.
        let specs = NeuronLayerSpec::plan(LayerPlan::Explicit(vec![8, 4]), 3, &RELU).unwrap();
        assert_eq!(specs.len(), 3, "expected 2 hidden + 1 output specs");
        assert_eq!(specs[0].neurons, 8);
        assert_eq!(specs[1].neurons, 4);
        assert_eq!(specs[2].neurons, 3, "output layer matches n_classes");
    }

    #[test]
    fn plan_explicit_rejects_zero_neuron_layer() {
        let err = NeuronLayerSpec::plan(LayerPlan::Explicit(vec![8, 0, 4]), 3, &RELU).unwrap_err();
        assert_eq!(err, LayerPlanError::ZeroNeuronLayer);
        assert_eq!(
            err.to_string(),
            "each hidden layer must have at least one neuron"
        );
    }

    #[test]
    fn plan_explicit_accepts_empty_layers() {
        // Empty = single-layer perceptron: just the output layer, no hidden layers.
        let specs = NeuronLayerSpec::plan(LayerPlan::Explicit(vec![]), 2, &RELU).unwrap();
        assert_eq!(specs.len(), 1, "expected only the output spec");
    }

    #[test]
    fn plan_auto_matches_infer_from() {
        // Auto plan defers to infer_from: ln(2 * 2 / 1000) ≈ -5.5 → 1 hidden layer.
        let specs = NeuronLayerSpec::plan(
            LayerPlan::Auto {
                n_features: 2,
                n_samples: 1000,
            },
            2,
            &RELU,
        )
        .unwrap();
        assert_eq!(specs.len(), 2, "expected 1 hidden + 1 output spec");
    }

    use ndarray_rand::rand::SeedableRng;
    use ndarray_rand::rand::rngs::StdRng;

    #[test]
    fn dense_spec_initializes_a_layer_at_the_flat_input() {
        let spec = LayerConfig::Dense {
            neurons: 4,
            activation: RELU.clone(),
        };
        let mut rng = StdRng::seed_from_u64(0);
        let layer = spec.initialization(&[3], &mut rng).unwrap();

        assert_eq!(layer.input_shape(), vec![3]);
        assert_eq!(layer.output_shape(), vec![4]);
        assert_eq!(layer.activation_name(), Some("relu"));
    }

    #[test]
    fn conv_spec_initializes_a_layer_at_the_spatial_input() {
        let spec = LayerConfig::Conv2d {
            out_channels: 2,
            kernel: (3, 3),
            stride: 1,
            padding: 0,
            activation: RELU.clone(),
        };
        let mut rng = StdRng::seed_from_u64(0);
        let layer = spec.initialization(&[1, 4, 4], &mut rng).unwrap();

        assert_eq!(layer.input_shape(), vec![1, 4, 4]);
        // conv_output_dim(4, 3, 1, 0) = 2 on each spatial axis.
        assert_eq!(layer.output_shape(), vec![2, 2, 2]);
    }

    #[test]
    fn dense_spec_rejects_a_non_flat_input() {
        let spec = LayerConfig::Dense {
            neurons: 4,
            activation: RELU.clone(),
        };
        let mut rng = StdRng::seed_from_u64(0);
        let err = spec.initialization(&[1, 4, 4], &mut rng).unwrap_err();

        assert_eq!(
            err,
            LayerConfigError::UnexpectedInputRank {
                expected: 1,
                got: vec![1, 4, 4],
            }
        );
    }

    #[test]
    fn conv_spec_rejects_a_non_spatial_input() {
        let spec = LayerConfig::Conv2d {
            out_channels: 2,
            kernel: (3, 3),
            stride: 1,
            padding: 0,
            activation: RELU.clone(),
        };
        let mut rng = StdRng::seed_from_u64(0);
        let err = spec.initialization(&[16], &mut rng).unwrap_err();

        assert_eq!(
            err,
            LayerConfigError::UnexpectedInputRank {
                expected: 3,
                got: vec![16],
            }
        );
    }

    #[test]
    fn conv_spec_rejects_a_window_larger_than_the_input() {
        // A 3×3 kernel with no padding cannot fit a 2×2 spatial input.
        let spec = LayerConfig::Conv2d {
            out_channels: 2,
            kernel: (3, 3),
            stride: 1,
            padding: 0,
            activation: RELU.clone(),
        };
        let mut rng = StdRng::seed_from_u64(0);
        let err = spec.initialization(&[1, 2, 2], &mut rng).unwrap_err();

        assert_eq!(
            err,
            LayerConfigError::WindowDoesNotFit {
                input: (2, 2),
                window: (3, 3),
                padding: 0,
            }
        );
    }
}
