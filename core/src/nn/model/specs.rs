//! Layer specifications: the declarative description of a network's architecture, resolved into
//! layers by [`NeuralNetwork::initialization`](crate::model::NeuralNetwork::initialization).

use crate::activations::{Activation, IDENTITY};
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
    /// The output layer is **linear** (it emits logits): softmax/sigmoid is not applied in the
    /// forward pass but folded into the cross-entropy loss during training and reapplied at
    /// inference by [`ClassifierActivations`](crate::classification::ClassifierActivations).
    /// The width still encodes the task — 1 logit for
    /// binary, `n_classes` logits for multi-class — so the inference activation stays inferable.
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
}
