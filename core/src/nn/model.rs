//! This module defines the core data structures for representing neural network architectures.
//!
//! It provides types for individual neuron layers ([`NeuronLayer`]) and entire networks ([`NeuralNetwork`]).
//! Each layer contains its weights, biases, and activation function, enabling flexible and modular
//! construction of multi-layer perceptron and similar models.

use crate::activations::{Activation, SIGMOID, SOFTMAX};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use ndarray_rand::rand::RngCore;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand::rngs::StdRng;
use std::iter::once;
use std::sync::Arc;

/// Represents a single layer in a neural network, containing weights and biases.
#[derive(Clone, Debug)]
pub struct NeuronLayer {
    /// A 2D array where each row corresponds to a neuron and each column corresponds to an input feature.
    pub weights: Array2<f32>,
    /// A 1D array where each element is the bias for the corresponding neuron.
    pub biases: Array1<f32>,
    /// The activation function applied to the output of this layer.
    pub activation: Arc<dyn Activation>,
}

/// Represents a neural network composed of multiple layers of neurons.
#[derive(Clone, Debug)]
pub struct NeuralNetwork {
    /// A vector of [`NeuronLayer`] instances, defining the architecture of the network.
    pub layers: Vec<NeuronLayer>,
}

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
    Auto,
}

/// Returns the last activation from a vector of activations.
/// # Panics
/// - When the `activations` vector is empty.
pub fn last_activation(activations: &[Array2<f32>]) -> Array2<f32> {
    activations
        .last()
        .expect("Ensure activations is not empty.")
        .to_owned()
}

impl NeuronLayer {
    /// Initializes a new `NeuronLayer` with random weights and biases drawn from `rng`.
    /// # Panics
    /// - When `neurons` or `inputs` are less than or equal to zero.
    /// # Arguments
    /// - `inputs`: The number of inputs to this layer (i.e., the number of neurons in the previous layer).
    /// - `spec`: The specifications for this layer, including the number of neurons and the activation method.
    /// - `rng`: The random number generator the weights are drawn from. Passing a seeded
    ///   generator makes the initialization reproducible.
    pub fn initialization(inputs: usize, spec: &NeuronLayerSpec, rng: &mut dyn RngCore) -> Self {
        assert!(
            spec.neurons > 0 && inputs > 0,
            "Neurons and inputs must be greater than zero."
        );

        let (weights, biases) = spec
            .activation
            .initialization()
            .apply((spec.neurons, inputs), rng);

        NeuronLayer {
            weights,
            biases,
            activation: spec.activation.clone(),
        }
    }

    /// Computes the forward pass of this layer given the inputs.
    /// # Arguments
    /// - `inputs`: A 2D array representing the inputs to this layer.
    /// # Returns
    /// - A 2D array representing the outputs of this layer after applying the configured activation function.
    pub fn forward(&self, inputs: &Array2<f32>) -> Array2<f32> {
        assert_eq!(
            inputs.nrows(),
            self.weights.ncols(),
            "Input shape does not match weights shape."
        );

        // Broadcasting bias to match the shape of the output
        let broadcasted_biases: Array2<f32> = self.biases.view().insert_axis(Axis(1)).to_owned();

        self.activation
            .apply((self.weights.dot(inputs) + &broadcasted_biases).view())
    }

    /// Returns the number of neurons in this layer.
    pub fn size(&self) -> usize {
        self.weights.nrows()
    }

    /// Returns `true` when all weights and biases in this layer are finite (no NaN or Inf).
    pub fn is_finite(&self) -> bool {
        self.weights.iter().all(|v| v.is_finite()) && self.biases.iter().all(|v| v.is_finite())
    }

    /// Returns the number of inputs to this layer.
    /// For example, this is the number of neurons in the previous layer,
    /// or the input size for the first layer.
    pub fn input_size(&self) -> usize {
        self.weights.ncols()
    }

    /// Returns the specifications of this layer.
    pub fn spec(&self) -> NeuronLayerSpec {
        NeuronLayerSpec {
            neurons: self.size(),
            activation: self.activation.clone(),
        }
    }
}

impl NeuralNetwork {
    /// Creates a new `NeuralNetwork` with the specified input size and layer specifications.
    ///
    /// The weights are drawn from a generator seeded with `seed`, so initialization is
    /// always reproducible: the same `seed` and architecture yield the same starting
    /// weights. There is intentionally no unseeded constructor — an implicit random
    /// initialization is exactly the source of non-reproducible (flaky) training runs.
    /// # Arguments
    /// - `inputs`: The number of inputs to the first layer of the network.
    /// - `layer_specs`: A slice of `NeuronLayerSpec` representing the specifications for each layer in the network.
    /// - `seed`: The seed for the weight-initialization generator.
    pub fn initialization(inputs: usize, layer_specs: &[NeuronLayerSpec], seed: u64) -> Self {
        assert!(inputs > 0, "Input size must be greater than zero.");
        assert!(
            !layer_specs.is_empty(),
            "At least one layer must be specified."
        );

        // A single generator threaded through every layer, so the layers draw
        // decorrelated weights from one reproducible stream.
        let mut rng = StdRng::seed_from_u64(seed);
        let mut layers = Vec::with_capacity(layer_specs.len());
        let mut layer_input = inputs;

        for layer_spec in layer_specs {
            layers.push(NeuronLayer::initialization(
                layer_input,
                layer_spec,
                &mut rng,
            ));
            layer_input = layer_spec.neurons;
        }

        NeuralNetwork { layers }
    }

    pub fn specs(&self) -> Vec<NeuronLayerSpec> {
        self.layers.iter().map(|layer| layer.spec()).collect()
    }

    /// Returns the input size of the network, which is the number of inputs to the first layer.
    pub fn input_size(&self) -> usize {
        self.layers[0].input_size()
    }

    /// Number of classes discriminated, inferred from the output layer:
    /// a single sigmoid output is binary (2 classes); k softmax outputs are k classes.
    pub fn n_classes(&self) -> usize {
        let out = self
            .layers
            .last()
            .expect("network has at least one layer")
            .size();
        if out == 1 { 2 } else { out }
    }

    /// Returns a summary of the network's architecture as a string,
    /// showing the number of neurons in each layer, including the input layer.
    pub fn summary(&self) -> String {
        once(self.input_size())
            .map(|size| format!("[{}]", size))
            .chain(
                self.specs()
                    .iter()
                    .map(|spec| format!("{}-{}", spec.neurons, spec.activation.name())),
            )
            .collect::<Vec<String>>()
            .join(" -> ")
    }

    /// Computes the forward pass through the network, returning the activations of each layer.
    /// # Panics
    /// - When the number of rows in `inputs` does not match the number of columns in the weights of the first layer.
    /// # Arguments
    /// - `inputs`: A 2D array representing the inputs to the network.
    pub fn forward(&self, inputs: ArrayView2<f32>) -> Vec<Array2<f32>> {
        assert_eq!(
            inputs.nrows(),
            self.layers[0].weights.ncols(),
            "Input shape does not match the first layer's weights shape."
        );

        self.layers
            .iter()
            .fold(vec![inputs.to_owned()], |mut acc, layer| {
                acc.push(layer.forward(acc.last().unwrap()));
                acc
            })
    }

    /// Returns `true` when all layers are finite (no NaN or Inf in any weight or bias).
    pub fn is_finite(&self) -> bool {
        self.layers.iter().all(|layer| layer.is_finite())
    }

    /// Predicts the output of the network given the inputs, returning the final activations.
    /// # Arguments
    /// - `inputs`: A 2D array representing the inputs to the network.
    pub fn predict(&self, inputs: ArrayView2<f32>) -> Array2<f32> {
        let output = last_activation(&self.forward(inputs));
        assert!(
            output.iter().all(|v| v.is_finite()),
            "non-finite predictions (NaN or inf): the model likely diverged during training"
        );
        output
    }

    /// Predicts the output of the network given a single input vector, returning the final activation.
    /// # Arguments
    /// - `input`: A 1D array representing a single input vector to the network.
    pub fn predict_single(&self, input: ArrayView1<f32>) -> Array1<f32> {
        let inputs = input.insert_axis(Axis(1));
        self.predict(inputs).column(0).to_owned()
    }
}

impl NeuronLayerSpec {
    /// Creates specifications for multiple hidden layers with the same activation function.
    /// # Arguments
    /// - `neurons`: An iterator over the number of neurons for each hidden layer.
    /// - `activation`: The activation function to be used for all hidden layers.
    /// # Returns
    /// A vector of `NeuronLayerSpec` instances, one for each hidden layer.
    pub fn hidden<A: Activation + 'static>(
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
    /// - For binary classification (n_classes = 2): 1 neuron with sigmoid activation
    /// - For multi-class classification (n_classes > 2): n_classes neurons with softmax activation
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
        match n_classes {
            2 => NeuronLayerSpec {
                neurons: 1,
                activation: SIGMOID.clone(),
            },
            _ => NeuronLayerSpec {
                neurons: n_classes,
                activation: SOFTMAX.clone(),
            },
        }
    }

    /// Creates a full network specification including hidden layers and an output layer.
    /// # Arguments
    /// - `hidden_neurons`: An iterator over the number of neurons for each hidden layer.
    /// - `hidden_activation`: The activation function to be used for all hidden layers.
    /// - `n_classes`: The number of classes for the output layer.
    /// # Returns
    /// A vector of `NeuronLayerSpec` instances, including hidden layers and the output layer.
    pub fn network_for<A: Activation + 'static>(
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
    /// This is the single entry point for turning the architecture choice (explicit
    /// hidden layers vs. inferred from the dataset) into layer specs; it delegates to
    /// [`network_for`](Self::network_for) or [`infer_from`](Self::infer_from). The
    /// dataset-shape arguments are only consulted for [`LayerPlan::Auto`].
    pub fn plan<A: Activation + 'static>(
        plan: LayerPlan,
        n_features: usize,
        n_classes: usize,
        n_samples: usize,
        hidden_activation: &Arc<A>,
    ) -> Vec<Self> {
        match plan {
            LayerPlan::Auto => {
                Self::infer_from(n_features, n_classes, n_samples, hidden_activation)
            }
            LayerPlan::Explicit(layers) => Self::network_for(layers, hidden_activation, n_classes),
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
    pub fn infer_from<A: Activation + 'static>(
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
    use crate::activations::{RELU, SIGMOID};
    use ndarray::{Array1, Array2, array};

    fn make_network(
        weights: Array2<f32>,
        biases: Array1<f32>,
        activation: Arc<dyn Activation>,
    ) -> NeuralNetwork {
        NeuralNetwork {
            layers: vec![NeuronLayer {
                weights,
                biases,
                activation,
            }],
        }
    }

    #[test]
    fn layer_accessors_report_dimensions_and_spec() {
        // 2 neurons, each taking 3 inputs.
        let layer = NeuronLayer {
            weights: Array2::zeros((2, 3)),
            biases: Array1::zeros(2),
            activation: RELU.clone(),
        };
        assert_eq!(layer.size(), 2);
        assert_eq!(layer.input_size(), 3);

        let spec = layer.spec();
        assert_eq!(spec.neurons, 2);
        assert_eq!(spec.activation.name(), "relu");
    }

    #[test]
    fn network_reports_specs_input_size_and_summary() {
        // 3 inputs -> 4 relu -> 3 softmax
        let specs = NeuronLayerSpec::network_for(vec![4], &*RELU, 3);
        let model = NeuralNetwork::initialization(3, &specs, 0);

        assert_eq!(model.input_size(), 3);

        let reported = model.specs();
        assert_eq!(reported.len(), 2);
        assert_eq!(reported[0].neurons, 4);
        assert_eq!(reported[1].neurons, 3);

        assert_eq!(model.summary(), "[3] -> 4-relu -> 3-softmax");
    }

    #[test]
    fn output_shape_matches_architecture() {
        // Network: 3 inputs -> 4 hidden (relu) -> 3 output (softmax), 5 samples
        // Note: n_classes=2 produces 1 sigmoid neuron (binary); use 3 for multi-class
        let specs = NeuronLayerSpec::network_for(vec![4], &*RELU, 3);
        let model = NeuralNetwork::initialization(3, &specs, 0);
        let inputs = Array2::zeros((3, 5)); // (features, samples)
        let output = model.predict(inputs.view());
        assert_eq!(output.shape(), &[3, 5]); // (classes, samples)
    }

    #[test]
    fn zero_weights_output_depends_only_on_bias() {
        // weights = 0 -> pre-activation = bias, regardless of input
        let weights = Array2::zeros((2, 3)); // 2 neurons, 3 inputs
        let biases = Array1::from_vec(vec![1.0, -1.0]);
        let model = make_network(weights, biases, RELU.clone());

        // Any input should produce relu(1.0)=1.0 and relu(-1.0)=0.0
        let inputs = array![[5.0, 9.0], [2.0, 7.0], [8.0, 3.0]];
        let output = model.predict(inputs.view());
        for col in output.columns() {
            assert!((col[0] - 1.0).abs() < 1e-6);
            assert!((col[1] - 0.0).abs() < 1e-6);
        }
    }

    #[test]
    fn identity_weights_with_relu_passes_positive_inputs() {
        // weights = I, bias = 0, relu -> output == input for positive values
        let model = make_network(Array2::eye(3), Array1::zeros(3), RELU.clone());
        let inputs = array![[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]];
        let output = model.predict(inputs.view());
        assert!((&output - &inputs).mapv(f32::abs).iter().all(|&v| v < 1e-6));
    }

    #[test]
    fn sigmoid_output_always_in_zero_one() {
        let specs = NeuronLayerSpec::network_for(vec![], &*SIGMOID, 2);
        let model = NeuralNetwork::initialization(4, &specs, 0);
        // f32 saturates to exactly 0.0 or 1.0 for large inputs, so use closed interval
        let inputs = array![
            [100.0, -100.0],
            [100.0, -100.0],
            [100.0, -100.0],
            [100.0, -100.0]
        ];
        let output = model.predict(inputs.view());
        for &v in output.iter() {
            assert!(
                (0.0..=1.0).contains(&v),
                "Sigmoid output {} not in [0, 1]",
                v
            );
        }
    }

    #[test]
    fn predict_returns_last_forward_activation() {
        let specs = NeuronLayerSpec::network_for(vec![4], &*RELU, 2);
        let model = NeuralNetwork::initialization(3, &specs, 0);
        let inputs = Array2::zeros((3, 5));

        let activations = model.forward(inputs.view());
        let predicted = model.predict(inputs.view());

        assert_eq!(predicted, *activations.last().unwrap());
    }

    #[test]
    #[should_panic(expected = "non-finite predictions")]
    fn predict_panics_when_model_has_diverged() {
        let mut model = make_network(Array2::zeros((1, 2)), Array1::zeros(1), SIGMOID.clone());
        model.layers[0].weights = array![[f32::NAN, 0.0]];
        let inputs = array![[1.0], [1.0]];
        model.predict(inputs.view());
    }

    #[test]
    fn is_finite_returns_true_for_valid_weights() {
        let model = make_network(Array2::zeros((1, 2)), Array1::zeros(1), SIGMOID.clone());
        assert!(model.is_finite());
    }

    #[test]
    fn is_finite_detects_nan_in_weights() {
        let mut model = make_network(Array2::zeros((1, 2)), Array1::zeros(1), SIGMOID.clone());
        model.layers[0].weights[[0, 0]] = f32::NAN;
        assert!(!model.is_finite());
    }

    #[test]
    fn is_finite_detects_inf_in_biases() {
        let mut model = make_network(Array2::zeros((1, 2)), Array1::zeros(1), SIGMOID.clone());
        model.layers[0].biases[0] = f32::INFINITY;
        assert!(!model.is_finite());
    }

    #[test]
    fn n_classes_derives_from_output_layer_size() {
        // 1 sigmoid output -> binary (2 classes)
        let binary =
            NeuralNetwork::initialization(3, &NeuronLayerSpec::network_for(vec![4], &*RELU, 2), 0);
        assert_eq!(binary.n_classes(), 2);

        // k softmax outputs -> k classes
        let multi =
            NeuralNetwork::initialization(3, &NeuronLayerSpec::network_for(vec![4], &*RELU, 5), 0);
        assert_eq!(multi.n_classes(), 5);
    }

    #[test]
    fn predict_single_matches_predict_on_one_sample() {
        let specs = NeuronLayerSpec::network_for(vec![4], &*RELU, 2);
        let model = NeuralNetwork::initialization(3, &specs, 0);

        let sample = array![1.0, 2.0, 3.0];
        let single = model.predict_single(sample.view());

        let batch = array![[1.0], [2.0], [3.0]]; // (features, 1 sample)
        let batch_output = model.predict(batch.view());

        assert!(
            (single - batch_output.column(0))
                .mapv(f32::abs)
                .iter()
                .all(|&v| v < 1e-6)
        );
    }

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
        // Explicit plan ignores the dataset-shape args and uses the given layers verbatim.
        let specs = NeuronLayerSpec::plan(LayerPlan::Explicit(vec![8, 4]), 2, 3, 1000, &RELU);
        assert_eq!(specs.len(), 3, "expected 2 hidden + 1 output specs");
        assert_eq!(specs[0].neurons, 8);
        assert_eq!(specs[1].neurons, 4);
        assert_eq!(specs[2].neurons, 3, "output layer matches n_classes");
    }

    #[test]
    fn plan_auto_matches_infer_from() {
        // Auto plan defers to infer_from: ln(2 * 2 / 1000) ≈ -5.5 → 1 hidden layer.
        let specs = NeuronLayerSpec::plan(LayerPlan::Auto, 2, 2, 1000, &RELU);
        assert_eq!(specs.len(), 2, "expected 1 hidden + 1 output spec");
    }
}
