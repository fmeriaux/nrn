//! This module defines the core data structures for representing neural network architectures.
//!
//! It provides types for individual neuron layers ([`NeuronLayer`]) and entire networks ([`NeuralNetwork`]).
//! Each layer contains its weights, biases, and activation function, enabling flexible and modular
//! construction of multi-layer perceptron and similar models.

use crate::activations::{Activation, SIGMOID, SOFTMAX};
use crate::data::scalers::{Scaler, ScalerFeatureMismatch, ScalerMethod};
use crate::layers::{Dense, Layer};
use ndarray::{Array1, Array2, ArrayD, ArrayView, ArrayView1, ArrayView2, Axis, Dimension, Ix2};
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand::rngs::StdRng;
use std::fmt;
use std::iter::once;
use std::sync::Arc;

/// Represents a neural network composed of a stack of layers.
#[derive(Clone, Debug)]
pub struct NeuralNetwork {
    /// The layers of the network, applied in order from input to output.
    layers: Vec<Box<dyn Layer>>,
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

/// Returns the last activation from a vector of activations.
/// # Panics
/// - When the `activations` vector is empty.
pub fn last_activation(activations: &[ArrayD<f32>]) -> ArrayD<f32> {
    activations
        .last()
        .expect("forward always yields at least the input activation")
        .to_owned()
}

impl NeuralNetwork {
    /// Assembles a network from a stack of layers, applied in order from input to output.
    /// # Panics
    /// When `layers` is empty.
    pub fn new(layers: Vec<Box<dyn Layer>>) -> Self {
        assert!(
            !layers.is_empty(),
            "A network must have at least one layer."
        );
        NeuralNetwork { layers }
    }

    /// Assembles a single-layer network from one layer.
    pub fn single(layer: impl Layer + 'static) -> Self {
        Self::new(vec![Box::new(layer)])
    }

    /// Appends a layer to the network, returning the network for chaining.
    /// # Panics
    /// When the layer's input shape does not match the previous layer's output shape.
    pub fn with_layer(mut self, layer: impl Layer + 'static) -> Self {
        if let Some(last) = self.layers.last() {
            assert_eq!(
                layer.input_shape(),
                last.output_shape(),
                "Layer input shape must match the previous layer's output shape."
            );
        }
        self.layers.push(Box::new(layer));
        self
    }

    /// The network's layers, in order from input to output.
    pub fn layers(&self) -> &[Box<dyn Layer>] {
        &self.layers
    }

    /// The network's layers, mutably, in order from input to output.
    pub(crate) fn layers_mut(&mut self) -> &mut [Box<dyn Layer>] {
        &mut self.layers
    }

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
            layers.push(
                Box::new(Dense::initialization(layer_input, layer_spec, &mut rng))
                    as Box<dyn Layer>,
            );
            layer_input = layer_spec.neurons;
        }

        Self::new(layers)
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
            .output_size();
        if out == 1 { 2 } else { out }
    }

    /// Returns a summary of the network's architecture as a string,
    /// showing the number of neurons in each layer, including the input layer.
    pub fn summary(&self) -> String {
        once(format!("[{}]", self.input_size()))
            .chain(
                self.layers
                    .iter()
                    .map(|layer| match layer.activation_name() {
                        Some(activation) => format!("{}-{}", layer.output_size(), activation),
                        None => layer.output_size().to_string(),
                    }),
            )
            .collect::<Vec<String>>()
            .join(" -> ")
    }

    /// Computes the forward pass through the network, returning the activations of each layer.
    ///
    /// The input is rank-agnostic: its last axis is the sample axis and its leading axes are the
    /// per-sample features, so a dense network takes a rank-2 `(features, samples)` batch while a
    /// spatial one (e.g. a leading `Conv2d`) takes a higher-rank `(channels, height, width, samples)`
    /// batch. Each layer reshapes its input as it needs.
    /// # Errors
    /// [`InputShapeMismatch`] when `inputs`' leading (feature) axes differ from the network's
    /// input shape.
    /// # Arguments
    /// - `inputs`: An array whose last axis is samples and whose leading axes are the features.
    pub fn forward<D: Dimension>(
        &self,
        inputs: ArrayView<f32, D>,
    ) -> Result<Vec<ArrayD<f32>>, InputShapeMismatch> {
        let inputs = inputs.into_dyn();
        self.validate_inputs(inputs.view())?;

        Ok(self
            .layers
            .iter()
            .fold(vec![inputs.to_owned()], |mut acc, layer| {
                let output = layer.forward(acc.last().unwrap().view());
                acc.push(output);
                acc
            }))
    }

    /// Returns `true` when all layers are finite (no NaN or Inf in any weight or bias).
    pub fn is_finite(&self) -> bool {
        self.layers.iter().all(|layer| layer.is_finite())
    }

    /// Validates that `inputs` carry the per-sample shape this network expects: the sample axis
    /// is last, so the leading axes are the per-sample features and must equal the first layer's
    /// [`input_shape`](crate::layers::Layer::input_shape).
    ///
    /// # Errors
    /// [`InputShapeMismatch`] when the leading axes differ from the first layer's input shape.
    pub fn validate_inputs<D: Dimension>(
        &self,
        inputs: ArrayView<f32, D>,
    ) -> Result<(), InputShapeMismatch> {
        let expected = self.layers[0].input_shape();
        let shape = inputs.shape();
        let found = &shape[..shape.len().saturating_sub(1)];
        (found == expected)
            .then_some(())
            .ok_or_else(|| InputShapeMismatch {
                expected,
                found: found.to_vec(),
            })
    }

    /// Predicts the output of the network given the inputs, returning the final activations.
    /// # Arguments
    /// - `inputs`: An array whose last axis is samples and whose leading axes are the features
    ///   (rank-2 `(features, samples)` for a dense network, higher-rank for a spatial one).
    /// # Errors
    /// [`InputShapeMismatch`] when the leading (feature) axes differ from the network's input shape.
    pub fn predict<D: Dimension>(
        &self,
        inputs: ArrayView<f32, D>,
    ) -> Result<Array2<f32>, InputShapeMismatch> {
        let output = last_activation(&self.forward(inputs)?)
            .into_dimensionality::<Ix2>()
            .expect("the output layer produces rank-2 (classes, samples) activations");
        assert!(
            output.iter().all(|v| v.is_finite()),
            "non-finite predictions (NaN or inf): the model likely diverged during training"
        );
        Ok(output)
    }

    /// Predicts the output of the network given a single input vector, returning the final activation.
    /// # Arguments
    /// - `input`: A 1D array representing a single input vector to the network.
    /// # Errors
    /// [`InputShapeMismatch`] when `input`'s length does not match [`Self::input_size`].
    pub fn predict_single(
        &self,
        input: ArrayView1<f32>,
    ) -> Result<Array1<f32>, InputShapeMismatch> {
        let inputs = input.insert_axis(Axis(1));
        Ok(self.predict(inputs)?.column(0).to_owned())
    }
}

/// A trained [`NeuralNetwork`] paired with the scaler fitted alongside it.
#[derive(Clone, Debug)]
pub struct Predictor {
    /// The trained network.
    pub network: NeuralNetwork,
    /// The scaler applied to raw inputs before prediction, when one is present.
    pub scaler: Option<ScalerMethod>,
}

impl Predictor {
    /// Pairs a network with an optional scaler.
    pub fn new(network: NeuralNetwork, scaler: Option<ScalerMethod>) -> Self {
        Self { network, scaler }
    }

    /// Predicts on a single raw input vector, applying the scaler first when present.
    ///
    /// # Errors
    /// [`PredictionError::Scaling`] when a scaler is present and the input does not match
    /// its fitted feature count, or [`PredictionError::Network`] when it does not match the
    /// network's input size.
    pub fn predict_single(&self, input: ArrayView1<f32>) -> Result<Array1<f32>, PredictionError> {
        let mut input = input.to_owned();
        if let Some(scaler) = &self.scaler {
            scaler.apply_single_inplace(input.view_mut())?;
        }
        Ok(self.network.predict_single(input.view())?)
    }

    /// Predicts on a batch of raw inputs `(features, samples)`, applying the scaler
    /// first when present.
    ///
    /// # Errors
    /// [`PredictionError::Scaling`] when a scaler is present and the inputs do not match
    /// its fitted feature count, or [`PredictionError::Network`] when they do not match the
    /// network's input size.
    pub fn predict(&self, inputs: ArrayView2<f32>) -> Result<Array2<f32>, PredictionError> {
        let mut inputs = inputs.to_owned();
        if let Some(scaler) = &self.scaler {
            scaler.apply_inplace(inputs.view_mut().reversed_axes())?;
        }
        Ok(self.network.predict(inputs.view())?)
    }
}

/// An instance's per-sample shape did not match the network's input shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputShapeMismatch {
    /// The per-sample input shape the network expects.
    pub expected: Vec<usize>,
    /// The per-sample shape the instance carries.
    pub found: Vec<usize>,
}

impl fmt::Display for InputShapeMismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "instance has shape {:?} but the model expects {:?}",
            self.found, self.expected
        )
    }
}

impl std::error::Error for InputShapeMismatch {}

/// A [`Predictor`] rejected an instance: either its scaler or its network found the
/// wrong number of features.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PredictionError {
    /// The scaler's fitted feature count did not match the input.
    Scaling(ScalerFeatureMismatch),
    /// The network's input shape did not match the input.
    Network(InputShapeMismatch),
}

impl fmt::Display for PredictionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PredictionError::Scaling(e) => write!(f, "{e}"),
            PredictionError::Network(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for PredictionError {}

impl From<ScalerFeatureMismatch> for PredictionError {
    fn from(error: ScalerFeatureMismatch) -> Self {
        PredictionError::Scaling(error)
    }
}

impl From<InputShapeMismatch> for PredictionError {
    fn from(error: InputShapeMismatch) -> Self {
        PredictionError::Network(error)
    }
}

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
    use crate::activations::{RELU, SIGMOID};
    use ndarray::{Array, Array1, Array2, IxDyn, array};

    fn make_network(
        weights: Array2<f32>,
        biases: Array1<f32>,
        activation: Arc<dyn Activation>,
    ) -> NeuralNetwork {
        NeuralNetwork::single(Dense::new(weights, biases, activation))
    }

    /// Downcasts a network's layer back to the concrete [`Dense`] to read or perturb its
    /// weights and biases in tests.
    fn dense_mut(model: &mut NeuralNetwork, index: usize) -> &mut Dense {
        model.layers[index]
            .as_any_mut()
            .downcast_mut::<Dense>()
            .unwrap()
    }

    #[test]
    fn network_reports_input_size_and_summary() {
        // 3 inputs -> 4 relu -> 3 softmax
        let specs = NeuronLayerSpec::network_for(vec![4], &*RELU, 3);
        let model = NeuralNetwork::initialization(3, &specs, 0);

        assert_eq!(model.input_size(), 3);
        assert_eq!(model.layers.len(), 2);
        assert_eq!(model.layers[0].output_size(), 4);
        assert_eq!(model.layers[1].output_size(), 3);

        assert_eq!(model.summary(), "[3] -> 4-relu -> 3-softmax");
    }

    #[test]
    fn convolutional_network_predicts_end_to_end_on_a_spatial_batch() {
        use crate::layers::{Conv2d, Flatten};

        // Conv2d (1×4×4 → 2×2×2) → Flatten (8) → Dense (1 sigmoid, binary). The rank-4
        // spatial batch threads through forward/predict unchanged; the Flatten collapses it
        // to the rank-2 the Dense head consumes, and predict yields (classes, samples).
        let conv = Conv2d::initialization(
            (1, 4, 4),
            2,
            (3, 3),
            1,
            0,
            RELU.clone(),
            &mut StdRng::seed_from_u64(0),
        );
        let head = Dense::initialization(
            8,
            &NeuronLayerSpec::output_for(2),
            &mut StdRng::seed_from_u64(1),
        );
        let model = NeuralNetwork::single(conv)
            .with_layer(Flatten::new(vec![2, 2, 2]))
            .with_layer(head);

        // A spatial input the dense predict path could never accept: (channels, height, width, samples).
        let inputs = Array::from_shape_fn(IxDyn(&[1, 4, 4, 3]), |idx| {
            ((idx[1] + idx[2] + idx[3]) as f32).sin()
        });
        let output = model.predict(inputs.view()).unwrap();
        assert_eq!(output.shape(), &[1, 3]); // (classes, samples)
        assert!(output.iter().all(|&v| (0.0..=1.0).contains(&v)));
    }

    #[test]
    fn validate_inputs_rejects_a_matching_count_but_wrong_arrangement() {
        use crate::layers::Conv2d;

        // A Conv2d-first network expects a per-sample shape of (1, 4, 4) = 16 features. A
        // rank-2 (16, samples) batch has the right feature count but the wrong arrangement:
        // the product check accepted it, the shape check rejects it.
        let conv = Conv2d::initialization(
            (1, 4, 4),
            2,
            (3, 3),
            1,
            0,
            RELU.clone(),
            &mut StdRng::seed_from_u64(0),
        );
        let model = NeuralNetwork::single(conv);

        let flat = Array2::<f32>::zeros((16, 5)); // 16 features, 5 samples — right count, flat.
        let err = model.validate_inputs(flat.view()).unwrap_err();
        assert_eq!(err.expected, vec![1, 4, 4]);
        assert_eq!(err.found, vec![16]);

        // The correctly-shaped rank-4 batch passes.
        let spatial = ArrayD::<f32>::zeros(IxDyn(&[1, 4, 4, 5]));
        assert!(model.validate_inputs(spatial.view()).is_ok());
    }

    #[test]
    #[should_panic(expected = "Layer input shape must match the previous layer's output shape")]
    fn with_layer_rejects_rank_mismatch_at_build_time() {
        use crate::layers::Conv2d;

        // Conv2d outputs a rank-3 (out_channels, out_h, out_w) per sample; feeding it straight
        // into a Dense (rank-1 input) without a Flatten is a shape mismatch the build rejects,
        // even though the feature counts could line up.
        let conv = Conv2d::initialization(
            (1, 4, 4),
            2,
            (3, 3),
            1,
            0,
            RELU.clone(),
            &mut StdRng::seed_from_u64(0),
        );
        // Conv2d output is (2, 2, 2) = 8 features; a Dense taking 8 inputs matches on count.
        let head = Dense::initialization(
            8,
            &NeuronLayerSpec::output_for(2),
            &mut StdRng::seed_from_u64(1),
        );
        let _ = NeuralNetwork::single(conv).with_layer(head);
    }

    #[test]
    fn output_shape_matches_architecture() {
        // Network: 3 inputs -> 4 hidden (relu) -> 3 output (softmax), 5 samples
        // Note: n_classes=2 produces 1 sigmoid neuron (binary); use 3 for multi-class
        let specs = NeuronLayerSpec::network_for(vec![4], &*RELU, 3);
        let model = NeuralNetwork::initialization(3, &specs, 0);
        let inputs = Array2::zeros((3, 5)); // (features, samples)
        let output = model.predict(inputs.view()).unwrap();
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
        let output = model.predict(inputs.view()).unwrap();
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
        let output = model.predict(inputs.view()).unwrap();
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
        let output = model.predict(inputs.view()).unwrap();
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

        let activations = model.forward(inputs.view()).unwrap();
        let predicted = model.predict(inputs.view()).unwrap();

        assert_eq!(predicted.into_dyn(), *activations.last().unwrap());
    }

    #[test]
    #[should_panic(expected = "non-finite predictions")]
    fn predict_panics_when_model_has_diverged() {
        let mut model = make_network(Array2::zeros((1, 2)), Array1::zeros(1), SIGMOID.clone());
        *dense_mut(&mut model, 0).affine_mut().weights_mut() = array![[f32::NAN, 0.0]];
        let inputs = array![[1.0], [1.0]];
        let _ = model.predict(inputs.view());
    }

    #[test]
    fn is_finite_returns_true_for_valid_weights() {
        let model = make_network(Array2::zeros((1, 2)), Array1::zeros(1), SIGMOID.clone());
        assert!(model.is_finite());
    }

    #[test]
    fn is_finite_detects_nan_in_weights() {
        let mut model = make_network(Array2::zeros((1, 2)), Array1::zeros(1), SIGMOID.clone());
        dense_mut(&mut model, 0).affine_mut().weights_mut()[[0, 0]] = f32::NAN;
        assert!(!model.is_finite());
    }

    #[test]
    fn is_finite_detects_inf_in_biases() {
        let mut model = make_network(Array2::zeros((1, 2)), Array1::zeros(1), SIGMOID.clone());
        dense_mut(&mut model, 0).affine_mut().biases_mut()[0] = f32::INFINITY;
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
        let single = model.predict_single(sample.view()).unwrap();

        let batch = array![[1.0], [2.0], [3.0]]; // (features, 1 sample)
        let batch_output = model.predict(batch.view()).unwrap();

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
