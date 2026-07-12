//! The [`NeuralNetwork`]: a stack of layers applied in order, plus the per-stage
//! [`Activations`] a forward pass captures and the [`InputShapeMismatch`] it can raise.

use crate::activations::Activation;
use crate::layers::{Dense, Layer, format_shape};
use crate::model::{LayerSpec, LayerSpecError, NeuronLayerSpec};
use ndarray::{ArrayD, ArrayView, ArrayViewD, Dimension};
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand::rngs::StdRng;
use std::collections::HashMap;
use std::fmt;
use std::iter::once;

/// Represents a neural network composed of a stack of layers.
#[derive(Clone, Debug)]
pub struct NeuralNetwork {
    /// The layers of the network, applied in order from input to output.
    layers: Vec<Box<dyn Layer>>,
}

/// The per-stage activations captured by a [`forward`](NeuralNetwork::forward) pass: stage 0
/// is the network input and stage `i + 1` is the output of layer `i`. There is always at
/// least the input stage, so [`output`](Activations::output) never fails.
#[derive(Clone, Debug)]
pub struct Activations(Vec<ArrayD<f32>>);

impl Activations {
    /// Wraps the per-stage activations of a forward pass, input first.
    /// # Panics
    /// When `stages` is empty. A forward pass always yields at least the input stage.
    pub(crate) fn new(stages: Vec<ArrayD<f32>>) -> Self {
        assert!(
            !stages.is_empty(),
            "Activations must carry at least the input stage."
        );
        Activations(stages)
    }

    /// The network's output: the final stage, i.e. the output layer's activations (logits).
    pub fn output(&self) -> ArrayViewD<'_, f32> {
        self.0.last().expect("Activations is never empty").view()
    }

    /// The per-stage activations in order, input first: stage 0 is the input, stage `i + 1`
    /// the output of layer `i`.
    pub fn stages(&self) -> &[ArrayD<f32>] {
        &self.0
    }

    /// Consumes into the owned per-stage activations, in order and input first.
    pub fn into_stages(self) -> Vec<ArrayD<f32>> {
        self.0
    }

    /// Consumes into the owned output stage, discarding the earlier stages.
    pub fn into_output(self) -> ArrayD<f32> {
        self.0
            .into_iter()
            .next_back()
            .expect("Activations is never empty")
    }

    /// Applies `activation` in place to the [`output`](Activations::output) stage, leaving the
    /// earlier stages untouched.
    pub fn finalize(&mut self, activation: &dyn Activation) {
        let output = self.0.last_mut().expect("Activations is never empty");
        activation.apply_inplace(output);
    }
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

    /// Rebuilds a network from its `input_shape` and each layer's [`LayerSpec`] paired with its
    /// named tensors, in layer order. The input shape is threaded through the stack, so every
    /// layer is built at the shape the previous one produces.
    pub fn from_specs_and_weights(
        input_shape: Vec<usize>,
        layers: Vec<(LayerSpec, HashMap<String, ArrayD<f32>>)>,
    ) -> Result<Self, LayerSpecError> {
        let mut shape = input_shape;
        let layers = layers
            .into_iter()
            .map(|(spec, tensors)| {
                let layer = spec.from_tensors(&shape, tensors)?;
                shape = layer.output_shape();
                Ok(layer)
            })
            .collect::<Result<Vec<_>, LayerSpecError>>()?;
        Ok(Self::new(layers))
    }

    /// Returns the input size of the network, which is the number of inputs to the first layer.
    pub fn input_size(&self) -> usize {
        self.layers[0].input_size()
    }

    /// The network's per-sample input shape: the first layer's
    /// [`input_shape`](crate::layers::Layer::input_shape), sample axis excluded.
    pub fn input_shape(&self) -> Vec<usize> {
        self.layers[0].input_shape()
    }

    /// This network's architecture as a stack of weight-free [`LayerSpec`]s, in order — the
    /// declarative counterpart to its tensors.
    pub fn specs(&self) -> Vec<LayerSpec> {
        self.layers.iter().map(|layer| layer.spec()).collect()
    }

    /// Returns a summary of the network's architecture as a string, showing the per-sample
    /// input shape and each layer's output shape (with its activation, when it has one).
    pub fn summary(&self) -> String {
        once(format_shape(&self.layers[0].input_shape()))
            .chain(self.layers.iter().map(|layer| layer.summary()))
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
    ) -> Result<Activations, InputShapeMismatch> {
        let inputs = inputs.into_dyn();
        self.validate_inputs(inputs.view())?;

        let stages = self
            .layers
            .iter()
            .fold(vec![inputs.to_owned()], |mut acc, layer| {
                let output = layer.forward(acc.last().unwrap().view());
                acc.push(output);
                acc
            });
        Ok(Activations::new(stages))
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

    /// Computes the forward pass of the network and returns the output layer's activations.
    ///
    /// # Dimension Layout (samples-last)
    /// The returned `ArrayD<f32>` retains the sample and spatial dimensions of the input,
    /// but prepends the class dimension as the first axis (axis 0).
    /// - For a dense 2D input `(features, samples)`, the output is a 2D array of shape `(n_classes, samples)`.
    /// - For a spatial 4D input `(channels, H, W, samples)`, the output is a 4D array of shape `(n_classes, H, W, samples)`.
    ///
    /// The final layer emits raw logits (an identity or otherwise non-saturating output
    /// activation); the softmax/sigmoid is folded into the loss and accuracy metrics.
    ///
    /// # Errors
    /// [`InputShapeMismatch`] when the leading (feature) axes differ from the network's input shape.
    pub fn output<D: Dimension>(
        &self,
        inputs: ArrayView<f32, D>,
    ) -> Result<ArrayD<f32>, InputShapeMismatch> {
        Ok(self.forward(inputs)?.output().to_owned())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::{Activation, RELU, SIGMOID};
    use ndarray::{Array1, Array2, IxDyn, array};
    use std::sync::Arc;

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
        // 3 inputs -> 4 relu -> 3 identity (logits)
        let specs = NeuronLayerSpec::network_for(vec![4], &*RELU, 3);
        let model = NeuralNetwork::initialization(3, &specs, 0);

        assert_eq!(model.input_size(), 3);
        assert_eq!(model.layers.len(), 2);
        assert_eq!(model.layers[0].output_size(), 4);
        assert_eq!(model.layers[1].output_size(), 3);

        assert_eq!(model.summary(), "[3] -> [4]-relu -> [3]-identity");
    }

    #[test]
    fn summary_shows_per_sample_shapes_and_omits_the_suffix_for_a_layer_without_activation() {
        use crate::layers::Flatten;

        // Flatten (2×2×2 → 8) preserves the spatial input shape in the summary and carries
        // no activation, so its segment is the bare output shape with no "-activation"
        // suffix; the Dense head keeps its suffix.
        let model =
            NeuralNetwork::single(Flatten::new(vec![2, 2, 2])).with_layer(Dense::initialization(
                8,
                &NeuronLayerSpec::output_for(2),
                &mut StdRng::seed_from_u64(0),
            ));

        assert_eq!(model.summary(), "[2, 2, 2] -> [8] -> [1]-identity");
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
        let output = model.output(inputs.view()).unwrap();
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
        let output = model.output(inputs.view()).unwrap();
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
        let output = model.output(inputs.view()).unwrap();
        assert!((&output - &inputs).mapv(f32::abs).iter().all(|&v| v < 1e-6));
    }

    #[test]
    fn predict_returns_last_forward_activation() {
        let specs = NeuronLayerSpec::network_for(vec![4], &*RELU, 2);
        let model = NeuralNetwork::initialization(3, &specs, 0);
        let inputs = Array2::zeros((3, 5));

        let activations = model.forward(inputs.view()).unwrap();
        let predicted = model.output(inputs.view()).unwrap();

        assert_eq!(predicted, activations.output().to_owned());
    }

    #[test]
    fn predict_passes_through_a_diverged_model_output() {
        // predict is a pure forward pass and does not guard against divergence (the trainer
        // does, via weight finiteness), so a NaN weight surfaces as a NaN prediction.
        let mut model = make_network(Array2::zeros((1, 2)), Array1::zeros(1), SIGMOID.clone());
        *dense_mut(&mut model, 0).affine_mut().weights_mut() = array![[f32::NAN, 0.0]];
        let inputs = array![[1.0], [1.0]];
        let prediction = model.output(inputs.view()).unwrap();
        assert!(prediction.iter().any(|v| v.is_nan()));
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

    /// Extracts a model's architecture and weights the way `io` will at the persistence
    /// boundary — pairing each [`LayerSpec`] with its named tensors — and reconstructs it via
    /// [`NeuralNetwork::from_specs_and_weights`].
    fn round_trip(model: &NeuralNetwork) -> NeuralNetwork {
        let layers = model
            .specs()
            .into_iter()
            .zip(model.layers())
            .map(|(spec, layer)| (spec, layer.named_tensors().into_iter().collect()))
            .collect();
        NeuralNetwork::from_specs_and_weights(model.input_shape(), layers).unwrap()
    }

    #[test]
    fn dense_network_round_trips_through_specs_and_weights() {
        // 3 inputs -> 4 relu -> 3 identity (logits).
        let specs = NeuronLayerSpec::network_for(vec![4], &*RELU, 3);
        let model = NeuralNetwork::initialization(3, &specs, 0);

        let rebuilt = round_trip(&model);

        assert_eq!(rebuilt.input_shape(), vec![3]);
        let inputs = Array2::from_shape_fn((3, 5), |(f, s)| (f + s) as f32);
        assert_eq!(
            rebuilt.output(inputs.view()).unwrap(),
            model.output(inputs.view()).unwrap(),
            "reconstructed network must predict identically"
        );
    }

    #[test]
    fn convolutional_network_round_trips_through_specs_and_weights() {
        use crate::layers::{Conv2d, Flatten};

        // Conv2d (1,4,4) -> (2,2,2) -> Flatten (8) -> Dense head (3 logits).
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
            &NeuronLayerSpec::output_for(3),
            &mut StdRng::seed_from_u64(1),
        );
        let model = NeuralNetwork::single(conv)
            .with_layer(Flatten::new(vec![2, 2, 2]))
            .with_layer(head);

        let rebuilt = round_trip(&model);

        assert_eq!(rebuilt.input_shape(), vec![1, 4, 4]);
        let inputs = ArrayD::from_shape_fn(IxDyn(&[1, 4, 4, 5]), |idx| {
            (idx[1] + idx[2] + idx[3]) as f32
        });
        assert_eq!(
            rebuilt.output(inputs.view()).unwrap(),
            model.output(inputs.view()).unwrap(),
            "reconstructed network must predict identically"
        );
    }
}
