//! This module defines the core data structures for representing neural network architectures.
//!
//! It provides types for individual neuron layers ([`NeuronLayer`]) and entire networks ([`NeuralNetwork`]).
//! Each layer contains its weights, biases, and activation function, enabling flexible and modular
//! construction of multi-layer perceptron and similar models.

use crate::activations::{Activation, SIGMOID, SOFTMAX};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use std::iter::once;
use std::sync::Arc;

/// Represents a single layer in a neural network, containing weights and biases.
#[derive(Clone)]
pub struct NeuronLayer {
    /// A 2D array where each row corresponds to a neuron and each column corresponds to an input feature.
    pub weights: Array2<f32>,
    /// A 1D array where each element is the bias for the corresponding neuron.
    pub biases: Array1<f32>,
    /// The activation function applied to the output of this layer.
    pub activation: Arc<dyn Activation>,
}

/// Represents a neural network composed of multiple layers of neurons.
#[derive(Clone)]
pub struct NeuralNetwork {
    /// A vector of [`NeuronLayer`] instances, defining the architecture of the network.
    pub layers: Vec<NeuronLayer>,
}

/// Represents the specifications for a neuron layer in a neural network.
pub struct NeuronLayerSpec {
    /// The number of neurons in this layer.
    pub neurons: usize,
    /// The activation function used in this layer.
    pub activation: Arc<dyn Activation>,
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
    /// Initializes a new `NeuronLayer` with random weights and biases.
    /// # Panics
    /// - When `neurons` or `inputs` are less than or equal to zero.
    /// # Arguments
    /// - `inputs`: The number of inputs to this layer (i.e., the number of neurons in the previous layer).
    /// - `spec`: The specifications for this layer, including the number of neurons and the activation method.
    pub fn initialization(inputs: usize, spec: &NeuronLayerSpec) -> Self {
        assert!(
            spec.neurons > 0 && inputs > 0,
            "Neurons and inputs must be greater than zero."
        );

        let (weights, biases) = spec
            .activation
            .initialization()
            .apply((spec.neurons, inputs));

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
    /// - A 2D array representing the outputs of this layer after applying the sigmoid activation function.
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
    /// Creates a new `NeuronNetwork` with the specified input size and layer specifications.
    /// # Arguments
    /// - `inputs`: The number of inputs to the first layer of the network.
    /// - `layer_specs`: A vector of `NeuronLayerSpec` representing the specifications for each layer in the network.
    pub fn initialization(inputs: usize, layer_specs: &Vec<NeuronLayerSpec>) -> Self {
        assert!(inputs > 0, "Input size must be greater than zero.");
        assert!(
            !layer_specs.is_empty(),
            "At least one layer must be specified."
        );

        let mut layers = Vec::with_capacity(layer_specs.len());
        let mut layer_input = inputs;

        for layer_spec in layer_specs {
            layers.push(NeuronLayer::initialization(layer_input, &layer_spec));
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
                acc.push(layer.forward(&acc.last().unwrap()));
                acc
            })
    }

    /// Predicts the output of the network given the inputs, returning the final activations.
    /// # Arguments
    /// - `inputs`: A 2D array representing the inputs to the network.
    pub fn predict(&self, inputs: ArrayView2<f32>) -> Array2<f32> {
        let activations = self.forward(inputs);
        last_activation(&activations)
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
    /// When `n_classes` is less than or equal to 1.
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
}
