use crate::core::activations::Activation;
use crate::synth::Dataset;
use ndarray::{Array1, Array2, Axis};
use ndarray_rand::rand_distr::num_traits::real::Real;
use std::iter::once;
use std::sync::Arc;

/// Represents a single layer in a neural network, containing weights and biases.
///
/// # Properties
/// - `weights`: A 2D array representing the weights of the neurons in this layer.
/// - `bias`: A 1D array representing the biases for each neuron in this layer.
/// - `activation`: The activation method used for the neurons in this layer.
#[derive(Clone)]
pub struct NeuronLayer {
    pub weights: Array2<f32>,
    pub bias: Array1<f32>,
    pub activation: Arc<dyn Activation>,
}

/// Represents a neural network composed of multiple layers of neurons.
///
/// # Properties
/// - `layers`: A vector of `NeuronLayer` instances, each representing a layer in the network.
#[derive(Clone)]
pub struct NeuronNetwork {
    pub layers: Vec<NeuronLayer>,
}

/// Represents the gradients computed during backpropagation for a single layer.
/// # Properties
/// - `dw`: A 2D array representing the gradients of the weights.
/// - `db`: A 1D array representing the gradients of the biases.
pub struct Gradients {
    pub dw: Array2<f32>,
    pub db: Array1<f32>,
}

/// Represents the specifications for a neuron layer in a neural network.
/// # Properties
/// - `neurons`: The number of neurons in this layer.
/// - `activation`: The activation method used for the neurons in this layer.
pub struct NeuronLayerSpec {
    pub neurons: usize,
    pub activation: Arc<dyn Activation>,
}

/// Returns the last activation from a vector of activations.
/// # Panics
/// - When the `activations` vector is empty.
pub(crate) fn last_activation(activations: &Vec<Array2<f32>>) -> Array2<f32> {
    activations
        .last()
        .cloned()
        .expect("Ensure activations is not empty.")
}

/// Computes the log loss between predictions and expectations.
/// # Panics
/// - When the lengths of `predictions` and `expectations` do not match.
pub(crate) fn log_loss(predictions: &Array2<f32>, expectations: &Array2<f32>) -> f32 {
    assert_eq!(
        predictions.len(),
        expectations.len(),
        "Predictions and expectations must have the same length."
    );

    let m = predictions.len() as f32;
    let ln = |vec: &Array2<f32>| vec.mapv(|val| f32::ln(val + f32::epsilon()));
    ((-expectations * ln(predictions) - (1.0 - expectations) * ln(&(1.0 - predictions))).sum()) / m
}

/// Computes the accuracy of predictions against expectations.
/// # Panics
/// - When the lengths of `predictions` and `expectations` do not match.
pub(crate) fn accuracy(predictions: &Array2<f32>, expectations: &Array2<f32>) -> f32 {
    assert_eq!(
        predictions.len(),
        expectations.len(),
        "Predictions and expectations must have the same length."
    );

    let total_samples = expectations.len() as f32;
    let correct = predictions
        .iter()
        .zip(expectations.iter())
        .filter(|&(pred, exp)| (pred - exp).abs() < 0.5)
        .count() as f32;

    correct / total_samples * 100.0
}

impl Dataset {
    pub fn to_model_shape(&self) -> (Array2<f32>, Array2<f32>) {
        let inputs: Array2<f32> = self.features.t().to_owned();

        let max_label = self.max_label();

        let expectations: Array2<f32> = if max_label > 1 {
            // If labels are not binary, we need to one-hot encode them
            let mut one_hot = Array2::zeros((max_label + 1, self.n_samples()));
            for (i, &label) in self.labels.iter().enumerate() {
                one_hot[[label as usize, i]] = 1.0;
            }
            one_hot
        } else {
            // If labels are binary, we can use them directly
            self.labels.to_owned().insert_axis(Axis(0))
        };

        (inputs, expectations)
    }
}

impl Gradients {
    /// Clips the gradients to a maximum norm, using the L2 norm.
    /// # Arguments
    /// - `max_norm`: The maximum norm to clip the gradients to.
    fn clip(&mut self, max_norm: f32) {
        let dw_norm = self.dw.iter().map(|x| x.powi(2)).sum::<f32>();
        let db_norm = self.db.iter().map(|x| x.powi(2)).sum::<f32>();
        let norm = (dw_norm + db_norm).sqrt();
        if norm > max_norm {
            let scale = max_norm / norm;
            self.dw.mapv_inplace(|x| x * scale);
            self.db.mapv_inplace(|x| x * scale);
        }
    }
}

impl NeuronLayer {
    /// Initializes a new `NeuronLayer` with random weights and biases.
    /// # Panics
    /// - When `neurons` or `inputs` are less than or equal to zero.
    /// # Arguments
    /// - `inputs`: The number of inputs to this layer (i.e., the number of neurons in the previous layer).
    /// - `spec`: The specifications for this layer, including the number of neurons and the activation method.
    fn initialization(inputs: usize, spec: &NeuronLayerSpec) -> Self {
        assert!(
            spec.neurons > 0 && inputs > 0,
            "Neurons and inputs must be greater than zero."
        );

        let initialization = spec.activation.get_initializer();
        let (weights, bias) = initialization.apply((spec.neurons, inputs));
        NeuronLayer {
            weights,
            bias,
            activation: spec.activation.clone(),
        }
    }

    /// Updates the weights and biases of this layer using the computed gradients and a learning rate.
    /// # Arguments
    /// - `gradients`: The gradients computed during backpropagation for this layer.
    /// - `learning_rate`: The learning rate to apply during the update.
    fn update(&mut self, gradients: Gradients, learning_rate: f32) {
        self.weights -= &(gradients.dw * learning_rate);
        self.bias -= &(gradients.db * learning_rate);
    }

    /// Computes the forward pass of this layer given the inputs.
    /// # Arguments
    /// - `inputs`: A 2D array representing the inputs to this layer.
    /// # Returns
    /// - A 2D array representing the outputs of this layer after applying the sigmoid activation function.
    fn forward(&self, inputs: &Array2<f32>) -> Array2<f32> {
        assert_eq!(
            inputs.nrows(),
            self.weights.ncols(),
            "Input shape does not match weights shape."
        );

        // Broadcasting bias to match the shape of the output
        let broadcasted_bias: Array2<f32> = self.bias.view().insert_axis(Axis(1)).to_owned();

        self.activation
            .apply(&(self.weights.dot(inputs) + &broadcasted_bias))
    }

    /// Returns the number of neurons in this layer.
    fn size(&self) -> usize {
        self.weights.nrows()
    }

    /// Returns the number of inputs to this layer.
    /// For example, this is the number of neurons in the previous layer,
    /// or the input size for the first layer.
    fn input_size(&self) -> usize {
        self.weights.ncols()
    }

    /// Returns the specifications of this layer.
    fn spec(&self) -> NeuronLayerSpec {
        NeuronLayerSpec {
            neurons: self.size(),
            activation: self.activation.clone(),
        }
    }
}

impl NeuronNetwork {
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

        NeuronNetwork { layers }
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
    fn forward(&self, inputs: &Array2<f32>) -> Vec<Array2<f32>> {
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
    pub fn predict(&self, inputs: &Array2<f32>) -> Array2<f32> {
        let activations = self.forward(inputs);
        last_activation(&activations)
    }

    /// Predicts the output of the network given a single input vector, returning the final activation.
    /// # Arguments
    /// - `input`: A 1D array representing a single input vector to the network.
    pub fn predict_single(&self, input: &Array1<f32>) -> Array1<f32> {
        let inputs = input.clone().insert_axis(Axis(1));
        self.predict(&inputs).column(0).to_owned()
    }

    /// Trains the network using the provided inputs and expectations, updating the weights and biases.
    /// # Panics
    /// - When the `learning_rate` is less than or equal to zero.
    /// - When the number of columns in `inputs` does not match the length of `expectations`.
    /// # Arguments
    /// - `inputs`: A 2D array representing the inputs to the network.
    /// - `expectations`: A 2D array representing the expected outputs for the inputs.
    /// - `learning_rate`: The learning rate to apply during the update.
    /// - `max_norm`: The maximum norm to clip the gradients to, preventing exploding gradients.
    pub fn train(
        &mut self,
        inputs: &Array2<f32>,
        expectations: &Array2<f32>,
        learning_rate: f32,
        max_norm: f32,
    ) -> Array2<f32> {
        assert!(
            learning_rate > 0.0,
            "Learning rate must be greater than zero."
        );

        assert_eq!(
            inputs.ncols(),
            expectations.ncols(),
            "Inputs and expectations must have the same number of samples."
        );

        let activations = self.forward(inputs);

        self.update(&activations, expectations, learning_rate, max_norm);

        last_activation(&activations)
    }

    /// Updates the weights and biases of the network using the computed gradients from backpropagation.
    /// # Arguments
    /// - `activations`: A vector of 2D arrays representing the activations of each layer.
    /// - `expectations`: A 2D array representing the expected outputs for the inputs.
    /// - `learning_rate`: The learning rate to apply during the update.
    /// - `max_norm`: The maximum norm to clip the gradients to, preventing exploding gradients.
    fn update(
        &mut self,
        activations: &Vec<Array2<f32>>,
        expectations: &Array2<f32>,
        learning_rate: f32,
        max_norm: f32,
    ) {
        let gradients = self.backward(activations, expectations);

        for (layer, mut layer_gradients) in self.layers.iter_mut().zip(gradients) {
            layer_gradients.clip(max_norm);
            layer.update(layer_gradients, learning_rate);
        }
    }

    /// Computes the gradients for each layer using backpropagation.
    /// # Arguments
    /// - `activations`: A vector of 2D arrays representing the activations of each layer.
    /// - `expectations`: A 2D array representing the expected outputs for the inputs.
    fn backward(
        &self,
        activations: &Vec<Array2<f32>>,
        expectations: &Array2<f32>,
    ) -> Vec<Gradients> {
        let m = expectations.ncols() as f32;

        let mut dz = activations.last().unwrap() - expectations;

        let mut gradients = Vec::with_capacity(activations.len() - 1);

        for i in (1..self.layers.len() + 1).rev() {
            let previous_activations = &activations[i - 1];
            let dw = dz.dot(&previous_activations.t()) / m;
            let db = dz.sum_axis(Axis(1)) / m;

            gradients.insert(0, Gradients { dw, db });

            if i > 1 {
                let next_layer = &self.layers[i - 1];
                dz = next_layer.weights.t().dot(&dz)
                    * next_layer
                        .activation
                        .derivative(previous_activations, expectations);
            }
        }

        gradients
    }
}
