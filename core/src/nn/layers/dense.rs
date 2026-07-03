use crate::activations::Activation;
use crate::gradients::LayerGradients;
use crate::layers::{BackwardPass, Layer, Parameter};
use crate::model::NeuronLayerSpec;
use ndarray::{Array1, Array2, ArrayD, ArrayView1, ArrayView2, Axis};
use ndarray_rand::rand::RngCore;
use std::any::Any;
use std::sync::Arc;

/// A fully connected layer: an affine map `weights · input + bias` followed by an
/// activation, applied to every sample in the batch.
#[derive(Clone, Debug)]
pub struct Dense {
    /// A 2D array where each row corresponds to a neuron and each column corresponds to an input feature.
    weights: Array2<f32>,
    /// A 1D array where each element is the bias for the corresponding neuron.
    biases: Array1<f32>,
    /// The activation function applied to the output of this layer.
    activation: Arc<dyn Activation>,
}

impl Dense {
    /// Assembles a `Dense` layer from explicit weights, biases, and activation.
    /// # Panics
    /// - When `weights` or `biases` are empty.
    /// - When the number of weight rows does not match the number of biases.
    /// # Arguments
    /// - `weights`: A `(neurons, inputs)` array, one row per neuron.
    /// - `biases`: A `(neurons)` array, one bias per neuron.
    /// - `activation`: The activation applied to this layer's output.
    pub fn new(weights: Array2<f32>, biases: Array1<f32>, activation: Arc<dyn Activation>) -> Self {
        assert!(
            weights.nrows() > 0 && weights.ncols() > 0,
            "Dense layer weights must be non-empty."
        );
        assert_eq!(
            weights.nrows(),
            biases.len(),
            "Dense layer needs one bias per neuron."
        );

        Dense {
            weights,
            biases,
            activation,
        }
    }

    /// Initializes a new `Dense` layer with random weights and biases drawn from `rng`.
    /// # Panics
    /// - When `spec.neurons` or `inputs` are less than or equal to zero.
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

        Self::new(weights, biases, spec.activation.clone())
    }

    /// Returns the specifications of this layer.
    pub fn spec(&self) -> NeuronLayerSpec {
        NeuronLayerSpec {
            neurons: self.weights.nrows(),
            activation: self.activation.clone(),
        }
    }

    /// This layer's weight matrix `(neurons, inputs)`.
    pub fn weights(&self) -> ArrayView2<'_, f32> {
        self.weights.view()
    }

    /// This layer's biases, one per neuron.
    pub fn biases(&self) -> ArrayView1<'_, f32> {
        self.biases.view()
    }

    /// The activation applied to this layer's output.
    pub fn activation(&self) -> &Arc<dyn Activation> {
        &self.activation
    }

    /// Mutable access to this layer's weight matrix.
    #[cfg(test)]
    pub(crate) fn weights_mut(&mut self) -> &mut Array2<f32> {
        &mut self.weights
    }

    /// Mutable access to this layer's biases.
    #[cfg(test)]
    pub(crate) fn biases_mut(&mut self) -> &mut Array1<f32> {
        &mut self.biases
    }
}

impl Layer for Dense {
    fn forward(&self, input: ArrayView2<f32>) -> Array2<f32> {
        assert_eq!(
            input.nrows(),
            self.weights.ncols(),
            "Input shape does not match weights shape."
        );

        // Broadcasting bias to match the shape of the output
        let broadcasted_biases: Array2<f32> = self.biases.view().insert_axis(Axis(1)).to_owned();

        self.activation
            .apply((self.weights.dot(&input) + &broadcasted_biases).view())
    }

    fn backward(
        &self,
        da: ArrayView2<f32>,
        input: ArrayView2<f32>,
        output: ArrayView2<f32>,
        compute_input_gradient: bool,
    ) -> BackwardPass {
        let m = input.ncols() as f32;

        // dz = dL/d(pre-activation); the activation's VJP turns dL/d(output) into it.
        let dz = self.activation.vjp(da, output);
        let dw = dz.dot(&input.t()) / m;
        let db = dz.sum_axis(Axis(1)) / m;
        // dL/d(input), the gradient handed back to the upstream layer.
        let input_gradient = compute_input_gradient.then(|| self.weights.t().dot(&dz));

        BackwardPass {
            gradients: LayerGradients(vec![dw.into_dyn(), db.into_dyn()]),
            input_gradient,
        }
    }

    fn parameters_mut(&mut self) -> Vec<Parameter<'_>> {
        vec![
            Parameter {
                value: self.weights.view_mut().into_dyn(),
                decays: true,
            },
            Parameter {
                value: self.biases.view_mut().into_dyn(),
                decays: false,
            },
        ]
    }

    fn input_size(&self) -> usize {
        self.weights.ncols()
    }

    fn output_size(&self) -> usize {
        self.weights.nrows()
    }

    fn is_finite(&self) -> bool {
        self.weights.iter().all(|v| v.is_finite()) && self.biases.iter().all(|v| v.is_finite())
    }

    fn kind(&self) -> &'static str {
        "dense"
    }

    fn named_tensors(&self) -> Vec<(String, ArrayD<f32>)> {
        vec![
            ("weights".to_string(), self.weights.clone().into_dyn()),
            ("biases".to_string(), self.biases.clone().into_dyn()),
        ]
    }

    fn activation_name(&self) -> Option<&str> {
        Some(self.activation.name())
    }

    fn weight_matrix(&self) -> Option<ArrayView2<'_, f32>> {
        Some(self.weights.view())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::{RELU, SIGMOID};
    use ndarray::array;

    #[test]
    fn accessors_report_dimensions_and_activation() {
        // 2 neurons, each taking 3 inputs.
        let layer = Dense {
            weights: Array2::zeros((2, 3)),
            biases: Array1::zeros(2),
            activation: RELU.clone(),
        };
        assert_eq!(layer.output_size(), 2);
        assert_eq!(layer.input_size(), 3);
        assert_eq!(layer.kind(), "dense");
        assert_eq!(layer.activation_name(), Some("relu"));
        assert_eq!(layer.weight_matrix().unwrap().dim(), (2, 3));

        let spec = layer.spec();
        assert_eq!(spec.neurons, 2);
        assert_eq!(spec.activation.name(), "relu");
    }

    #[test]
    fn parameters_are_weights_then_bias_with_decay_flags() {
        let mut layer = Dense {
            weights: array![[1.0, 2.0], [3.0, 4.0]],
            biases: array![5.0, 6.0],
            activation: RELU.clone(),
        };
        let params = layer.parameters_mut();
        assert_eq!(params.len(), 2);
        // Weights decay, the bias does not.
        assert!(params[0].decays);
        assert_eq!(params[0].value.shape(), &[2, 2]);
        assert!(!params[1].decays);
        assert_eq!(params[1].value.shape(), &[2]);
    }

    #[test]
    fn named_tensors_are_weights_and_biases() {
        let layer = Dense {
            weights: array![[1.0, 2.0]],
            biases: array![3.0],
            activation: RELU.clone(),
        };
        let tensors = layer.named_tensors();
        assert_eq!(tensors[0].0, "weights");
        assert_eq!(tensors[0].1, array![[1.0, 2.0]].into_dyn());
        assert_eq!(tensors[1].0, "biases");
        assert_eq!(tensors[1].1, array![3.0].into_dyn());
    }

    #[test]
    fn backward_gradients_match_numerical_approximation() {
        // Isolated single dense layer: with an upstream gradient of all ones, the loss
        // is L = sum(output), so finite differences of that sum recover the analytical
        // gradients. backward divides the parameter gradients by the sample count, so
        // the numerical estimate (which does not) is compared against `grad * m`.
        let layer = Dense {
            weights: array![[0.2, -0.4, 0.1], [0.5, 0.3, -0.2]],
            biases: array![0.05, -0.1],
            activation: SIGMOID.clone(),
        };
        let input = array![
            [0.5, -0.3, 0.8, 0.1],
            [0.2, 0.7, -0.5, 0.4],
            [-0.1, 0.6, 0.3, -0.4]
        ];
        let m = input.ncols() as f32;

        let output = layer.forward(input.view());
        let da = Array2::<f32>::ones(output.dim());
        let pass = layer.backward(da.view(), input.view(), output.view(), true);
        let grads = pass.gradients;
        let da_prev = pass.input_gradient.expect("input gradient requested");

        let loss = |layer: &Dense| layer.forward(input.view()).sum();
        let eps = 1e-3_f32;
        let tolerance = 5e-2_f32;
        let check = |analytical: f32, numerical: f32, label: &str| {
            let rel = (numerical - analytical).abs()
                / (numerical.abs().max(analytical.abs()).max(1e-2) + 1e-8);
            assert!(
                rel < tolerance,
                "{label}: analytical={analytical:.6}, numerical={numerical:.6}"
            );
        };

        // Weight and bias gradients: perturb each parameter, take the central difference
        // of the loss, and compare to `grad * m`.
        for i in 0..layer.output_size() {
            for j in 0..layer.input_size() {
                let mut plus = layer.clone();
                plus.weights[[i, j]] += eps;
                let mut minus = layer.clone();
                minus.weights[[i, j]] -= eps;
                let numerical = (loss(&plus) - loss(&minus)) / (2.0 * eps);
                check(grads[0][[i, j]] * m, numerical, &format!("dw[{i},{j}]"));
            }
            let mut plus = layer.clone();
            plus.biases[i] += eps;
            let mut minus = layer.clone();
            minus.biases[i] -= eps;
            let numerical = (loss(&plus) - loss(&minus)) / (2.0 * eps);
            check(grads[1][i] * m, numerical, &format!("db[{i}]"));
        }

        // Gradient with respect to the input: perturb each input entry.
        for k in 0..layer.input_size() {
            for s in 0..input.ncols() {
                let mut plus = input.clone();
                plus[[k, s]] += eps;
                let mut minus = input.clone();
                minus[[k, s]] -= eps;
                let numerical = (layer.forward(plus.view()).sum()
                    - layer.forward(minus.view()).sum())
                    / (2.0 * eps);
                check(da_prev[[k, s]], numerical, &format!("da_prev[{k},{s}]"));
            }
        }
    }
}
