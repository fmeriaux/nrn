use crate::activations::Activation;
use crate::affine::Affine;
use crate::gradients::LayerGradients;
use crate::layers::{BackwardPass, Layer, LayerConfigError, LayerKind, Parameter};
use crate::model::NeuronLayerSpec;
use ndarray::{Array1, Array2, ArrayD, ArrayView1, ArrayView2, ArrayViewD, Ix1, Ix2};
use ndarray_rand::rand::RngCore;
use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

/// A fully connected layer: an affine map `weights · input + bias` followed by an
/// activation, applied independently at every position of the batch. The leading axis holds
/// the features the weights map; every trailing axis (the samples, plus any spatial axes) is
/// a position the same shared weights act on. A plain `(features, samples)` batch is the
/// degenerate case with no spatial axes.
#[derive(Clone, Debug)]
pub struct Dense {
    /// The affine map: one row of weights per neuron, one bias per neuron.
    affine: Affine,
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
        Dense {
            affine: Affine::new(weights, biases),
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
            neurons: self.affine.outputs(),
            activation: self.activation.clone(),
        }
    }

    /// This layer's weight matrix `(neurons, inputs)`.
    pub fn weights(&self) -> ArrayView2<'_, f32> {
        self.affine.weights()
    }

    /// This layer's biases, one per neuron.
    pub fn biases(&self) -> ArrayView1<'_, f32> {
        self.affine.biases()
    }

    /// The activation applied to this layer's output.
    pub fn activation(&self) -> &Arc<dyn Activation> {
        &self.activation
    }

    /// Builds a `Dense` layer from its configuration and tensors.
    /// # Arguments
    /// - `config`: Carries the `"activation"` name.
    /// - `tensors`: Carries the `"weights"` (rank-2) and `"biases"` (rank-1) tensors.
    pub(super) fn from_config(
        config: &HashMap<String, String>,
        mut tensors: HashMap<String, ArrayD<f32>>,
    ) -> Result<Self, LayerConfigError> {
        let weights = super::take_tensor::<Ix2>(&mut tensors, "weights")?;
        let biases = super::take_tensor::<Ix1>(&mut tensors, "biases")?;
        let activation = super::config_activation(config)?;
        Ok(Dense::new(weights, biases, activation))
    }

    /// Mutable access to this layer's affine map.
    #[cfg(test)]
    pub(crate) fn affine_mut(&mut self) -> &mut Affine {
        &mut self.affine
    }
}

impl Layer for Dense {
    fn forward(&self, input: ArrayViewD<f32>) -> ArrayD<f32> {
        let in_features = self.affine.inputs();
        let out_neurons = self.affine.outputs();
        assert_eq!(
            input.shape()[0],
            in_features,
            "Input feature axis does not match weights shape."
        );

        // Fold every trailing (sample and any spatial) axis into a single column axis, so the
        // shared weights apply at every position through one rank-2 affine matmul, then unfold
        // the result back to the input's shape with the output feature count on the leading axis.
        let columns = input.len() / in_features;
        let flat = input
            .to_shape((in_features, columns))
            .expect("a (features, ..) batch folds to (features, columns)");
        let pre_activation = self.affine.forward(flat.view());
        let activated = self.activation.apply(pre_activation.view().into_dyn());

        let mut output_shape = input.shape().to_vec();
        output_shape[0] = out_neurons;
        activated
            .into_shape_with_order(output_shape)
            .expect("the matmul result folds back to the input shape with the output features")
    }

    fn backward(
        &self,
        da: ArrayViewD<f32>,
        input: ArrayViewD<f32>,
        output: ArrayViewD<f32>,
        compute_input_gradient: bool,
    ) -> BackwardPass {
        let in_features = self.affine.inputs();
        let out_neurons = self.affine.outputs();
        let input_shape = input.shape().to_vec();
        let columns = input.len() / in_features;

        // Fold the trailing axes into the column axis, mirroring forward: each position (spatial
        // cell × sample) is an independent column the activation VJP and the affine math act on.
        let da = da
            .to_shape((out_neurons, columns))
            .expect("da folds to (outputs, columns)");
        let output = output
            .to_shape((out_neurons, columns))
            .expect("output folds to (outputs, columns)");
        let input = input
            .to_shape((in_features, columns))
            .expect("input folds to (inputs, columns)");

        // dz = dL/d(pre-activation); the activation's VJP turns dL/d(output) into it. The shared
        // weights average over every column — positions and samples alike.
        let dz = self
            .activation
            .vjp(da.view().into_dyn(), output.view().into_dyn())
            .into_dimensionality::<Ix2>()
            .expect("the VJP preserves the rank-2 (outputs, columns) shape");
        let (dw, db, dinput) = self.affine.backward(
            dz.view(),
            input.view(),
            columns as f32,
            compute_input_gradient,
        );

        let input_gradient = dinput.map(|dinput| {
            dinput
                .into_shape_with_order(input_shape)
                .expect("the input gradient folds back to the input shape")
        });

        BackwardPass {
            gradients: LayerGradients(vec![dw.into_dyn(), db.into_dyn()]),
            input_gradient,
        }
    }

    fn parameters_mut(&mut self) -> Vec<Parameter<'_>> {
        let (weights, biases) = self.affine.parameters_mut();
        vec![
            Parameter {
                value: weights.view_mut().into_dyn(),
                decays: true,
            },
            Parameter {
                value: biases.view_mut().into_dyn(),
                decays: false,
            },
        ]
    }

    fn input_shape(&self) -> Vec<usize> {
        vec![self.affine.inputs()]
    }

    fn output_shape(&self) -> Vec<usize> {
        vec![self.affine.outputs()]
    }

    fn is_finite(&self) -> bool {
        self.affine.is_finite()
    }

    fn kind(&self) -> LayerKind {
        LayerKind::Dense
    }

    fn config(&self) -> Vec<(String, String)> {
        vec![("activation".to_string(), self.activation.name().to_string())]
    }

    fn named_tensors(&self) -> Vec<(String, ArrayD<f32>)> {
        vec![
            (
                "weights".to_string(),
                self.affine.weights().to_owned().into_dyn(),
            ),
            (
                "biases".to_string(),
                self.affine.biases().to_owned().into_dyn(),
            ),
        ]
    }

    fn activation_name(&self) -> Option<&str> {
        Some(self.activation.name())
    }

    fn weight_matrix(&self) -> Option<ArrayView2<'_, f32>> {
        Some(self.affine.weights())
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
    use ndarray::{Axis, array};

    #[test]
    fn accessors_report_dimensions_and_activation() {
        // 2 neurons, each taking 3 inputs.
        let layer = Dense::new(Array2::zeros((2, 3)), Array1::zeros(2), RELU.clone());
        assert_eq!(layer.output_size(), 2);
        assert_eq!(layer.input_size(), 3);
        assert_eq!(layer.kind(), LayerKind::Dense);
        assert_eq!(layer.activation_name(), Some("relu"));
        assert_eq!(layer.weight_matrix().unwrap().dim(), (2, 3));

        let spec = layer.spec();
        assert_eq!(spec.neurons, 2);
        assert_eq!(spec.activation.name(), "relu");
    }

    #[test]
    fn parameters_are_weights_then_bias_with_decay_flags() {
        let mut layer = Dense::new(
            array![[1.0, 2.0], [3.0, 4.0]],
            array![5.0, 6.0],
            RELU.clone(),
        );
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
        let layer = Dense::new(array![[1.0, 2.0]], array![3.0], RELU.clone());
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
        let layer = Dense::new(
            array![[0.2, -0.4, 0.1], [0.5, 0.3, -0.2]],
            array![0.05, -0.1],
            SIGMOID.clone(),
        );
        let input = array![
            [0.5, -0.3, 0.8, 0.1],
            [0.2, 0.7, -0.5, 0.4],
            [-0.1, 0.6, 0.3, -0.4]
        ];
        let m = input.ncols() as f32;

        let output = layer.forward(input.view().into_dyn());
        let da = ArrayD::<f32>::ones(output.raw_dim());
        let pass = layer.backward(da.view(), input.view().into_dyn(), output.view(), true);
        let grads = pass.gradients;
        let da_prev = pass.input_gradient.expect("input gradient requested");

        let loss = |layer: &Dense| layer.forward(input.view().into_dyn()).sum();
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
                plus.affine.weights_mut()[[i, j]] += eps;
                let mut minus = layer.clone();
                minus.affine.weights_mut()[[i, j]] -= eps;
                let numerical = (loss(&plus) - loss(&minus)) / (2.0 * eps);
                check(grads[0][[i, j]] * m, numerical, &format!("dw[{i},{j}]"));
            }
            let mut plus = layer.clone();
            plus.affine.biases_mut()[i] += eps;
            let mut minus = layer.clone();
            minus.affine.biases_mut()[i] -= eps;
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
                let numerical = (layer.forward(plus.view().into_dyn()).sum()
                    - layer.forward(minus.view().into_dyn()).sum())
                    / (2.0 * eps);
                check(da_prev[[k, s]], numerical, &format!("da_prev[{k},{s}]"));
            }
        }
    }

    #[test]
    fn forward_matches_the_2d_math_and_keeps_rank_two() {
        let layer = Dense::new(
            array![[0.2, -0.4, 0.1], [0.5, 0.3, -0.2]],
            array![0.05, -0.1],
            SIGMOID.clone(),
        );
        let input = array![
            [0.5, -0.3, 0.8, 0.1],
            [0.2, 0.7, -0.5, 0.4],
            [-0.1, 0.6, 0.3, -0.4]
        ];

        // Reference computed with the column-major math the layer used before the N-D switch.
        let biases = layer.biases().insert_axis(Axis(1)).to_owned();
        let reference = layer
            .activation()
            .apply((layer.weights().dot(&input) + &biases).view().into_dyn());

        let output = layer.forward(input.view().into_dyn());
        assert_eq!(output.ndim(), 2);
        assert_eq!(output.shape(), &[layer.output_size(), input.ncols()]);
        // Bit-exact: the N-D pipe only reinterprets the dimension type, it does not touch values.
        assert_eq!(output, reference.into_dyn());

        // The input gradient round-trips back to rank 2 with the expected shape.
        let da = ArrayD::<f32>::ones(output.raw_dim());
        let pass = layer.backward(da.view(), input.view().into_dyn(), output.view(), true);
        let input_gradient = pass.input_gradient.expect("input gradient requested");
        assert_eq!(input_gradient.ndim(), 2);
        assert_eq!(input_gradient.shape(), &[layer.input_size(), input.ncols()]);
    }

    #[test]
    fn forward_applies_the_same_weights_at_every_spatial_position() {
        use ndarray::{Array, IxDyn, s};

        // 2 inputs -> 3 outputs, applied per position of a (2, H=2, W=2, samples=2) batch.
        let layer = Dense::new(
            array![[0.2, -0.4], [0.5, 0.3], [-0.1, 0.6]],
            array![0.05, -0.1, 0.2],
            SIGMOID.clone(),
        );
        let input = Array::from_shape_fn(IxDyn(&[2, 2, 2, 2]), |idx| {
            (idx[0] + idx[1] + idx[2] + idx[3]) as f32 * 0.1 - 0.3
        });

        let output = layer.forward(input.view());
        assert_eq!(output.shape(), &[3, 2, 2, 2]);

        // Every (h, w, s) position is an independent affine + activation on that position's
        // feature column with the same shared weights, so it must match the rank-2 layer run
        // on that single column.
        for h in 0..2 {
            for w in 0..2 {
                for s in 0..2 {
                    let column = input.slice(s![.., h, w, s]).insert_axis(Axis(1)).to_owned();
                    let expected = layer.forward(column.view().into_dyn());
                    let got = output.slice(s![.., h, w, s]);
                    let want = expected.index_axis(Axis(1), 0);
                    for (g, w) in got.iter().zip(want.iter()) {
                        assert!((g - w).abs() < 1e-6, "mismatch at ({h},{w},{s})");
                    }
                }
            }
        }
    }

    #[test]
    fn backward_gradients_on_a_spatial_batch_match_numerical_approximation() {
        use ndarray::{Array, IxDyn};

        // The trailing (spatial + sample) axes fold into the columns the shared weights average
        // over, so the parameter gradients are the mean over positions × samples: m = columns.
        let layer = Dense::new(
            array![[0.2, -0.4], [0.5, 0.3], [-0.1, 0.6]],
            array![0.05, -0.1, 0.2],
            SIGMOID.clone(),
        );
        let input = Array::from_shape_fn(IxDyn(&[2, 2, 2, 3]), |idx| {
            ((idx[0] * 7 + idx[1] * 3 + idx[2] * 2 + idx[3]) % 11) as f32 * 0.1 - 0.5
        });
        let in_features = layer.input_size();
        let m = (input.len() / in_features) as f32; // columns = positions × samples

        let output = layer.forward(input.view());
        let da = ArrayD::<f32>::ones(output.raw_dim());
        let pass = layer.backward(da.view(), input.view(), output.view(), true);
        let grads = pass.gradients;
        let da_prev = pass.input_gradient.expect("input gradient requested");
        assert_eq!(da_prev.shape(), input.shape());

        // With an upstream gradient of all ones, the loss is L = sum(output); finite differences
        // of that sum recover the analytical gradients. backward divides by m, so the numerical
        // estimate (which does not) is compared against `grad * m`.
        let loss = |layer: &Dense| layer.forward(input.view()).sum();
        // A coarser step than the rank-2 case: L sums over more columns, so f32 cancellation
        // dominates a smaller one.
        let eps = 1e-2_f32;
        let tolerance = 5e-2_f32;
        let check = |analytical: f32, numerical: f32, label: &str| {
            let rel = (numerical - analytical).abs()
                / (numerical.abs().max(analytical.abs()).max(1e-2) + 1e-8);
            assert!(
                rel < tolerance,
                "{label}: analytical={analytical:.6}, numerical={numerical:.6}"
            );
        };

        for o in 0..layer.output_size() {
            for j in 0..layer.input_size() {
                let mut plus = layer.clone();
                plus.affine.weights_mut()[[o, j]] += eps;
                let mut minus = layer.clone();
                minus.affine.weights_mut()[[o, j]] -= eps;
                let numerical = (loss(&plus) - loss(&minus)) / (2.0 * eps);
                check(grads[0][[o, j]] * m, numerical, &format!("dw[{o},{j}]"));
            }
            let mut plus = layer.clone();
            plus.affine.biases_mut()[o] += eps;
            let mut minus = layer.clone();
            minus.affine.biases_mut()[o] -= eps;
            let numerical = (loss(&plus) - loss(&minus)) / (2.0 * eps);
            check(grads[1][o] * m, numerical, &format!("db[{o}]"));
        }
    }
}
