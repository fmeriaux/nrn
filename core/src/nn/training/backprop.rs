use crate::data::ModelDataset;
use crate::gradients::{GradientClipping, LayerGradients};
use crate::loss_functions::LossFunction;
use crate::model::{InputShapeMismatch, NeuralNetwork, last_activation};
use crate::optimizers::Optimizer;
use crate::schedulers::Scheduler;
use ndarray::{ArrayD, ArrayView2, Ix2};
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand::rngs::StdRng;
use std::sync::Arc;

/// 2⁶⁴⁄φ rounded to an odd integer — the golden-ratio (splitmix64) multiplier. Mixed
/// with the epoch number, it scatters consecutive epochs across the whole seed space,
/// so each epoch's mini-batch shuffle differs from its neighbours'.
const GOLDEN_RATIO: u64 = 0x9E37_79B9_7F4A_7C15;

/// Mini-batch SGD configuration for a single epoch. Bundling the batch `size` with the
/// `rng` that shuffles it makes the two inseparable: passing `None` to
/// [`NeuralNetwork::train`] is full-batch (no size, no shuffle), while `Some` always
/// carries both. This rules out the illegal "size without a generator" (or vice versa)
/// states a pair of separate parameters would allow. Built via [`MiniBatch::new`].
pub struct MiniBatch {
    size: usize,
    rng: StdRng,
}

impl MiniBatch {
    /// Builds the mini-batch config for one epoch: batches of `size`, shuffled by a
    /// generator seeded from a mix of the run's `seed` and the `epoch`. Deriving it
    /// from `(seed, epoch)` keeps the shuffle a pure function of the two — reproducible
    /// and resume-safe, with no RNG state to persist across a checkpoint.
    pub fn new(size: usize, seed: u64, epoch: usize) -> Self {
        let rng = StdRng::seed_from_u64(seed ^ (epoch as u64).wrapping_mul(GOLDEN_RATIO));
        Self { size, rng }
    }
}

impl NeuralNetwork {
    /// Trains the network for one epoch using the provided dataset.
    /// # Errors
    /// [`InputShapeMismatch`] when the dataset's feature rows do not match [`Self::input_size`].
    /// # Arguments
    /// - `dataset`: The dataset containing inputs and targets for training.
    /// - `loss_function`: The loss function to use for computing the loss and its gradient.
    /// - `optimizer`: The optimizer to use for updating the weights and biases.
    /// - `scheduler`: The learning rate scheduler, stepped once per epoch.
    /// - `clipping`: The gradient clipping strategy to apply during training.
    /// - `mini_batch`: `Some(MiniBatch)` performs mini-batch SGD, shuffling the dataset
    ///   each epoch; `None` performs full-batch gradient descent.
    pub fn train(
        &mut self,
        dataset: &ModelDataset,
        loss_function: &Arc<dyn LossFunction>,
        optimizer: &mut dyn Optimizer,
        scheduler: &mut dyn Scheduler,
        clipping: &GradientClipping,
        mini_batch: Option<MiniBatch>,
    ) -> Result<(), InputShapeMismatch> {
        // Scheduler steps once per epoch regardless of batch size
        let lr = scheduler.step();
        optimizer.set_learning_rate(lr);

        match mini_batch {
            None => {
                let activations = self.forward(dataset.inputs().view())?;
                self.update_parameters(
                    &activations,
                    dataset.targets().view(),
                    loss_function,
                    optimizer,
                    clipping,
                );
            }
            Some(MiniBatch { size, mut rng }) => {
                for batch in dataset.batches(size, &mut rng) {
                    let activations = self.forward(batch.inputs().view())?;
                    self.update_parameters(
                        &activations,
                        batch.targets().view(),
                        loss_function,
                        optimizer,
                        clipping,
                    );
                }
            }
        }

        Ok(())
    }

    /// Updates the weights and biases of the network using the computed gradients from backpropagation.
    /// # Arguments
    /// - `activations`: A vector of 2D arrays representing the activations of each layer.
    /// - `targets`: A 2D array representing the expected outputs for the inputs.
    /// - `loss_function`: The loss function to use for computing the loss and its gradient.
    /// - `optimizer`: The optimizer to use for updating the weights and biases.
    /// - `clipping`: The gradient clipping strategy to apply during training.
    fn update_parameters(
        &mut self,
        activations: &[ArrayD<f32>],
        targets: ArrayView2<f32>,
        loss_function: &Arc<dyn LossFunction>,
        optimizer: &mut dyn Optimizer,
        clipping: &GradientClipping,
    ) {
        let gradients = self.backward(activations, targets, loss_function);

        for (layer_index, (layer, mut layer_gradients)) in
            self.layers_mut().iter_mut().zip(gradients).enumerate()
        {
            layer_gradients.clip_by(clipping);

            optimizer.update_layer(layer_index, layer.as_mut(), &layer_gradients);
        }

        optimizer.step();
    }

    /// Computes the gradients for each layer using backpropagation.
    ///
    /// Each layer turns the gradient of the loss with respect to its output into its own
    /// parameter gradients and the gradient with respect to its input, which becomes the
    /// upstream layer's incoming gradient.
    /// # Arguments
    /// - `activations`: A vector of 2D arrays representing the activations of each layer.
    /// - `targets`: A 2D array representing the true labels for the inputs.
    fn backward(
        &self,
        activations: &[ArrayD<f32>],
        targets: ArrayView2<f32>,
        loss_function: &Arc<dyn LossFunction>,
    ) -> Vec<LayerGradients> {
        let last_act = last_activation(activations)
            .into_dimensionality::<Ix2>()
            .expect("the output layer produces rank-2 (classes, samples) activations");

        let mut da = loss_function.gradient(last_act.view(), targets).into_dyn();

        let layers = self.layers();
        let mut gradients = Vec::with_capacity(layers.len());

        for i in (0..layers.len()).rev() {
            let input = activations[i].view();
            let output = activations[i + 1].view();
            // The input layer (i == 0) has no upstream layer to receive an input gradient.
            let pass = layers[i].backward(da.view(), input, output, i > 0);
            gradients.push(pass.gradients);
            if let Some(input_gradient) = pass.input_gradient {
                da = input_gradient;
            }
        }

        // Collected from the output layer back to the input; restore input-to-output order.
        gradients.reverse();

        gradients
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::SIGMOID;
    use crate::layers::Dense;
    use crate::learning_rate::LearningRate;
    use crate::loss_functions::CROSS_ENTROPY_LOSS;
    use crate::model::{NeuralNetwork, NeuronLayerSpec};
    use crate::optimizers::StochasticGradientDescent;
    use crate::schedulers::ConstantScheduler;
    use crate::weight_decay::WeightDecay;
    use ndarray::{Array1, Array2, array};

    /// Downcasts a network's layer to the concrete [`Dense`] to read its weights and biases.
    fn dense(model: &NeuralNetwork, index: usize) -> &Dense {
        model.layers()[index]
            .as_any()
            .downcast_ref::<Dense>()
            .unwrap()
    }

    /// Downcasts a network's layer to the concrete [`Dense`] to set or perturb its
    /// weights and biases.
    fn dense_mut(model: &mut NeuralNetwork, index: usize) -> &mut Dense {
        model.layers_mut()[index]
            .as_any_mut()
            .downcast_mut::<Dense>()
            .unwrap()
    }

    fn compute_loss(
        model: &NeuralNetwork,
        inputs: &Array2<f32>,
        targets: &Array2<f32>,
        loss_fn: &Arc<dyn LossFunction>,
    ) -> f32 {
        let pred = model.output(inputs.view()).unwrap();
        loss_fn.compute(pred.view(), targets.view())
    }

    #[test]
    fn backprop_gradients_match_numerical_approximation() {
        // Network: 2 inputs -> 2 hidden (sigmoid) -> 1 output (linear logits, binary)
        // Using sigmoid in the hidden layer to avoid relu's non-differentiable point at 0
        let specs = NeuronLayerSpec::network_for(vec![2], &*SIGMOID, 2);
        let mut model = NeuralNetwork::initialization(2, &specs, 0);

        // Fixed weights for reproducibility
        *dense_mut(&mut model, 0).affine_mut().weights_mut() = array![[0.1, -0.2], [0.3, 0.1]];
        *dense_mut(&mut model, 0).affine_mut().biases_mut() = Array1::from_vec(vec![0.05, -0.05]);
        *dense_mut(&mut model, 1).affine_mut().weights_mut() = array![[0.4, -0.1]];
        *dense_mut(&mut model, 1).affine_mut().biases_mut() = Array1::from_vec(vec![0.1]);

        let inputs = array![[0.5, -0.3, 0.8], [0.2, 0.7, -0.5]]; // (2 features, 3 samples)
        let targets = array![[1.0, 0.0, 1.0]]; // (1 output, 3 samples)

        let loss_fn: Arc<dyn LossFunction> = CROSS_ENTROPY_LOSS.clone();
        let eps = 1e-4_f32;
        // f32 finite differences amplify rounding errors for small gradients;
        // 5% tolerates f32 noise while still catching real bugs (which cause >20% error)
        let tolerance = 5e-2_f32;

        let activations = model.forward(inputs.view()).unwrap();
        let analytical_grads = model.backward(&activations, targets.view(), &loss_fn);

        // Floor the denominator (~largest gradient scale) to avoid amplifying
        // f32 noise when the gradient is small
        let check = |analytical: f32, numerical: f32, label: &str| {
            let rel_diff = (numerical - analytical).abs()
                / (numerical.abs().max(analytical.abs()).max(1e-2) + 1e-8);
            assert!(
                rel_diff < tolerance,
                "{label}: analytical={analytical:.6}, numerical={numerical:.6}, rel_diff={rel_diff:.6}"
            );
        };

        for (layer_idx, layer_grads) in analytical_grads.iter().enumerate() {
            let (rows, cols) = dense(&model, layer_idx).weights().dim();
            for i in 0..rows {
                for j in 0..cols {
                    let mut m_plus = model.clone();
                    dense_mut(&mut m_plus, layer_idx).affine_mut().weights_mut()[[i, j]] += eps;
                    let mut m_minus = model.clone();
                    dense_mut(&mut m_minus, layer_idx)
                        .affine_mut()
                        .weights_mut()[[i, j]] -= eps;
                    let numerical = (compute_loss(&m_plus, &inputs, &targets, &loss_fn)
                        - compute_loss(&m_minus, &inputs, &targets, &loss_fn))
                        / (2.0 * eps);
                    check(
                        layer_grads[0][[i, j]],
                        numerical,
                        &format!("W[{layer_idx}][{i},{j}]"),
                    );
                }
            }

            for i in 0..dense(&model, layer_idx).biases().len() {
                let mut m_plus = model.clone();
                dense_mut(&mut m_plus, layer_idx).affine_mut().biases_mut()[i] += eps;
                let mut m_minus = model.clone();
                dense_mut(&mut m_minus, layer_idx).affine_mut().biases_mut()[i] -= eps;
                let numerical = (compute_loss(&m_plus, &inputs, &targets, &loss_fn)
                    - compute_loss(&m_minus, &inputs, &targets, &loss_fn))
                    / (2.0 * eps);
                check(
                    layer_grads[1][[i]],
                    numerical,
                    &format!("b[{layer_idx}][{i}]"),
                );
            }
        }
    }

    #[test]
    fn backprop_gradients_match_numerical_approximation_softmax_output() {
        // Single-layer network (inputs → linear logits), no hidden layers.
        // Checks the with-logits gradient (p − y) against finite differences; a single output
        // keeps gradients O(0.1), well within f32 precision.
        let specs = NeuronLayerSpec::network_for(vec![], &*SIGMOID, 3);
        let mut model = NeuralNetwork::initialization(2, &specs, 0);

        *dense_mut(&mut model, 0).affine_mut().weights_mut() =
            array![[0.5, -0.3], [0.2, 0.8], [-0.4, 0.1]];
        *dense_mut(&mut model, 0).affine_mut().biases_mut() =
            Array1::from_vec(vec![0.1, -0.2, 0.1]);

        // 4 samples across 3 classes
        let inputs = array![[1.0, -1.0, 0.5, -0.5], [0.5, 0.5, -0.5, -0.5]];
        let targets = array![
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ];

        let loss_fn: Arc<dyn LossFunction> = CROSS_ENTROPY_LOSS.clone();
        let eps = 1e-4_f32;
        let tolerance = 5e-2_f32;

        let activations = model.forward(inputs.view()).unwrap();
        let analytical_grads = model.backward(&activations, targets.view(), &loss_fn);

        let check = |analytical: f32, numerical: f32, label: &str| {
            let rel_diff = (numerical - analytical).abs()
                / (numerical.abs().max(analytical.abs()).max(1e-2) + 1e-8);
            assert!(
                rel_diff < tolerance,
                "{label}: analytical={analytical:.6}, numerical={numerical:.6}, rel_diff={rel_diff:.6}"
            );
        };

        for (layer_idx, layer_grads) in analytical_grads.iter().enumerate() {
            let (rows, cols) = dense(&model, layer_idx).weights().dim();
            for i in 0..rows {
                for j in 0..cols {
                    let mut m_plus = model.clone();
                    dense_mut(&mut m_plus, layer_idx).affine_mut().weights_mut()[[i, j]] += eps;
                    let mut m_minus = model.clone();
                    dense_mut(&mut m_minus, layer_idx)
                        .affine_mut()
                        .weights_mut()[[i, j]] -= eps;
                    let numerical = (compute_loss(&m_plus, &inputs, &targets, &loss_fn)
                        - compute_loss(&m_minus, &inputs, &targets, &loss_fn))
                        / (2.0 * eps);
                    check(
                        layer_grads[0][[i, j]],
                        numerical,
                        &format!("W[{layer_idx}][{i},{j}]"),
                    );
                }
            }
            for i in 0..dense(&model, layer_idx).biases().len() {
                let mut m_plus = model.clone();
                dense_mut(&mut m_plus, layer_idx).affine_mut().biases_mut()[i] += eps;
                let mut m_minus = model.clone();
                dense_mut(&mut m_minus, layer_idx).affine_mut().biases_mut()[i] -= eps;
                let numerical = (compute_loss(&m_plus, &inputs, &targets, &loss_fn)
                    - compute_loss(&m_minus, &inputs, &targets, &loss_fn))
                    / (2.0 * eps);
                check(
                    layer_grads[1][[i]],
                    numerical,
                    &format!("b[{layer_idx}][{i}]"),
                );
            }
        }
    }

    #[test]
    fn backward_is_finite_when_binary_logit_saturates() {
        // Regression: an extreme output logit must still yield finite gradients.
        let specs = NeuronLayerSpec::network_for(vec![], &*SIGMOID, 2);
        let mut model = NeuralNetwork::initialization(2, &specs, 0);

        // Large weights drive the single output logit to ±200.
        *dense_mut(&mut model, 0).affine_mut().weights_mut() = array![[100.0, 100.0]];
        *dense_mut(&mut model, 0).affine_mut().biases_mut() = Array1::from_vec(vec![0.0]);

        let inputs = array![[1.0, -1.0], [1.0, -1.0]]; // logits +200 / -200
        let targets = array![[1.0, 0.0]];

        let loss_fn: Arc<dyn LossFunction> = CROSS_ENTROPY_LOSS.clone();
        let activations = model.forward(inputs.view()).unwrap();
        // The linear output emits logits.
        assert_eq!(
            last_activation(&activations),
            array![[200.0, -200.0]].into_dyn()
        );

        let grads = model.backward(&activations, targets.view(), &loss_fn);
        for g in &grads {
            assert!(
                g.iter()
                    .flat_map(|tensor| tensor.iter())
                    .all(|v| v.is_finite()),
                "saturated logit produced non-finite gradients: {:?}",
                g.0
            );
        }
    }

    #[test]
    fn backward_feeds_raw_output_to_the_loss_gradient() {
        // backward() hands the output layer's raw logits to the loss's gradient, unclamped.
        use std::sync::Mutex;

        struct SpyLoss {
            seen: Mutex<Option<Array2<f32>>>,
        }

        impl LossFunction for SpyLoss {
            fn name(&self) -> &'static str {
                "Spy"
            }
            fn compute(&self, _predictions: ArrayView2<f32>, _targets: ArrayView2<f32>) -> f32 {
                0.0
            }
            fn gradient(
                &self,
                predictions: ArrayView2<f32>,
                _targets: ArrayView2<f32>,
            ) -> Array2<f32> {
                *self.seen.lock().unwrap() = Some(predictions.to_owned());
                Array2::zeros(predictions.raw_dim())
            }
        }

        let specs = NeuronLayerSpec::network_for(vec![], &*SIGMOID, 2);
        let mut model = NeuralNetwork::initialization(2, &specs, 0);
        *dense_mut(&mut model, 0).affine_mut().weights_mut() = array![[100.0, 100.0]];
        *dense_mut(&mut model, 0).affine_mut().biases_mut() = Array1::from_vec(vec![0.0]);
        let inputs = array![[1.0, -1.0], [1.0, -1.0]]; // logits +200 / -200
        let targets = array![[1.0, 0.0]];

        let activations = model.forward(inputs.view()).unwrap();
        assert_eq!(
            last_activation(&activations),
            array![[200.0, -200.0]].into_dyn()
        );

        let spy = Arc::new(SpyLoss {
            seen: Mutex::new(None),
        });
        let loss_fn: Arc<dyn LossFunction> = spy.clone();
        let _ = model.backward(&activations, targets.view(), &loss_fn);

        let seen = spy
            .seen
            .lock()
            .unwrap()
            .clone()
            .expect("gradient was called");
        assert_eq!(
            seen,
            array![[200.0, -200.0]],
            "backward must feed the raw output logits to gradient(), not a clamped copy"
        );
    }

    #[test]
    fn flatten_layer_threads_through_training_without_parameters() {
        use crate::layers::Flatten;

        // A parameterless Flatten as the first layer (rank-2 in, rank-2 out here, so a
        // no-op reshape) must thread through forward, backprop, and the optimizer: the
        // optimizer skips its empty parameter list while the Dense head still learns.
        let head = Dense::initialization(
            4,
            &NeuronLayerSpec::output_for(2),
            &mut StdRng::seed_from_u64(1),
        );
        let mut model = NeuralNetwork::new(vec![Box::new(Flatten::new(vec![4])), Box::new(head)]);

        let before = dense(&model, 1).weights().to_owned();

        let inputs = Array2::from_shape_fn((4, 12), |(r, c)| ((r + c) as f32).cos());
        let mut targets = Array2::zeros((1, 12));
        for c in 0..12 {
            targets[[0, c]] = (c % 2) as f32;
        }
        let dataset = ModelDataset::new(inputs, targets);

        let loss: Arc<dyn LossFunction> = CROSS_ENTROPY_LOSS.clone();
        let lr = LearningRate::new(0.1).unwrap();
        let mut optimizer = StochasticGradientDescent::new(lr, WeightDecay::ZERO);
        let mut scheduler = ConstantScheduler::new(lr);

        model
            .train(
                &dataset,
                &loss,
                &mut optimizer as &mut dyn Optimizer,
                &mut scheduler as &mut dyn Scheduler,
                &GradientClipping::None,
                None,
            )
            .unwrap();

        let after = dense(&model, 1).weights().to_owned();
        assert_ne!(before, after, "the Dense head should have learned");
    }

    #[test]
    fn training_a_convolutional_network_decreases_the_loss() {
        use crate::activations::Activation;
        use crate::layers::{Conv2d, Flatten};
        use ndarray::Array4;

        // Conv2d(1×4×4 → 2×2×2) → Flatten(8) → Dense(8 → 1, sigmoid binary): a spatial
        // stack trained end-to-end from a rank-4 samples-last dataset.
        let mut rng = StdRng::seed_from_u64(0);
        let sigmoid: Arc<dyn Activation> = SIGMOID.clone();
        let conv = Conv2d::initialization((1, 4, 4), 2, (3, 3), 1, 0, sigmoid, &mut rng);
        let flatten = Flatten::new(vec![2, 2, 2]);
        let head = Dense::initialization(8, &NeuronLayerSpec::output_for(2), &mut rng);
        let mut model = NeuralNetwork::single(conv)
            .with_layer(flatten)
            .with_layer(head);

        // Two linearly separable clusters: the first half bright (label 1), the rest
        // dark (label 0), so the loss has a clear direction to fall in.
        let n = 16;
        let inputs = Array4::from_shape_fn((1, 4, 4, n), |(_, h, w, s)| {
            let sign = if s < n / 2 { 1.0 } else { -1.0 };
            sign * (0.5 + 0.1 * ((h + w) as f32).cos())
        });
        let targets = Array2::from_shape_fn((1, n), |(_, s)| if s < n / 2 { 1.0 } else { 0.0 });
        let dataset = ModelDataset::new(inputs.clone(), targets.clone());

        let loss: Arc<dyn LossFunction> = CROSS_ENTROPY_LOSS.clone();
        let loss_now = |model: &NeuralNetwork| {
            let pred = model.output(inputs.view()).unwrap();
            loss.compute(pred.view(), targets.view())
        };

        let initial = loss_now(&model);

        let lr = LearningRate::new(0.5).unwrap();
        let mut optimizer = StochasticGradientDescent::new(lr, WeightDecay::ZERO);
        let mut scheduler = ConstantScheduler::new(lr);
        for _ in 0..100 {
            model
                .train(
                    &dataset,
                    &loss,
                    &mut optimizer as &mut dyn Optimizer,
                    &mut scheduler as &mut dyn Scheduler,
                    &GradientClipping::None,
                    None,
                )
                .unwrap();
        }

        let final_loss = loss_now(&model);
        assert!(
            final_loss < initial,
            "training a CNN should decrease the loss: initial={initial:.6}, final={final_loss:.6}"
        );
    }

    #[test]
    fn training_is_deterministic_for_a_fixed_seed() {
        // Two runs with the same seed and data produce bit-identical weights.
        fn run() -> NeuralNetwork {
            let specs = NeuronLayerSpec::network_for(vec![8, 4], &*SIGMOID, 3);
            let mut model = NeuralNetwork::initialization(5, &specs, 7);

            let inputs = Array2::from_shape_fn((5, 40), |(r, c)| ((r + c) as f32).sin());
            let mut targets = Array2::zeros((3, 40));
            for c in 0..40 {
                targets[[c % 3, c]] = 1.0;
            }
            let dataset = ModelDataset::new(inputs, targets);

            let loss: Arc<dyn LossFunction> = CROSS_ENTROPY_LOSS.clone();
            let lr = LearningRate::new(0.05).unwrap();
            let mut optimizer = StochasticGradientDescent::new(lr, WeightDecay::ZERO);
            let mut scheduler = ConstantScheduler::new(lr);

            for epoch in 0..5 {
                model
                    .train(
                        &dataset,
                        &loss,
                        &mut optimizer as &mut dyn Optimizer,
                        &mut scheduler as &mut dyn Scheduler,
                        &GradientClipping::None,
                        Some(MiniBatch::new(16, 7, epoch)),
                    )
                    .unwrap();
            }
            model
        }

        let a = run();
        let b = run();
        for i in 0..a.layers().len() {
            assert_eq!(
                dense(&a, i).weights(),
                dense(&b, i).weights(),
                "weights diverged between identical runs"
            );
            assert_eq!(
                dense(&a, i).biases(),
                dense(&b, i).biases(),
                "biases diverged between identical runs"
            );
        }
    }
}
