use crate::data::ModelDataset;
use crate::gradients::{GradientClipping, Gradients};
use crate::loss_functions::LossFunction;
use crate::model::{NeuralNetwork, last_activation};
use crate::optimizers::Optimizer;
use crate::schedulers::Scheduler;
use ndarray::{Array2, ArrayView2, Axis};
use ndarray_rand::rand;
use std::sync::Arc;

impl NeuralNetwork {
    /// Trains the network for one epoch using the provided dataset.
    /// # Panics
    /// - When the number of columns in `inputs` does not match the number of columns in `targets`.
    /// # Arguments
    /// - `dataset`: The dataset containing inputs and targets for training.
    /// - `loss_function`: The loss function to use for computing the loss and its gradient.
    /// - `optimizer`: The optimizer to use for updating the weights and biases.
    /// - `scheduler`: The learning rate scheduler, stepped once per epoch.
    /// - `clipping`: The gradient clipping strategy to apply during training.
    /// - `batch_size`: If `Some(n)`, performs mini-batch SGD with batches of size `n`, shuffling
    ///   the dataset each epoch. If `None`, performs full-batch gradient descent.
    pub fn train(
        &mut self,
        dataset: &ModelDataset,
        loss_function: &Arc<dyn LossFunction>,
        optimizer: &mut dyn Optimizer,
        scheduler: &mut dyn Scheduler,
        clipping: &GradientClipping,
        batch_size: Option<usize>,
    ) {
        let n_samples = dataset.inputs.ncols();
        assert_eq!(
            n_samples,
            dataset.targets.ncols(),
            "Inputs and targets must have the same number of samples."
        );

        // Scheduler steps once per epoch regardless of batch size
        let lr = scheduler.step();
        optimizer.set_learning_rate(lr);

        match batch_size {
            None => {
                let activations = self.forward(dataset.inputs.view());
                self.update_parameters(
                    &activations,
                    dataset.targets.view(),
                    loss_function,
                    optimizer,
                    clipping,
                );
            }
            Some(size) => {
                for batch in dataset.batches(size, &mut rand::rng()) {
                    let activations = self.forward(batch.inputs.view());
                    self.update_parameters(
                        &activations,
                        batch.targets.view(),
                        loss_function,
                        optimizer,
                        clipping,
                    );
                }
            }
        }
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
        activations: &[Array2<f32>],
        targets: ArrayView2<f32>,
        loss_function: &Arc<dyn LossFunction>,
        optimizer: &mut dyn Optimizer,
        clipping: &GradientClipping,
    ) {
        let gradients = self.backward(activations, targets, loss_function);

        for (layer_index, (layer, mut layer_gradients)) in
            self.layers.iter_mut().zip(gradients).enumerate()
        {
            layer_gradients.clip_by(clipping);

            optimizer.update(layer_index, layer, &layer_gradients);
        }

        optimizer.step();
    }

    /// Computes the gradients for each layer using backpropagation.
    /// # Arguments
    /// - `activations`: A vector of 2D arrays representing the activations of each layer.
    /// - `targets`: A 2D array representing the true labels for the inputs.
    fn backward(
        &self,
        activations: &[Array2<f32>],
        targets: ArrayView2<f32>,
        loss_function: &Arc<dyn LossFunction>,
    ) -> Vec<Gradients> {
        let m = targets.ncols() as f32;

        let last_act = last_activation(activations);
        // Clip to (0, 1) interior so gradient() and vjp() see the same values;
        // their product then cancels exactly to p − y even near saturation.
        let safe_act = last_act.mapv(|p| p.clamp(1e-15_f32, 1.0 - 1e-15_f32));
        let loss_grad = loss_function.gradient(safe_act.view(), targets);
        let mut dz = self
            .layers
            .last()
            .unwrap()
            .activation
            .vjp(loss_grad.view(), safe_act.view());

        let mut gradients = Vec::with_capacity(activations.len() - 1);

        for i in (1..self.layers.len() + 1).rev() {
            let previous_activations = activations[i - 1].view();
            let dw = dz.dot(&previous_activations.t()) / m;
            let db = dz.sum_axis(Axis(1)) / m;

            gradients.insert(0, Gradients { dw, db });

            if i > 1 {
                let next_layer = &self.layers[i - 1];
                let da = next_layer.weights.t().dot(&dz);
                dz = self.layers[i - 2]
                    .activation
                    .vjp(da.view(), previous_activations);
            }
        }

        gradients
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::SIGMOID;
    use crate::loss_functions::CROSS_ENTROPY_LOSS;
    use crate::model::{NeuralNetwork, NeuronLayerSpec};
    use ndarray::{Array1, array};

    fn compute_loss(
        model: &NeuralNetwork,
        inputs: &Array2<f32>,
        targets: &Array2<f32>,
        loss_fn: &Arc<dyn LossFunction>,
    ) -> f32 {
        let pred = model.predict(inputs.view());
        loss_fn.compute(pred.view(), targets.view())
    }

    #[test]
    fn backprop_gradients_match_numerical_approximation() {
        // Network: 2 inputs -> 2 hidden (sigmoid) -> 1 output (sigmoid, binary)
        // Using sigmoid everywhere to avoid relu's non-differentiable point at 0
        let specs = NeuronLayerSpec::network_for(vec![2], &*SIGMOID, 2);
        let mut model = NeuralNetwork::initialization(2, &specs);

        // Fixed weights for reproducibility
        model.layers[0].weights = array![[0.1, -0.2], [0.3, 0.1]];
        model.layers[0].biases = Array1::from_vec(vec![0.05, -0.05]);
        model.layers[1].weights = array![[0.4, -0.1]];
        model.layers[1].biases = Array1::from_vec(vec![0.1]);

        let inputs = array![[0.5, -0.3, 0.8], [0.2, 0.7, -0.5]]; // (2 features, 3 samples)
        let targets = array![[1.0, 0.0, 1.0]]; // (1 output, 3 samples)

        let loss_fn: Arc<dyn LossFunction> = CROSS_ENTROPY_LOSS.clone();
        let eps = 1e-4_f32;
        // f32 finite differences amplify rounding errors for small gradients;
        // 5% tolerates f32 noise while still catching real bugs (which cause >20% error)
        let tolerance = 5e-2_f32;

        let activations = model.forward(inputs.view());
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
            let (rows, cols) = model.layers[layer_idx].weights.dim();
            for i in 0..rows {
                for j in 0..cols {
                    let mut m_plus = model.clone();
                    m_plus.layers[layer_idx].weights[[i, j]] += eps;
                    let mut m_minus = model.clone();
                    m_minus.layers[layer_idx].weights[[i, j]] -= eps;
                    let numerical = (compute_loss(&m_plus, &inputs, &targets, &loss_fn)
                        - compute_loss(&m_minus, &inputs, &targets, &loss_fn))
                        / (2.0 * eps);
                    check(
                        layer_grads.dw[[i, j]],
                        numerical,
                        &format!("W[{layer_idx}][{i},{j}]"),
                    );
                }
            }

            for i in 0..model.layers[layer_idx].biases.len() {
                let mut m_plus = model.clone();
                m_plus.layers[layer_idx].biases[i] += eps;
                let mut m_minus = model.clone();
                m_minus.layers[layer_idx].biases[i] -= eps;
                let numerical = (compute_loss(&m_plus, &inputs, &targets, &loss_fn)
                    - compute_loss(&m_minus, &inputs, &targets, &loss_fn))
                    / (2.0 * eps);
                check(
                    layer_grads.db[i],
                    numerical,
                    &format!("b[{layer_idx}][{i}]"),
                );
            }
        }
    }

    #[test]
    fn backprop_gradients_match_numerical_approximation_softmax_output() {
        // Single-layer network (inputs → softmax), no hidden layers.
        // Verifies that loss.gradient() (→ -y/p) composed with softmax.vjp() yields the
        // correct dz (= p - y) and that the resulting weight/bias gradients match finite
        // differences. The hidden-layer chain rule is covered by the sigmoid test.
        // A single output layer keeps gradients large (O(0.1)) and well within f32 precision.
        let specs = NeuronLayerSpec::network_for(vec![], &*SIGMOID, 3);
        let mut model = NeuralNetwork::initialization(2, &specs);

        model.layers[0].weights = array![[0.5, -0.3], [0.2, 0.8], [-0.4, 0.1]];
        model.layers[0].biases = Array1::from_vec(vec![0.1, -0.2, 0.1]);

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

        let activations = model.forward(inputs.view());
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
            let (rows, cols) = model.layers[layer_idx].weights.dim();
            for i in 0..rows {
                for j in 0..cols {
                    let mut m_plus = model.clone();
                    m_plus.layers[layer_idx].weights[[i, j]] += eps;
                    let mut m_minus = model.clone();
                    m_minus.layers[layer_idx].weights[[i, j]] -= eps;
                    let numerical = (compute_loss(&m_plus, &inputs, &targets, &loss_fn)
                        - compute_loss(&m_minus, &inputs, &targets, &loss_fn))
                        / (2.0 * eps);
                    check(
                        layer_grads.dw[[i, j]],
                        numerical,
                        &format!("W[{layer_idx}][{i},{j}]"),
                    );
                }
            }
            for i in 0..model.layers[layer_idx].biases.len() {
                let mut m_plus = model.clone();
                m_plus.layers[layer_idx].biases[i] += eps;
                let mut m_minus = model.clone();
                m_minus.layers[layer_idx].biases[i] -= eps;
                let numerical = (compute_loss(&m_plus, &inputs, &targets, &loss_fn)
                    - compute_loss(&m_minus, &inputs, &targets, &loss_fn))
                    / (2.0 * eps);
                check(
                    layer_grads.db[i],
                    numerical,
                    &format!("b[{layer_idx}][{i}]"),
                );
            }
        }
    }
}
