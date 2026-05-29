use crate::data::ModelDataset;
use crate::loss_functions::LossFunction;
use crate::model::{last_activation, NeuralNetwork};
use crate::nn::schedulers::Scheduler;
use crate::optimizers::Optimizer;
use ndarray::{Array1, Array2, ArrayView2, Axis};
use ndarray_rand::rand;
use std::sync::{Arc, Mutex};

/// Small constant to prevent division by zero in gradient clipping.
/// This value was chosen to be sufficiently small to avoid affecting the clipping behavior
/// while ensuring numerical stability during calculations.
const EPSILON: f32 = 1e-6;

/// Represents the learning rate used in optimization algorithms.
#[derive(Clone, Copy, Debug)]
pub struct LearningRate(f32);

impl LearningRate {
    /// Creates a new `LearningRate` instance with the specified value.
    /// # Panics
    /// - When the provided `value` is negative.
    /// # Arguments
    /// - `value`: The learning rate value to be used in optimization algorithms.
    pub fn new(value: f32) -> Self {
        assert!(value >= 0.0, "Learning rate must be non-negative.");
        LearningRate(value)
    }

    /// Returns the current learning rate value.
    pub fn value(&self) -> f32 {
        self.0
    }
}

pub enum GradientClipping {
    /// No gradient clipping is applied.
    None,
    /// Gradients are clipped to a maximum norm using the L2 norm.
    Norm { max_norm: f32 },
    /// Gradients are clipped to a maximum value element-wise.
    Value { min: f32, max: f32 },
}

/// Represents the gradients computed during backpropagation for a single layer.
pub struct Gradients {
    /// A 2D array where each element represents the gradient of the corresponding weight.
    pub dw: Array2<f32>,
    /// A 1D array where each element represents the gradient of the corresponding bias.
    pub db: Array1<f32>,
}

impl Gradients {
    /// Clips the gradients to a maximum norm, using the L2 norm.
    /// # Arguments
    /// - `max_norm`: The maximum norm to clip the gradients to.
    pub fn clip(&mut self, max_norm: f32) {
        let dw_norm = self.dw.mapv(|x| x.powi(2)).sum();
        let db_norm = self.db.mapv(|x| x.powi(2)).sum();
        let norm = (dw_norm + db_norm).sqrt();

        if norm > max_norm {
            let scale = max_norm / (norm + EPSILON);
            self.dw.mapv_inplace(|x| x * scale);
            self.db.mapv_inplace(|x| x * scale);
        }
    }

    /// Clips the gradients to a specified range element-wise.
    /// # Arguments
    /// - `min`: The minimum value to clip the gradients to.
    /// - `max`: The maximum value to clip the gradients to.
    pub fn clip_value(&mut self, min: f32, max: f32) {
        self.dw.mapv_inplace(|x| x.clamp(min, max));
        self.db.mapv_inplace(|x| x.clamp(min, max));
    }

    /// Clips the gradients based on the specified `GradientClipping` strategy.
    /// # Arguments
    /// - `clipping`: The `GradientClipping` strategy to apply.
    pub fn clip_by(&mut self, clipping: &GradientClipping) {
        match clipping {
            GradientClipping::None => {}
            GradientClipping::Norm { max_norm } => self.clip(*max_norm),
            GradientClipping::Value { min, max } => self.clip_value(*min, *max),
        }
    }
}

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
        optimizer: &Arc<Mutex<dyn Optimizer>>,
        scheduler: &Arc<Mutex<dyn Scheduler>>,
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
        let lr = scheduler.lock().expect("scheduler mutex was poisoned").step();
        optimizer.lock().expect("optimizer mutex was poisoned").set_learning_rate(lr);

        match batch_size {
            None => {
                let activations = self.forward(dataset.inputs.view());
                self.update_parameters(&activations, dataset.targets.view(), loss_function, optimizer, clipping);
            }
            Some(size) => {
                for batch in dataset.batches(size, &mut rand::thread_rng()) {
                    let activations = self.forward(batch.inputs.view());
                    self.update_parameters(&activations, batch.targets.view(), loss_function, optimizer, clipping);
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
        optimizer: &Arc<Mutex<dyn Optimizer>>,
        clipping: &GradientClipping,
    ) {
        let gradients = self.backward(activations, targets, loss_function);

        let mut optimizer = optimizer.lock().expect("optimizer mutex was poisoned");

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

        let mut dz = loss_function.gradient(last_activation(activations).view(), targets);

        let mut gradients = Vec::with_capacity(activations.len() - 1);

        for i in (1..self.layers.len() + 1).rev() {
            let previous_activations = activations[i - 1].view();
            let dw = dz.dot(&previous_activations.t()) / m;
            let db = dz.sum_axis(Axis(1)) / m;

            gradients.insert(0, Gradients { dw, db });

            if i > 1 {
                let next_layer = &self.layers[i - 1];
                dz = next_layer.weights.t().dot(&dz)
                    * self.layers[i - 2]
                        .activation
                        .derivative(previous_activations, targets);
            }
        }

        gradients
    }
}

/// Early stopping mechanism is used to halt training when the model's performance on a validation set
/// stops improving, thereby preventing overfitting.
pub struct EarlyStopping {
    /// The number of consecutive epochs without improvement after which training will be stopped.
    patience: usize,
    /// The best loss observed so far during training.
    best_loss: f32,
    /// The number of consecutive epochs without improvement.
    epochs_without_improvement: usize,
    /// Whether to restore the model to the state with the best observed loss when stopping.
    restore_best_model: bool,
    /// The best model observed during training, if `restore_best_model` is true.
    pub best_model: Option<NeuralNetwork>,
}

impl EarlyStopping {
    /// Creates a new `EarlyStopping` instance with the specified patience and restore_best_model flag.
    /// # Panics
    /// - When the `patience` is zero.
    pub fn new(patience: usize, restore_best_model: bool) -> Self {
        assert!(patience > 0, "Patience must be greater than zero.");

        EarlyStopping {
            patience,
            best_loss: f32::INFINITY,
            epochs_without_improvement: 0,
            restore_best_model,
            best_model: None,
        }
    }

    /// Checks if training should be stopped based on the current loss.
    pub fn check(&mut self, current_loss: f32, model: &NeuralNetwork) -> bool {
        if current_loss < self.best_loss {
            self.best_loss = current_loss;
            self.epochs_without_improvement = 0;

            if self.restore_best_model {
                self.best_model = Some(model.clone());
            }
        } else {
            self.epochs_without_improvement += 1;
        }

        self.epochs_without_improvement >= self.patience
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::SIGMOID;
    use crate::loss_functions::CROSS_ENTROPY_LOSS;
    use crate::model::{NeuralNetwork, NeuronLayerSpec};
    use ndarray::{Array1, array};

    fn compute_loss(model: &NeuralNetwork, inputs: &Array2<f32>, targets: &Array2<f32>, loss_fn: &Arc<dyn LossFunction>) -> f32 {
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
                    check(layer_grads.dw[[i, j]], numerical, &format!("W[{layer_idx}][{i},{j}]"));
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
                check(layer_grads.db[i], numerical, &format!("b[{layer_idx}][{i}]"));
            }
        }
    }
}