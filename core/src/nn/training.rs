use crate::accuracies::Accuracy;
use crate::loss_functions::LossFunction;
use crate::model::{NeuralNetwork, last_activation};
use crate::optimizers::Optimizer;
use ndarray::{Array1, Array2, ArrayView2, Axis};
use std::sync::{Arc, Mutex};

/// Small constant to prevent division by zero in gradient clipping.
/// This value was chosen to be sufficiently small to avoid affecting the clipping behavior
/// while ensuring numerical stability during calculations.
const EPSILON: f32 = 1e-6;

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
    /// Trains the network using the provided inputs and targets, updating the weights and biases.
    /// # Panics
    /// - When the `learning_rate` is less than or equal to zero.
    /// - When the number of columns in `inputs` does not match the length of `targets`.
    /// # Arguments
    /// - `inputs`: A 2D array representing the inputs to the network.
    /// - `targets`: A 2D array representing the true labels for the inputs.
    /// - `loss_function`: The loss function to use for computing the loss and its gradient.
    /// - `optimizer`: The optimizer to use for updating the weights and biases.
    /// - `clipping`: The gradient clipping strategy to apply during training.
    pub fn train(
        &mut self,
        inputs: ArrayView2<f32>,
        targets: ArrayView2<f32>,
        loss_function: &Arc<dyn LossFunction>,
        optimizer: &Arc<Mutex<dyn Optimizer>>,
        clipping: &GradientClipping,
    ) -> Array2<f32> {
        assert_eq!(
            inputs.ncols(),
            targets.ncols(),
            "Inputs and targets must have the same number of samples."
        );

        let activations = self.forward(inputs);

        self.update_parameters(&activations, targets, loss_function, optimizer, clipping);

        last_activation(&activations)
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

        let mut optimizer = optimizer.lock().unwrap();

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

/// Training history to store the state of the model, loss, and accuracy at each interval for visualization and analysis.
pub struct History {
    /// The interval (in epochs) at which checkpoints are recorded.
    pub interval: usize,
    /// The recorded states of the model at each checkpoint.
    pub model: Vec<NeuralNetwork>,
    /// The recorded loss values at each checkpoint.
    pub loss: Vec<f32>,
    /// The recorded training accuracy values at each checkpoint.
    pub train_accuracy: Vec<f32>,
    /// The recorded test accuracy values at each checkpoint.
    pub test_accuracy: Vec<f32>,
}

impl History {
    /// Creates a new `TrainingHistory` instance with a specified interval and an epoch count.
    /// When the `interval` is zero, it will return `None`.
    /// # Arguments
    /// - `interval`: The number of epochs between each checkpoint.
    /// - `epochs`: The total number of epochs for the training process.
    pub fn by_interval(interval: usize, epochs: usize) -> Option<Self> {
        let capacity = epochs
            .checked_div(interval)
            .map(|step| if step > 0 { step } else { 1 });

        capacity
            // Add one to include the initial state before training starts
            .map(|capacity| capacity + 1)
            // Initialize vectors with the calculated capacity
            .map(|capacity| History {
                interval,
                model: Vec::with_capacity(capacity),
                loss: Vec::with_capacity(capacity),
                train_accuracy: Vec::with_capacity(capacity),
                test_accuracy: Vec::with_capacity(capacity),
            })
    }

    /// Registers a new checkpoint in the training history with the current state of the model, loss, and accuracy.
    /// # Arguments
    /// - `model`: The current state of the `NeuronNetwork`.
    /// - `train_predictions`: The final outputs (predictions) of the model for the training data.
    /// - `train_targets`: The true labels for the training data.
    /// - `test_predictions`: The final outputs (predictions) of the model for the test data.
    /// - `test_targets`: The true labels for the test data.
    pub fn checkpoint(
        &mut self,
        model: &NeuralNetwork,
        loss_function: &Arc<dyn LossFunction>,
        accuracy: &Arc<dyn Accuracy>,
        train_predictions: ArrayView2<f32>,
        train_targets: ArrayView2<f32>,
        test_predictions: ArrayView2<f32>,
        test_targets: ArrayView2<f32>,
    ) {
        self.model.push(model.clone());
        self.loss
            .push(loss_function.compute(train_predictions, train_targets));
        self.train_accuracy
            .push(accuracy.compute(train_predictions, train_targets));
        self.test_accuracy
            .push(accuracy.compute(test_predictions, test_targets));
    }

    /// Returns the final training accuracy recorded in the training history, if available.
    pub fn final_train_accuracy(&self) -> Option<f32> {
        self.train_accuracy.last().copied()
    }

    /// Returns the final test accuracy recorded in the training history, if available.
    pub fn final_test_accuracy(&self) -> Option<f32> {
        self.test_accuracy.last().copied()
    }

    /// Returns the final loss value recorded in the training history, if available.
    pub fn final_loss(&self) -> Option<f32> {
        self.loss.last().copied()
    }

    fn range_for_vec(data: &Vec<f32>) -> Option<(f32, f32)> {
        if data.is_empty() {
            return None;
        }

        let (min, max) = data
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), &value| {
                (min.min(value), max.max(value))
            });

        Some((min, max))
    }

    /// Returns the range of loss values recorded in the training history as a tuple (min_loss, max_loss).
    /// If no loss values are recorded, it returns `None`.
    pub fn loss_range(&self) -> Option<(f32, f32)> {
        Self::range_for_vec(&self.loss)
    }

    /// Returns the range of test accuracy values recorded in the training history as a tuple (min_accuracy, max_accuracy).
    /// If no accuracy values are recorded, it returns `None`.
    pub fn train_accuracy_range(&self) -> Option<(f32, f32)> {
        Self::range_for_vec(&self.train_accuracy)
    }

    /// Returns the range of test accuracy values recorded in the training history as a tuple (min_accuracy, max_accuracy).
    /// If no accuracy values are recorded, it returns `None`.
    pub fn test_accuracy_range(&self) -> Option<(f32, f32)> {
        Self::range_for_vec(&self.test_accuracy)
    }

    /// Returns the combined range of training and test accuracy values recorded in the training history
    /// as a tuple (min_accuracy, max_accuracy).
    /// If no accuracy values are recorded, it returns `None`.
    pub fn accuracy_range(&self) -> Option<(f32, f32)> {
        let (train_min, train_max) = self.train_accuracy_range()?;
        let (test_min, test_max) = self.test_accuracy_range()?;
        Some((train_min.min(test_min), train_max.max(test_max)))
    }
}
