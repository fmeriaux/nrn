use crate::core::neuron_network::{NeuronNetwork, accuracy, log_loss};
use ndarray::Array2;

/// Training history to store the state of the model, loss, and accuracy at each interval for visualization and analysis.
/// # Properties
/// - `interval`: Number of iterations between each recorded checkpoint.
/// - `model`: Vector of `NeuronNetwork` representing the state of the model at each recorded checkpoint.
/// - `loss`: Vector of loss values corresponding to each recorded checkpoint.
/// - `train_accuracy`: Vector of training accuracy values corresponding to each recorded checkpoint.
/// - `test_accuracy`: Vector of test accuracy values corresponding to each recorded checkpoint.
pub struct TrainingHistory {
    pub interval: usize,
    pub model: Vec<NeuronNetwork>,
    pub loss: Vec<f32>,
    pub train_accuracy: Vec<f32>,
    pub test_accuracy: Vec<f32>,
}

impl TrainingHistory {
    /// Creates a new `TrainingHistory` instance with a specified interval and an epoch count.
    /// When the `interval` is zero, it will return `None`.
    /// # Arguments
    /// - `interval`: The number of epochs between each checkpoint.
    /// - `epochs`: The total number of epochs for the training process.
    pub fn by_interval(interval: usize, epochs: usize) -> Option<Self> {
        let capacity = epochs
            .checked_div(interval)
            .map(|step| if step > 0 { step } else { 1 });

        capacity.map(|capacity| TrainingHistory {
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
    /// - `train_activations`: The activations (outputs) of the model for the training data.
    /// - `train_expectations`: The expected outputs (labels) for the training data.
    /// - `test_activations`: The activations (outputs) of the model for the test data.
    /// - `test_expectations`: The expected outputs (labels) for the test data.
    pub fn checkpoint(
        &mut self,
        model: &NeuronNetwork,
        train_activations: &Array2<f32>,
        train_expectations: &Array2<f32>,
        test_activations: &Array2<f32>,
        test_expectations: &Array2<f32>,
    ) {
        self.model.push(model.clone());
        self.loss
            .push(log_loss(&train_activations, &train_expectations));
        self.train_accuracy
            .push(accuracy(&train_activations, &train_expectations));
        self.test_accuracy
            .push(accuracy(&test_activations, &test_expectations));
    }
}
