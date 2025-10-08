use crate::evaluation::EvaluationSet;
use crate::model::NeuralNetwork;

/// Manages the recording of model checkpoints during training.
#[derive(Clone)]
pub struct Checkpoints {
    /// The interval (in epochs) at which checkpoints are recorded.
    pub interval: usize,
    /// The recorded snapshots of the model at each checkpoint.
    pub snapshots: Vec<NeuralNetwork>,
    /// The recorded evaluations on the model at each checkpoint for each training set.
    pub evaluations: Vec<EvaluationSet>,
}

impl Checkpoints {
    /// Creates a new `Checkpoints` instance that records checkpoints at specified intervals.
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
            .map(|capacity| Checkpoints {
                interval,
                snapshots: Vec::with_capacity(capacity),
                evaluations: Vec::with_capacity(capacity),
            })
    }

    /// Registers a new checkpoint with the current model and its evaluations.
    /// This method records the current state of the model and its evaluation metrics on the training,
    /// validation (if available), and test datasets.
    /// # Arguments
    /// - `model`: The current state of the `NeuronNetwork`.
    /// - `evaluations`: The evaluations of the model on the training, validation (if available), and test datasets.
    pub fn record(&mut self, model: &NeuralNetwork, evaluations: &EvaluationSet) {
        self.snapshots.push(model.clone());
        self.evaluations.push(evaluations.clone());
    }

    /// Returns the number of recorded checkpoints.
    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    /// Returns the final evaluation recorded at the last checkpoint, if available.
    pub fn final_evaluation(&self) -> Option<EvaluationSet> {
        self.evaluations.last().copied()
    }

    /// Returns the training losses recorded at each checkpoint.
    pub fn train_losses(&self) -> Vec<f32> {
        self.evaluations
            .iter()
            .map(|eval| eval.train.loss)
            .collect()
    }

    /// Returns the final training loss recorded at the last checkpoint, if available.
    pub fn final_train_loss(&self) -> Option<f32> {
        self.evaluations.last().map(|eval| eval.train.loss)
    }

    /// Returns the validation losses recorded at each checkpoint, if available.
    pub fn validation_losses(&self) -> Vec<f32> {
        self.evaluations
            .iter()
            .filter_map(|eval| eval.validation.map(|val| val.loss))
            .collect()
    }

    /// Returns the final validation loss recorded at the last checkpoint, if available.
    pub fn final_validation_loss(&self) -> Option<f32> {
        self.evaluations.last().and_then(|eval| eval.validation.map(|val| val.loss))
    }

    /// Returns the test losses recorded at each checkpoint.
    pub fn test_losses(&self) -> Vec<f32> {
        self.evaluations.iter().map(|eval| eval.test.loss).collect()
    }

    /// Returns the final test loss recorded at the last checkpoint, if available.
    pub fn final_test_loss(&self) -> Option<f32> {
        self.evaluations.last().map(|eval| eval.test.loss)
    }

    /// Returns the training accuracies recorded at each checkpoint.
    pub fn train_accuracies(&self) -> Vec<f32> {
        self.evaluations
            .iter()
            .map(|eval| eval.train.accuracy)
            .collect()
    }

    /// Returns the final training accuracy recorded at the last checkpoint, if available.
    pub fn final_train_accuracy(&self) -> Option<f32> {
        self.evaluations.last().map(|eval| eval.train.accuracy)
    }

    /// Returns the validation accuracies recorded at each checkpoint, if available.
    pub fn validation_accuracies(&self) -> Vec<f32> {
        self.evaluations
            .iter()
            .filter_map(|eval| eval.validation.map(|val| val.accuracy))
            .collect()
    }

    /// Returns the final validation accuracy recorded at the last checkpoint, if available.
    pub fn final_validation_accuracy(&self) -> Option<f32> {
        self.evaluations.last().and_then(|eval| eval.validation.map(|val| val.accuracy))
    }

    /// Returns the test accuracies recorded at each checkpoint.
    pub fn test_accuracies(&self) -> Vec<f32> {
        self.evaluations
            .iter()
            .map(|eval| eval.test.accuracy)
            .collect()
    }

    /// Returns the final test accuracy recorded at the last checkpoint, if available.
    pub fn final_test_accuracy(&self) -> Option<f32> {
        self.evaluations.last().map(|eval| eval.test.accuracy)
    }

    /// Returns the range of loss values recorded as a tuple (min_loss, max_loss).
    /// If no loss values are recorded, it returns `None`.
    pub fn loss_range(&self) -> Option<(f32, f32)> {
        self.train_losses()
            .into_iter()
            .chain(self.validation_losses())
            .chain(self.test_losses())
            .collect::<Vec<f32>>()
            .range()
    }

    /// Returns the range of accuracy values recorded as a tuple (min_accuracy, max_accuracy).
    /// If no accuracy values are recorded, it returns `None`.
    pub fn accuracy_range(&self) -> Option<(f32, f32)> {
        self.train_accuracies()
            .into_iter()
            .chain(self.validation_accuracies())
            .chain(self.test_accuracies())
            .collect::<Vec<f32>>()
            .range()
    }
}

trait Range {
    fn range(&self) -> Option<(f32, f32)>;
}

impl Range for Vec<f32> {
    fn range(&self) -> Option<(f32, f32)> {
        if self.is_empty() {
            return None;
        }

        let (min, max) = self
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), &value| {
                (min.min(value), max.max(value))
            });

        Some((min, max))
    }
}
