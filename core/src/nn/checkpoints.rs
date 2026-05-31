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
        self.evaluations.push(*evaluations);
    }

    /// Returns the number of recorded checkpoints.
    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    /// Returns true if no checkpoints have been recorded.
    pub fn is_empty(&self) -> bool {
        self.snapshots.is_empty()
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
        self.evaluations
            .last()
            .and_then(|eval| eval.validation.map(|val| val.loss))
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
        self.evaluations
            .last()
            .and_then(|eval| eval.validation.map(|val| val.accuracy))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::SIGMOID;
    use crate::evaluation::Evaluation;
    use crate::model::{NeuralNetwork, NeuronLayer};
    use ndarray::array;

    /// Builds an `EvaluationSet` with explicit metric values, optionally including a validation split.
    fn eval_set(
        train: (f32, f32),
        validation: Option<(f32, f32)>,
        test: (f32, f32),
    ) -> EvaluationSet {
        EvaluationSet {
            train: Evaluation {
                loss: train.0,
                accuracy: train.1,
            },
            validation: validation.map(|(loss, accuracy)| Evaluation { loss, accuracy }),
            test: Evaluation {
                loss: test.0,
                accuracy: test.1,
            },
        }
    }

    /// A minimal single-layer network used as a snapshot payload.
    fn dummy_model() -> NeuralNetwork {
        NeuralNetwork {
            layers: vec![NeuronLayer {
                weights: array![[0.0, 0.0]],
                biases: array![0.0],
                activation: SIGMOID.clone(),
            }],
        }
    }

    #[test]
    fn by_interval_zero_returns_none() {
        assert!(Checkpoints::by_interval(0, 100).is_none());
    }

    #[test]
    fn by_interval_reserves_capacity_for_each_step_plus_initial() {
        // 100 epochs / interval 10 → 10 checkpoints + 1 initial state
        let cp = Checkpoints::by_interval(10, 100).unwrap();
        assert_eq!(cp.interval, 10);
        assert!(cp.snapshots.capacity() >= 11);
        assert!(cp.evaluations.capacity() >= 11);
        assert!(cp.is_empty());
        assert_eq!(cp.len(), 0);
    }

    #[test]
    fn by_interval_larger_than_epochs_keeps_at_least_one_step() {
        // interval > epochs → step would be 0, clamped to 1 (+1 initial)
        let cp = Checkpoints::by_interval(200, 100).unwrap();
        assert!(cp.snapshots.capacity() >= 2);
    }

    #[test]
    fn record_appends_snapshot_and_evaluation() {
        let mut cp = Checkpoints::by_interval(1, 2).unwrap();
        let model = dummy_model();

        cp.record(&model, &eval_set((1.0, 50.0), None, (1.5, 40.0)));
        cp.record(&model, &eval_set((0.5, 70.0), None, (0.8, 60.0)));

        assert_eq!(cp.len(), 2);
        assert!(!cp.is_empty());
        assert_eq!(cp.snapshots.len(), 2);
        assert_eq!(cp.evaluations.len(), 2);
    }

    #[test]
    fn empty_checkpoints_have_no_final_metrics() {
        let cp = Checkpoints::by_interval(1, 1).unwrap();
        assert!(cp.final_evaluation().is_none());
        assert!(cp.final_train_loss().is_none());
        assert!(cp.final_validation_loss().is_none());
        assert!(cp.final_test_loss().is_none());
        assert!(cp.final_train_accuracy().is_none());
        assert!(cp.final_validation_accuracy().is_none());
        assert!(cp.final_test_accuracy().is_none());
        assert!(cp.loss_range().is_none());
        assert!(cp.accuracy_range().is_none());
        assert!(cp.validation_losses().is_empty());
        assert!(cp.validation_accuracies().is_empty());
    }

    #[test]
    fn series_and_finals_track_recorded_values() {
        let mut cp = Checkpoints::by_interval(1, 2).unwrap();
        let model = dummy_model();
        cp.record(
            &model,
            &eval_set((1.0, 50.0), Some((1.2, 45.0)), (1.5, 40.0)),
        );
        cp.record(
            &model,
            &eval_set((0.5, 70.0), Some((0.6, 65.0)), (0.8, 60.0)),
        );

        assert_eq!(cp.train_losses(), vec![1.0, 0.5]);
        assert_eq!(cp.validation_losses(), vec![1.2, 0.6]);
        assert_eq!(cp.test_losses(), vec![1.5, 0.8]);
        assert_eq!(cp.train_accuracies(), vec![50.0, 70.0]);
        assert_eq!(cp.validation_accuracies(), vec![45.0, 65.0]);
        assert_eq!(cp.test_accuracies(), vec![40.0, 60.0]);

        assert_eq!(cp.final_train_loss(), Some(0.5));
        assert_eq!(cp.final_validation_loss(), Some(0.6));
        assert_eq!(cp.final_test_loss(), Some(0.8));
        assert_eq!(cp.final_train_accuracy(), Some(70.0));
        assert_eq!(cp.final_validation_accuracy(), Some(65.0));
        assert_eq!(cp.final_test_accuracy(), Some(60.0));
        assert!(cp.final_evaluation().is_some());
    }

    #[test]
    fn validation_metrics_are_skipped_when_absent() {
        let mut cp = Checkpoints::by_interval(1, 1).unwrap();
        cp.record(&dummy_model(), &eval_set((1.0, 50.0), None, (1.5, 40.0)));

        assert!(cp.validation_losses().is_empty());
        assert!(cp.validation_accuracies().is_empty());
        assert!(cp.final_validation_loss().is_none());
        assert!(cp.final_validation_accuracy().is_none());
    }

    #[test]
    fn ranges_span_all_splits() {
        let mut cp = Checkpoints::by_interval(1, 2).unwrap();
        let model = dummy_model();
        cp.record(
            &model,
            &eval_set((1.0, 50.0), Some((1.2, 45.0)), (1.5, 40.0)),
        );
        cp.record(
            &model,
            &eval_set((0.5, 70.0), Some((0.6, 65.0)), (0.8, 90.0)),
        );

        // loss spans the smallest validation loss to the largest train loss
        assert_eq!(cp.loss_range(), Some((0.5, 1.5)));
        // accuracy spans the smallest test accuracy to the largest test accuracy
        assert_eq!(cp.accuracy_range(), Some((40.0, 90.0)));
    }
}
