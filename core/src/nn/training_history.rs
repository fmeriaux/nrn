use crate::evaluation::EvaluationSet;
use std::path::PathBuf;

/// In-memory training history: snapshot evaluations and optional disk paths for lazy model loading.
#[derive(Clone)]
pub struct TrainingHistory {
    /// The interval (in epochs) at which snapshots were recorded.
    pub interval: usize,
    /// The recorded evaluations at each checkpoint for each training split.
    pub evaluations: Vec<EvaluationSet>,
    /// Sorted snapshot file paths, set only when loaded from disk via [`TrainingHistory::load`].
    /// Empty for in-memory histories built with [`by_interval`] / [`record`].
    pub(crate) snapshot_paths: Vec<PathBuf>,
}

impl TrainingHistory {
    /// Creates a new `TrainingHistory` that records evaluations at `interval` epochs.
    /// Returns `None` when `interval` is zero.
    pub fn by_interval(interval: usize, epochs: usize) -> Option<Self> {
        let capacity = epochs
            .checked_div(interval)
            .map(|step| if step > 0 { step } else { 1 });

        capacity
            // Add one to include the initial state before training starts
            .map(|capacity| capacity + 1)
            .map(|capacity| TrainingHistory {
                interval,
                evaluations: Vec::with_capacity(capacity),
                snapshot_paths: Vec::new(),
            })
    }

    /// Records evaluations for the current epoch.
    pub fn record(&mut self, evaluations: &EvaluationSet) {
        self.evaluations.push(*evaluations);
    }

    /// Returns the number of recorded snapshots.
    pub fn len(&self) -> usize {
        self.evaluations.len()
    }

    /// Returns true if no snapshots have been recorded.
    pub fn is_empty(&self) -> bool {
        self.evaluations.is_empty()
    }

    /// Returns the final evaluation recorded at the last snapshot, if available.
    pub fn final_evaluation(&self) -> Option<EvaluationSet> {
        self.evaluations.last().copied()
    }

    /// Returns the training losses recorded at each snapshot.
    pub fn train_losses(&self) -> Vec<f32> {
        self.evaluations
            .iter()
            .map(|eval| eval.train.loss)
            .collect()
    }

    /// Returns the final training loss, if available.
    pub fn final_train_loss(&self) -> Option<f32> {
        self.evaluations.last().map(|eval| eval.train.loss)
    }

    /// Returns the validation losses recorded at each snapshot, if available.
    pub fn validation_losses(&self) -> Vec<f32> {
        self.evaluations
            .iter()
            .filter_map(|eval| eval.validation.map(|val| val.loss))
            .collect()
    }

    /// Returns the final validation loss, if available.
    pub fn final_validation_loss(&self) -> Option<f32> {
        self.evaluations
            .last()
            .and_then(|eval| eval.validation.map(|val| val.loss))
    }

    /// Returns the test losses recorded at each snapshot.
    pub fn test_losses(&self) -> Vec<f32> {
        self.evaluations.iter().map(|eval| eval.test.loss).collect()
    }

    /// Returns the final test loss, if available.
    pub fn final_test_loss(&self) -> Option<f32> {
        self.evaluations.last().map(|eval| eval.test.loss)
    }

    /// Returns the training accuracies recorded at each snapshot.
    pub fn train_accuracies(&self) -> Vec<f32> {
        self.evaluations
            .iter()
            .map(|eval| eval.train.accuracy)
            .collect()
    }

    /// Returns the final training accuracy, if available.
    pub fn final_train_accuracy(&self) -> Option<f32> {
        self.evaluations.last().map(|eval| eval.train.accuracy)
    }

    /// Returns the validation accuracies recorded at each snapshot, if available.
    pub fn validation_accuracies(&self) -> Vec<f32> {
        self.evaluations
            .iter()
            .filter_map(|eval| eval.validation.map(|val| val.accuracy))
            .collect()
    }

    /// Returns the final validation accuracy, if available.
    pub fn final_validation_accuracy(&self) -> Option<f32> {
        self.evaluations
            .last()
            .and_then(|eval| eval.validation.map(|val| val.accuracy))
    }

    /// Returns the test accuracies recorded at each snapshot.
    pub fn test_accuracies(&self) -> Vec<f32> {
        self.evaluations
            .iter()
            .map(|eval| eval.test.accuracy)
            .collect()
    }

    /// Returns the final test accuracy, if available.
    pub fn final_test_accuracy(&self) -> Option<f32> {
        self.evaluations.last().map(|eval| eval.test.accuracy)
    }

    /// Returns the (min, max) range of all recorded loss values, or `None` if empty.
    pub fn loss_range(&self) -> Option<(f32, f32)> {
        self.train_losses()
            .into_iter()
            .chain(self.validation_losses())
            .chain(self.test_losses())
            .collect::<Vec<f32>>()
            .range()
    }

    /// Returns the (min, max) range of all recorded accuracy values, or `None` if empty.
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
    use crate::evaluation::Evaluation;

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

    #[test]
    fn by_interval_zero_returns_none() {
        assert!(TrainingHistory::by_interval(0, 100).is_none());
    }

    #[test]
    fn by_interval_reserves_capacity_for_each_step_plus_initial() {
        let th = TrainingHistory::by_interval(10, 100).unwrap();
        assert_eq!(th.interval, 10);
        assert!(th.evaluations.capacity() >= 11);
        assert!(th.is_empty());
        assert_eq!(th.len(), 0);
    }

    #[test]
    fn by_interval_larger_than_epochs_keeps_at_least_one_step() {
        let th = TrainingHistory::by_interval(200, 100).unwrap();
        assert!(th.evaluations.capacity() >= 2);
    }

    #[test]
    fn record_appends_evaluation() {
        let mut th = TrainingHistory::by_interval(1, 2).unwrap();

        th.record(&eval_set((1.0, 50.0), None, (1.5, 40.0)));
        th.record(&eval_set((0.5, 70.0), None, (0.8, 60.0)));

        assert_eq!(th.len(), 2);
        assert!(!th.is_empty());
        assert_eq!(th.evaluations.len(), 2);
    }

    #[test]
    fn empty_history_has_no_final_metrics() {
        let th = TrainingHistory::by_interval(1, 1).unwrap();
        assert!(th.final_evaluation().is_none());
        assert!(th.final_train_loss().is_none());
        assert!(th.final_validation_loss().is_none());
        assert!(th.final_test_loss().is_none());
        assert!(th.final_train_accuracy().is_none());
        assert!(th.final_validation_accuracy().is_none());
        assert!(th.final_test_accuracy().is_none());
        assert!(th.loss_range().is_none());
        assert!(th.accuracy_range().is_none());
        assert!(th.validation_losses().is_empty());
        assert!(th.validation_accuracies().is_empty());
    }

    #[test]
    fn series_and_finals_track_recorded_values() {
        let mut th = TrainingHistory::by_interval(1, 2).unwrap();
        th.record(&eval_set((1.0, 50.0), Some((1.2, 45.0)), (1.5, 40.0)));
        th.record(&eval_set((0.5, 70.0), Some((0.6, 65.0)), (0.8, 60.0)));

        assert_eq!(th.train_losses(), vec![1.0, 0.5]);
        assert_eq!(th.validation_losses(), vec![1.2, 0.6]);
        assert_eq!(th.test_losses(), vec![1.5, 0.8]);
        assert_eq!(th.train_accuracies(), vec![50.0, 70.0]);
        assert_eq!(th.validation_accuracies(), vec![45.0, 65.0]);
        assert_eq!(th.test_accuracies(), vec![40.0, 60.0]);

        assert_eq!(th.final_train_loss(), Some(0.5));
        assert_eq!(th.final_validation_loss(), Some(0.6));
        assert_eq!(th.final_test_loss(), Some(0.8));
        assert_eq!(th.final_train_accuracy(), Some(70.0));
        assert_eq!(th.final_validation_accuracy(), Some(65.0));
        assert_eq!(th.final_test_accuracy(), Some(60.0));
        assert!(th.final_evaluation().is_some());
    }

    #[test]
    fn validation_metrics_are_skipped_when_absent() {
        let mut th = TrainingHistory::by_interval(1, 1).unwrap();
        th.record(&eval_set((1.0, 50.0), None, (1.5, 40.0)));

        assert!(th.validation_losses().is_empty());
        assert!(th.validation_accuracies().is_empty());
        assert!(th.final_validation_loss().is_none());
        assert!(th.final_validation_accuracy().is_none());
    }

    #[test]
    fn ranges_span_all_splits() {
        let mut th = TrainingHistory::by_interval(1, 2).unwrap();
        th.record(&eval_set((1.0, 50.0), Some((1.2, 45.0)), (1.5, 40.0)));
        th.record(&eval_set((0.5, 70.0), Some((0.6, 65.0)), (0.8, 90.0)));

        assert_eq!(th.loss_range(), Some((0.5, 1.5)));
        assert_eq!(th.accuracy_range(), Some((40.0, 90.0)));
    }
}
