use crate::evaluation::EvaluationSet;

/// A single recorded point of a training run: the absolute epoch number and
/// the evaluation computed at that epoch.
#[derive(Clone, Copy)]
pub struct Checkpoint {
    pub epoch: usize,
    pub evaluation: EvaluationSet,
}

/// Pure value object: the recorded checkpoints of a training run, ordered by epoch.
#[derive(Clone)]
pub struct Checkpoints(Vec<Checkpoint>);

impl Checkpoints {
    pub fn new(checkpoints: Vec<Checkpoint>) -> Self {
        Checkpoints(checkpoints)
    }

    /// Returns the number of recorded checkpoints.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns true if no checkpoints have been recorded.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Returns the absolute epoch number of each recorded checkpoint.
    pub fn epochs(&self) -> Vec<usize> {
        self.0.iter().map(|c| c.epoch).collect()
    }

    /// Returns the final evaluation recorded at the last checkpoint, if available.
    pub fn final_evaluation(&self) -> Option<EvaluationSet> {
        self.0.last().map(|c| c.evaluation)
    }

    /// Returns the training losses recorded at each checkpoint.
    pub fn train_losses(&self) -> Vec<f32> {
        self.0.iter().map(|c| c.evaluation.train.loss).collect()
    }

    /// Returns the final training loss, if available.
    pub fn final_train_loss(&self) -> Option<f32> {
        self.0.last().map(|c| c.evaluation.train.loss)
    }

    /// Returns the validation losses recorded at each checkpoint, if available.
    pub fn validation_losses(&self) -> Vec<f32> {
        self.0
            .iter()
            .filter_map(|c| c.evaluation.validation.map(|val| val.loss))
            .collect()
    }

    /// Returns the final validation loss, if available.
    pub fn final_validation_loss(&self) -> Option<f32> {
        self.0
            .last()
            .and_then(|c| c.evaluation.validation.map(|val| val.loss))
    }

    /// Returns the test losses recorded at each checkpoint.
    pub fn test_losses(&self) -> Vec<f32> {
        self.0.iter().map(|c| c.evaluation.test.loss).collect()
    }

    /// Returns the final test loss, if available.
    pub fn final_test_loss(&self) -> Option<f32> {
        self.0.last().map(|c| c.evaluation.test.loss)
    }

    /// Returns the training accuracies recorded at each checkpoint.
    pub fn train_accuracies(&self) -> Vec<f32> {
        self.0.iter().map(|c| c.evaluation.train.accuracy).collect()
    }

    /// Returns the final training accuracy, if available.
    pub fn final_train_accuracy(&self) -> Option<f32> {
        self.0.last().map(|c| c.evaluation.train.accuracy)
    }

    /// Returns the validation accuracies recorded at each checkpoint, if available.
    pub fn validation_accuracies(&self) -> Vec<f32> {
        self.0
            .iter()
            .filter_map(|c| c.evaluation.validation.map(|val| val.accuracy))
            .collect()
    }

    /// Returns the final validation accuracy, if available.
    pub fn final_validation_accuracy(&self) -> Option<f32> {
        self.0
            .last()
            .and_then(|c| c.evaluation.validation.map(|val| val.accuracy))
    }

    /// Returns the test accuracies recorded at each checkpoint.
    pub fn test_accuracies(&self) -> Vec<f32> {
        self.0.iter().map(|c| c.evaluation.test.accuracy).collect()
    }

    /// Returns the final test accuracy, if available.
    pub fn final_test_accuracy(&self) -> Option<f32> {
        self.0.last().map(|c| c.evaluation.test.accuracy)
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

    fn checkpoint(
        epoch: usize,
        train: (f32, f32),
        validation: Option<(f32, f32)>,
        test: (f32, f32),
    ) -> Checkpoint {
        Checkpoint {
            epoch,
            evaluation: EvaluationSet {
                train: Evaluation {
                    loss: train.0,
                    accuracy: train.1,
                },
                validation: validation.map(|(loss, accuracy)| Evaluation { loss, accuracy }),
                test: Evaluation {
                    loss: test.0,
                    accuracy: test.1,
                },
            },
        }
    }

    #[test]
    fn new_checkpoints_is_empty_when_no_checkpoints() {
        let checkpoints = Checkpoints::new(Vec::new());
        assert!(checkpoints.is_empty());
        assert_eq!(checkpoints.len(), 0);
    }

    #[test]
    fn empty_checkpoints_has_no_final_metrics() {
        let checkpoints = Checkpoints::new(Vec::new());
        assert!(checkpoints.final_evaluation().is_none());
        assert!(checkpoints.final_train_loss().is_none());
        assert!(checkpoints.final_validation_loss().is_none());
        assert!(checkpoints.final_test_loss().is_none());
        assert!(checkpoints.final_train_accuracy().is_none());
        assert!(checkpoints.final_validation_accuracy().is_none());
        assert!(checkpoints.final_test_accuracy().is_none());
        assert!(checkpoints.loss_range().is_none());
        assert!(checkpoints.accuracy_range().is_none());
        assert!(checkpoints.validation_losses().is_empty());
        assert!(checkpoints.validation_accuracies().is_empty());
        assert!(checkpoints.epochs().is_empty());
    }

    #[test]
    fn series_and_finals_track_recorded_values() {
        let checkpoints = Checkpoints::new(vec![
            checkpoint(0, (1.0, 50.0), Some((1.2, 45.0)), (1.5, 40.0)),
            checkpoint(1, (0.5, 70.0), Some((0.6, 65.0)), (0.8, 60.0)),
        ]);

        assert_eq!(checkpoints.train_losses(), vec![1.0, 0.5]);
        assert_eq!(checkpoints.validation_losses(), vec![1.2, 0.6]);
        assert_eq!(checkpoints.test_losses(), vec![1.5, 0.8]);
        assert_eq!(checkpoints.train_accuracies(), vec![50.0, 70.0]);
        assert_eq!(checkpoints.validation_accuracies(), vec![45.0, 65.0]);
        assert_eq!(checkpoints.test_accuracies(), vec![40.0, 60.0]);

        assert_eq!(checkpoints.final_train_loss(), Some(0.5));
        assert_eq!(checkpoints.final_validation_loss(), Some(0.6));
        assert_eq!(checkpoints.final_test_loss(), Some(0.8));
        assert_eq!(checkpoints.final_train_accuracy(), Some(70.0));
        assert_eq!(checkpoints.final_validation_accuracy(), Some(65.0));
        assert_eq!(checkpoints.final_test_accuracy(), Some(60.0));
        assert!(checkpoints.final_evaluation().is_some());

        assert_eq!(checkpoints.epochs(), vec![0, 1]);
    }

    #[test]
    fn validation_metrics_are_skipped_when_absent() {
        let checkpoints = Checkpoints::new(vec![checkpoint(0, (1.0, 50.0), None, (1.5, 40.0))]);

        assert!(checkpoints.validation_losses().is_empty());
        assert!(checkpoints.validation_accuracies().is_empty());
        assert!(checkpoints.final_validation_loss().is_none());
        assert!(checkpoints.final_validation_accuracy().is_none());
    }

    #[test]
    fn ranges_span_all_splits() {
        let checkpoints = Checkpoints::new(vec![
            checkpoint(0, (1.0, 50.0), Some((1.2, 45.0)), (1.5, 40.0)),
            checkpoint(1, (0.5, 70.0), Some((0.6, 65.0)), (0.8, 90.0)),
        ]);

        assert_eq!(checkpoints.loss_range(), Some((0.5, 1.5)));
        assert_eq!(checkpoints.accuracy_range(), Some((40.0, 90.0)));
    }
}
