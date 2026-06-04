use crate::evaluation::EvaluationSet;
use crate::model::NeuralNetwork;
use crate::recorders::SnapshotRecorder;
use std::io::Result;

/// Orchestrates when model snapshots are written during training.
///
/// Wraps a [`SnapshotRecorder`] and adds interval-based scheduling so the
/// training loop stays free of checkpoint-timing decisions. The concrete
/// recorder (file-backed or no-op) is supplied by the caller.
pub struct Checkpoints {
    recorder: Box<dyn SnapshotRecorder>,
    interval: usize,
    total_epochs: usize,
}

impl Checkpoints {
    /// Creates a `Checkpoints` backed by the given recorder.
    pub fn new(recorder: Box<dyn SnapshotRecorder>, interval: usize, total_epochs: usize) -> Self {
        Checkpoints {
            recorder,
            interval,
            total_epochs,
        }
    }

    /// Returns `true` if `epoch` (0-indexed) is a checkpoint boundary.
    pub fn is_due(&self, epoch: usize) -> bool {
        let n = epoch + 1;
        self.interval > 0 && (n % self.interval == 0 || n == self.total_epochs)
    }

    /// Records the model state unconditionally.
    ///
    /// Use for: initial snapshot before training, divergence recovery,
    /// early stopping flush when the interval didn't already capture the epoch.
    pub fn record(
        &mut self,
        model: &NeuralNetwork,
        eval: &EvaluationSet,
        epoch: usize,
    ) -> Result<()> {
        self.recorder.record(model, eval, epoch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::RELU;
    use crate::evaluation::Evaluation;
    use crate::model::{NeuralNetwork, NeuronLayerSpec};
    use crate::recorders::NoOpSnapshotRecorder;

    fn sample_model() -> NeuralNetwork {
        let specs = NeuronLayerSpec::network_for(vec![2], &*RELU, 2);
        NeuralNetwork::initialization(2, &specs)
    }

    fn sample_eval() -> EvaluationSet {
        EvaluationSet {
            train: Evaluation {
                loss: 0.5,
                accuracy: 0.8,
            },
            validation: None,
            test: Evaluation {
                loss: 0.6,
                accuracy: 0.75,
            },
        }
    }

    #[test]
    fn is_due_respects_interval() {
        let c = Checkpoints::new(Box::new(NoOpSnapshotRecorder), 5, 20);
        assert!(!c.is_due(0));
        assert!(!c.is_due(3));
        assert!(c.is_due(4)); // epoch 5 — boundary
        assert!(!c.is_due(5));
        assert!(c.is_due(9)); // epoch 10 — boundary
        assert!(c.is_due(19)); // epoch 20 — last
    }

    #[test]
    fn is_due_false_when_interval_zero() {
        let c = Checkpoints::new(Box::new(NoOpSnapshotRecorder), 0, 10);
        for epoch in 0..10 {
            assert!(!c.is_due(epoch));
        }
    }

    #[test]
    fn record_delegates_to_recorder() {
        let mut c = Checkpoints::new(Box::new(NoOpSnapshotRecorder), 5, 10);
        assert!(c.record(&sample_model(), &sample_eval(), 5).is_ok());
    }
}
