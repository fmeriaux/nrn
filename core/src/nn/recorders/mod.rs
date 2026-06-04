use crate::evaluation::EvaluationSet;
use crate::model::NeuralNetwork;
use std::io::Result;

/// Observes training progress by recording model snapshots at each checkpoint.
///
/// The training loop calls [`record`] at each checkpoint interval without
/// knowing whether snapshots are written to disk or discarded.
pub trait Recorder {
    fn record(&mut self, model: &NeuralNetwork, evaluation: &EvaluationSet) -> Result<()>;
}

/// A recorder that discards all snapshots (used when checkpointing is disabled).
pub struct NoOpRecorder;

impl Recorder for NoOpRecorder {
    fn record(&mut self, _model: &NeuralNetwork, _evaluation: &EvaluationSet) -> Result<()> {
        Ok(())
    }
}

/// Orchestrates when model snapshots are written during training.
///
/// Wraps a [`Recorder`] and adds interval-based scheduling so the training
/// loop stays free of checkpoint-timing decisions. The concrete recorder
/// (file-backed or no-op) is supplied by the caller.
pub struct Checkpoints {
    recorder: Box<dyn Recorder>,
    interval: usize,
    total_epochs: usize,
}

impl Checkpoints {
    /// Creates a `Checkpoints` backed by the given recorder.
    pub fn new(recorder: Box<dyn Recorder>, interval: usize, total_epochs: usize) -> Self {
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
    pub fn record(&mut self, model: &NeuralNetwork, eval: &EvaluationSet) -> Result<()> {
        self.recorder.record(model, eval)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::RELU;
    use crate::evaluation::Evaluation;
    use crate::model::{NeuralNetwork, NeuronLayerSpec};

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
    fn noop_recorder_record_always_succeeds() {
        let mut r = NoOpRecorder;
        assert!(r.record(&sample_model(), &sample_eval()).is_ok());
    }

    #[test]
    fn checkpoints_is_due_respects_interval() {
        let c = Checkpoints::new(Box::new(NoOpRecorder), 5, 20);
        assert!(!c.is_due(0)); // epoch 1
        assert!(!c.is_due(3)); // epoch 4
        assert!(c.is_due(4)); // epoch 5 — boundary
        assert!(!c.is_due(5)); // epoch 6
        assert!(c.is_due(9)); // epoch 10 — boundary
        assert!(c.is_due(19)); // epoch 20 — last epoch
    }

    #[test]
    fn checkpoints_is_due_false_when_interval_zero() {
        let c = Checkpoints::new(Box::new(NoOpRecorder), 0, 10);
        for epoch in 0..10 {
            assert!(!c.is_due(epoch));
        }
    }

    #[test]
    fn checkpoints_record_delegates_to_recorder() {
        let mut c = Checkpoints::new(Box::new(NoOpRecorder), 5, 10);
        assert!(c.record(&sample_model(), &sample_eval()).is_ok());
    }
}
