use crate::evaluation::EvaluationSet;
use crate::model::NeuralNetwork;
use crate::recorders::SnapshotRecorder;
use std::io::Result;

/// Orchestrates when model snapshots are written during training.
pub enum Checkpoints {
    /// Checkpointing disabled — all record calls are no-ops.
    Disabled,
    /// Records a snapshot every `interval` epochs and at the final epoch.
    Recording {
        recorder: Box<dyn SnapshotRecorder>,
        interval: usize,
        total_epochs: usize,
    },
}

impl Checkpoints {
    pub fn disabled() -> Self {
        Checkpoints::Disabled
    }

    /// # Panics
    /// Panics if `interval` is zero.
    pub fn recording(
        recorder: Box<dyn SnapshotRecorder>,
        interval: usize,
        total_epochs: usize,
    ) -> Self {
        assert!(interval > 0, "checkpoint interval must be > 0");
        Checkpoints::Recording { recorder, interval, total_epochs }
    }

    pub fn is_due(&self, epoch: usize) -> bool {
        match self {
            Checkpoints::Disabled => false,
            Checkpoints::Recording { interval, total_epochs, .. } => {
                epoch % interval == 0 || epoch == *total_epochs
            }
        }
    }

    /// Records a checkpoint if `epoch` falls on a scheduled boundary.
    /// Returns `true` if a snapshot was written.
    pub fn record(
        &mut self,
        model: &NeuralNetwork,
        eval: &EvaluationSet,
        epoch: usize,
    ) -> Result<bool> {
        if !self.is_due(epoch) {
            return Ok(false);
        }
        if let Checkpoints::Recording { recorder, .. } = self {
            recorder.record(model, eval, epoch)?;
        }
        Ok(true)
    }

    /// Records unconditionally, regardless of the interval schedule.
    /// Used for divergence recovery and early-stopping flushes.
    pub fn force_record(
        &mut self,
        model: &NeuralNetwork,
        eval: &EvaluationSet,
        epoch: usize,
    ) -> Result<()> {
        if let Checkpoints::Recording { recorder, .. } = self {
            recorder.record(model, eval, epoch)?;
        }
        Ok(())
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
            train: Evaluation { loss: 0.5, accuracy: 0.8 },
            validation: None,
            test: Evaluation { loss: 0.6, accuracy: 0.75 },
        }
    }

    struct Noop;
    impl SnapshotRecorder for Noop {
        fn record(&mut self, _: &NeuralNetwork, _: &EvaluationSet, _: usize) -> Result<()> {
            Ok(())
        }
    }

    fn recording(interval: usize, total: usize) -> Checkpoints {
        Checkpoints::recording(Box::new(Noop), interval, total)
    }

    #[test]
    fn disabled_never_records() {
        let mut c = Checkpoints::disabled();
        for epoch in 0..10 {
            assert!(!c.record(&sample_model(), &sample_eval(), epoch).unwrap());
        }
    }

    #[test]
    fn record_returns_true_at_interval_boundary() {
        let mut c = recording(5, 20);
        assert!(!c.record(&sample_model(), &sample_eval(), 1).unwrap());
        assert!(!c.record(&sample_model(), &sample_eval(), 4).unwrap());
        assert!(c.record(&sample_model(), &sample_eval(), 5).unwrap());  // epoch 5
        assert!(c.record(&sample_model(), &sample_eval(), 10).unwrap()); // epoch 10
        assert!(c.record(&sample_model(), &sample_eval(), 20).unwrap()); // epoch 20 — last
    }

    #[test]
    #[should_panic(expected = "interval must be > 0")]
    fn recording_panics_on_zero_interval() {
        Checkpoints::recording(Box::new(Noop), 0, 10);
    }

    #[test]
    fn is_due_is_public_so_callers_can_skip_expensive_work() {
        // is_due must be callable without going through record(), so training
        // loops can skip forward passes when no checkpoint is needed.
        let c = recording(5, 20);
        assert!(!c.is_due(1));
        assert!(!c.is_due(4));
        assert!(c.is_due(5));
        assert!(c.is_due(10));
        assert!(c.is_due(20)); // final epoch
    }

    #[test]
    fn force_record_always_succeeds() {
        let mut c = Checkpoints::disabled();
        assert!(c.force_record(&sample_model(), &sample_eval(), 2).is_ok());
    }
}
