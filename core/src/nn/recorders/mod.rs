use crate::evaluation::EvaluationSet;
use crate::model::NeuralNetwork;
use std::io::Result;
use std::path::Path;

/// Observes training progress by recording model snapshots at each checkpoint.
///
/// The training loop calls [`record`] at each checkpoint interval without
/// knowing whether snapshots are written to disk or discarded.
pub trait Recorder {
    fn record(&mut self, model: &NeuralNetwork, evaluation: &EvaluationSet) -> Result<()>;

    /// Returns the directory where snapshots are written, or `None` for no-op recorders.
    fn dir(&self) -> Option<&Path>;
}

/// A recorder that discards all snapshots (used when checkpointing is disabled).
pub struct NoOpRecorder;

impl Recorder for NoOpRecorder {
    fn record(&mut self, _model: &NeuralNetwork, _evaluation: &EvaluationSet) -> Result<()> {
        Ok(())
    }

    fn dir(&self) -> Option<&Path> {
        None
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
    fn noop_recorder_dir_is_none() {
        assert!(NoOpRecorder.dir().is_none());
    }
}
