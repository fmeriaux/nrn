use crate::evaluation::EvaluationSet;
use crate::model::NeuralNetwork;
use crate::recorders::SnapshotRecorder;
use std::io::Result;

/// A recorder that discards all snapshots (used when checkpointing is disabled).
pub struct NoOpSnapshotRecorder;

impl SnapshotRecorder for NoOpSnapshotRecorder {
    fn record(
        &mut self,
        _model: &NeuralNetwork,
        _evaluation: &EvaluationSet,
        _epoch: usize,
    ) -> Result<()> {
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

    #[test]
    fn record_always_succeeds() {
        let mut r = NoOpSnapshotRecorder;
        let eval = EvaluationSet {
            train: Evaluation {
                loss: 0.5,
                accuracy: 0.8,
            },
            validation: None,
            test: Evaluation {
                loss: 0.6,
                accuracy: 0.75,
            },
        };
        assert!(r.record(&sample_model(), &eval, 42).is_ok());
    }
}
