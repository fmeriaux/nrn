mod checkpoints;
mod noop;

pub use checkpoints::Checkpoints;
pub use noop::NoOpSnapshotRecorder;

use crate::evaluation::EvaluationSet;
use crate::model::NeuralNetwork;
use std::io::Result;

/// Observes training progress by recording model snapshots at each checkpoint.
///
/// The training loop calls [`record`] at each checkpoint interval without
/// knowing whether snapshots are written to disk or discarded.
pub trait SnapshotRecorder {
    fn record(
        &mut self,
        model: &NeuralNetwork,
        evaluation: &EvaluationSet,
        epoch: usize,
    ) -> Result<()>;
}
