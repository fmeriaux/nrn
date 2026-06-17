use crate::console::{MODEL_ICON, saved_at};
use nrn::evaluation::EvaluationSet;
use nrn::model::NeuralNetwork;
use nrn::training::{CallbackResult, TrainerCallback, TrainingOutcome};
use std::path::PathBuf;

/// Saves the final model to disk once training ends, unless the run diverged
/// without recovery (in which case `model` is `None`).
pub struct ModelSaver {
    path: PathBuf,
}

impl ModelSaver {
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }
}

impl TrainerCallback for ModelSaver {
    fn on_train_end(
        &mut self,
        _outcome: TrainingOutcome,
        model: Option<&NeuralNetwork>,
        _eval: Option<&EvaluationSet>,
        _epoch: usize,
    ) -> CallbackResult {
        if let Some(model) = model {
            let path = model.save(&self.path)?;
            saved_at(MODEL_ICON, "NEURAL NETWORK", path);
        }
        Ok(())
    }
}
