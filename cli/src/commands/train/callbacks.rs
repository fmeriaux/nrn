//! The CLI's [`TrainerCallback`] adapters: the console reporter and the final
//! model saver. They share no code — only the trait — but both are small glue
//! between the training lifecycle and the CLI's display / persistence. Split
//! into a `callbacks/` submodule if this file grows.

use crate::display::{
    Artifacts, Describe, HyperParametersView, completed, saved, show, styled_bar, trace, warning,
};
use indicatif::ProgressBar;
use nrn::data::ModelSplit;
use nrn::evaluation::EvaluationSet;
use nrn::model::NeuralNetwork;
use nrn::optimizers::Optimizer;
use nrn::schedulers::Scheduler;
use nrn::training::{CallbackResult, HyperParameters, TrainerCallback, TrainingOutcome};
use std::path::{Path, PathBuf};

// ─── ConsoleMonitor ─────────────────────────────────────────────────────────

/// Reports the training run lifecycle on the console: a progress bar tracks
/// epochs, and the run configuration / final outcome are narrated as trace
/// lines. The bar is suspended while printing so lines never get garbled by
/// its redraws, and cleared before the final summary.
pub struct ConsoleMonitor {
    bar: ProgressBar,
    current: HyperParameters,
    previous: Option<HyperParameters>,
}

impl ConsoleMonitor {
    pub fn new(current: HyperParameters, previous: Option<HyperParameters>) -> Self {
        Self {
            bar: styled_bar(),
            current,
            previous,
        }
    }
}

impl TrainerCallback for ConsoleMonitor {
    fn on_restore(
        &mut self,
        epoch_start: usize,
        optimizer: Option<&dyn Optimizer>,
        scheduler: Option<&dyn Scheduler>,
    ) -> CallbackResult {
        if let Some(scheduler) = scheduler {
            completed!(
                "Restored {} scheduler state from checkpoint at epoch {epoch_start}",
                scheduler.name()
            );
        }
        if let Some(optimizer) = optimizer {
            completed!(
                "Restored {} optimizer state from checkpoint at epoch {epoch_start}",
                optimizer.name()
            );
        }
        Ok(())
    }

    fn on_train_start(&mut self, split: &ModelSplit) -> CallbackResult {
        self.bar.set_length(self.current.epochs() as u64);
        self.bar.set_message("Training");

        let view = HyperParametersView {
            current: &self.current,
            previous: self.previous.as_ref(),
        };

        self.bar.suspend(|| {
            show(&view);
            trace(&split.describe());
        });

        Ok(())
    }

    fn on_epoch_end(&mut self, _epoch: usize) -> CallbackResult {
        self.bar.inc(1);
        Ok(())
    }

    fn on_train_end(
        &mut self,
        outcome: TrainingOutcome,
        _model: Option<&NeuralNetwork>,
        eval: Option<&EvaluationSet>,
        epoch: usize,
    ) -> CallbackResult {
        self.bar.finish_and_clear();

        match outcome {
            TrainingOutcome::Completed => {
                completed!("Training completed");
                trace(&eval.expect("eval is present on completion").describe());
            }
            TrainingOutcome::EarlyStopped { restored } => {
                completed!("Early stopping triggered at epoch {epoch}");
                if restored {
                    trace("Restored the best model observed during training");
                }
                if let Some(eval) = eval {
                    trace(&eval.describe());
                }
            }
            TrainingOutcome::Diverged { recovered: true } => {
                warning!(
                    "Model diverged at epoch {epoch} (NaN/Inf); recovered best model from early stopping."
                );
                if let Some(eval) = eval {
                    trace(&eval.describe());
                }
            }
            TrainingOutcome::Diverged { recovered: false } => {}
        }

        Ok(())
    }
}

// ─── ModelSaver ─────────────────────────────────────────────────────────────

/// Saves the final model to disk once training ends, unless the run diverged
/// without recovery (in which case `model` is `None`).
pub struct ModelSaver {
    path: PathBuf,
}

impl ModelSaver {
    /// Saves the final model beside `run_dir`, in a file named `model_name`.
    /// Start and resume both construct the saver here, so the model can't land
    /// in two different places.
    pub fn new(run_dir: &Path, model_name: &str) -> Self {
        Self {
            path: run_dir.parent().unwrap_or(Path::new(".")).join(model_name),
        }
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
            saved(&Artifacts::single(
                "Neural Network",
                model.save(&self.path)?,
            ));
        }
        Ok(())
    }
}
