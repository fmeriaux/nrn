use crate::display::{Summary, completed, trace, warning};
use console::style;
use nrn::callbacks::{TrainingCallback, TrainingOutcome};
use nrn::evaluation::EvaluationSet;
use nrn::training::TrainingConfig;
use std::io::Result;

/// Narrates the training run lifecycle on the console. Holds no state of its
/// own — everything it prints is derived from the hook arguments (`config` at
/// start, `outcome`/`eval`/`epoch` at end).
pub struct ConsoleReporter;

impl ConsoleReporter {
    pub fn new() -> Self {
        Self
    }
}

impl TrainingCallback for ConsoleReporter {
    fn on_train_start(&mut self, config: &TrainingConfig) -> Result<()> {
        trace(&format!(
            "Training for {} epochs",
            style(config.epochs).yellow()
        ));
        trace(&format!(
            "Using {} loss function",
            style(config.loss.name()).bold().blue()
        ));
        trace(&format!(
            "Using {} optimizer",
            style(config.optimizer.name()).bold().blue()
        ));
        trace(&format!(
            "Using {} scheduler",
            style(config.scheduler.name()).bold().blue()
        ));
        trace(&config.clipping.summary());

        match config.batch_size {
            Some(batch_size) => trace(&format!(
                "Using mini-batches of {} samples",
                style(batch_size).yellow()
            )),
            None => trace("Using full-batch gradient descent"),
        }

        if config.eval_interval > 0 {
            trace(&format!(
                "Recording a checkpoint every {} epochs",
                style(config.eval_interval).yellow()
            ));
        }

        Ok(())
    }

    fn on_train_end(
        &mut self,
        outcome: TrainingOutcome,
        eval: Option<&EvaluationSet>,
        epoch: usize,
    ) -> Result<()> {
        match outcome {
            TrainingOutcome::Completed => completed(&format!(
                "{} | {}",
                style("Training completed").bright().green(),
                eval.expect("eval is present on completion").summary()
            )),
            TrainingOutcome::EarlyStopped { restored } => {
                completed(&format!(
                    "Early stopping triggered at epoch {}",
                    style(epoch).yellow()
                ));
                if restored {
                    trace("Restored the best model observed during training");
                }
            }
            TrainingOutcome::Diverged { recovered: true } => warning(&format!(
                "Model diverged at epoch {} (NaN/Inf); recovered best model from early stopping.",
                epoch
            )),
            TrainingOutcome::Diverged { recovered: false } => {}
        }

        Ok(())
    }
}
