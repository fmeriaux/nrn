use crate::console::{Summary, completed, styled_bar, trace, warning};
use console::style;
use indicatif::ProgressBar;
use nrn::evaluation::EvaluationSet;
use nrn::model::NeuralNetwork;
use nrn::training::{HyperParams, TrainingCallback, TrainingOutcome};
use std::io::Result;

/// Reports the training run lifecycle on the console: a progress bar tracks
/// epochs, and the run configuration / final outcome are narrated as trace
/// lines. The bar is suspended while printing so lines never get garbled by
/// its redraws, and cleared before the final summary.
pub struct ConsoleMonitor {
    bar: ProgressBar,
}

impl ConsoleMonitor {
    pub fn new() -> Self {
        Self { bar: styled_bar() }
    }
}

impl TrainingCallback for ConsoleMonitor {
    fn on_train_start(&mut self, hyperparams: &HyperParams) -> Result<()> {
        self.bar.set_length(hyperparams.epochs as u64);
        self.bar.set_message("Training");

        self.bar.suspend(|| {
            trace(&format!(
                "Training for {} epochs",
                style(hyperparams.epochs).yellow()
            ));
            trace(&format!(
                "Using {} loss function",
                style(hyperparams.loss.name()).bold().blue()
            ));
            trace(&format!(
                "Using {} optimizer",
                style(hyperparams.optimizer.name()).bold().blue()
            ));
            trace(&format!(
                "Using {} scheduler",
                style(hyperparams.scheduler.name()).bold().blue()
            ));
            trace(&hyperparams.clipping.summary());

            match hyperparams.batch_size {
                Some(batch_size) => trace(&format!(
                    "Using mini-batches of {} samples",
                    style(batch_size).yellow()
                )),
                None => trace("Using full-batch gradient descent"),
            }

            if hyperparams.checkpoint_interval > 0 {
                trace(&format!(
                    "Recording a checkpoint every {} epochs",
                    style(hyperparams.checkpoint_interval).yellow()
                ));
            }
        });

        Ok(())
    }

    fn on_epoch_end(&mut self, _epoch: usize) -> Result<()> {
        self.bar.inc(1);
        Ok(())
    }

    fn on_train_end(
        &mut self,
        outcome: TrainingOutcome,
        _model: Option<&NeuralNetwork>,
        eval: Option<&EvaluationSet>,
        epoch: usize,
    ) -> Result<()> {
        self.bar.finish_and_clear();

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
                if let Some(eval) = eval {
                    trace(&eval.summary());
                }
            }
            TrainingOutcome::Diverged { recovered: true } => {
                warning(&format!(
                    "Model diverged at epoch {} (NaN/Inf); recovered best model from early stopping.",
                    epoch
                ));
                if let Some(eval) = eval {
                    trace(&eval.summary());
                }
            }
            TrainingOutcome::Diverged { recovered: false } => {}
        }

        Ok(())
    }
}
