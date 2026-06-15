use super::recap::print_recap;
use crate::console::{Summary, completed, styled_bar, trace, warning};
use console::style;
use indicatif::ProgressBar;
use nrn::evaluation::EvaluationSet;
use nrn::model::NeuralNetwork;
use nrn::optimizers::Optimizer;
use nrn::schedulers::Scheduler;
use nrn::training::{HyperParameters, TrainerCallback, TrainingOutcome};
use std::io::Result;

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
    ) -> Result<()> {
        if let Some(scheduler) = scheduler {
            completed(&format!(
                "Restored {} scheduler state from checkpoint at epoch {epoch_start}",
                scheduler.name()
            ));
        }
        if let Some(optimizer) = optimizer {
            completed(&format!(
                "Restored {} optimizer state from checkpoint at epoch {epoch_start}",
                optimizer.name()
            ));
        }
        Ok(())
    }

    fn on_train_start(&mut self) -> Result<()> {
        self.bar.set_length(self.current.epochs() as u64);
        self.bar.set_message("Training");

        self.bar.suspend(|| {
            print_recap(&self.current, self.previous.as_ref());
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
