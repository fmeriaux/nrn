use crate::console::{TRACE_ICON, completed, styled_bar, trace, warning};
use console::style;
use indicatif::ProgressBar;
use nrn::data::ModelSplit;
use nrn::evaluation::{Evaluation, EvaluationSet};
use nrn::model::NeuralNetwork;
use nrn::optimizers::Optimizer;
use nrn::schedulers::Scheduler;
use nrn::training::{
    CallbackResult, EarlyStoppingConfig, GradientClipping, HyperParameters, LossConfig,
    OptimizerConfig, SchedulerConfig, TrainerCallback, TrainingOutcome,
};

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

    fn on_train_start(&mut self, split: &ModelSplit) -> CallbackResult {
        self.bar.set_length(self.current.epochs() as u64);
        self.bar.set_message("Training");

        self.bar.suspend(|| {
            print_recap(&self.current, self.previous.as_ref());
            trace(&split_summary(split));
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
            TrainingOutcome::Completed => completed(&format!(
                "{} · {}",
                style("Training completed").bright().green(),
                eval_set_summary(eval.expect("eval is present on completion"))
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
                    trace(&eval_set_summary(eval));
                }
            }
            TrainingOutcome::Diverged { recovered: true } => {
                warning(&format!(
                    "Model diverged at epoch {} (NaN/Inf); recovered best model from early stopping.",
                    epoch
                ));
                if let Some(eval) = eval {
                    trace(&eval_set_summary(eval));
                }
            }
            TrainingOutcome::Diverged { recovered: false } => {}
        }

        Ok(())
    }
}

/// A recap row: a label and a renderer that formats the matching field of a
/// [`HyperParameters`] into its displayed value.
type Row = (&'static str, fn(&HyperParameters) -> String);

/// Recap rows in display order. [`print_recap`] iterates over these to align
/// the leader dots and to diff each field against `previous`.
const ROWS: &[Row] = &[
    ("Epochs", |h| h.epochs().to_string()),
    ("Loss", |h| loss_value(h.loss())),
    ("Optimizer", optimizer_value),
    ("Scheduler", |h| scheduler_value(h.scheduler())),
    ("Clipping", |h| clipping_value(h.clipping())),
    ("Batches", |h| batches_value(h.batch_size())),
    ("Split", |h| {
        format!("val {} · test {}", h.val_ratio(), h.test_ratio())
    }),
    ("Checkpoints", |h| {
        checkpoints_value(h.checkpoint_interval())
    }),
    ("Early stopping", |h| {
        early_stopping_value(h.early_stopping())
    }),
];

/// Prints the `TRAINING HYPERPARAMETERS` recap: one bullet-leader line per
/// field of `current`. If `previous` is given and a field renders differently
/// from its value in `previous`, the line is annotated `▲ was <previous value>`.
fn print_recap(current: &HyperParameters, previous: Option<&HyperParameters>) {
    let label_width = ROWS.iter().map(|(label, _)| label.len()).max().unwrap_or(0);

    println!(
        "{} {}",
        style(TRACE_ICON).bright().blue(),
        style("TRAINING HYPERPARAMETERS").bold().blue()
    );

    for (label, render) in ROWS {
        let value = render(current);
        let was = previous.map(render).filter(|prev| *prev != value);
        print_row(label, &value, label_width, was);
    }
}

fn print_row(field: &str, value: &str, label_width: usize, was: Option<String>) {
    let leader = ".".repeat(label_width + 3 - field.len());
    let mut line = format!(
        "   {} {} {}",
        style(field).cyan(),
        style(leader).dim(),
        style(value).yellow()
    );

    if let Some(was) = was {
        line.push_str(&format!("   {}", style(format!("▲ was {was}")).yellow()));
    }

    println!("{line}");
}

/// The realized per-split sample counts, complementing the recap's `Split` ratios.
fn split_summary(split: &ModelSplit) -> String {
    format!(
        "Samples · Train={} · Val={} · Test={}",
        style(split.train_size()).yellow(),
        style(split.validation_size()).yellow(),
        style(split.test_size()).yellow(),
    )
}

/// Formats an [`EvaluationSet`] as `Train(..) · Val(..) · Test(..)`, each split's
/// loss/accuracy via [`evaluation_summary`] and `N/A` for an absent validation split.
fn eval_set_summary(eval: &EvaluationSet) -> String {
    format!(
        "Train({}) · Val({}) · Test({})",
        evaluation_summary(&eval.train),
        eval.validation
            .as_ref()
            .map_or_else(|| "N/A".to_string(), evaluation_summary),
        evaluation_summary(&eval.test),
    )
}

/// Formats a single [`Evaluation`] as `L=<loss> · A=<accuracy>%`.
fn evaluation_summary(eval: &Evaluation) -> String {
    format!(
        "L={:.4} · A={:.1}{}",
        style(eval.loss).yellow(),
        style(eval.accuracy).yellow(),
        style("%").yellow(),
    )
}

fn loss_value(loss: &LossConfig) -> String {
    match loss {
        LossConfig::CrossEntropy => "Cross-Entropy".to_string(),
    }
}

fn optimizer_value(hyperparameters: &HyperParameters) -> String {
    let name = match hyperparameters.optimizer() {
        OptimizerConfig::Sgd => "Stochastic Gradient Descent (SGD)",
        OptimizerConfig::Adam => "Adam",
    };
    format!("{name} · lr {}", hyperparameters.lr().value())
}

fn scheduler_value(scheduler: &SchedulerConfig) -> String {
    match scheduler {
        SchedulerConfig::Constant => "constant".to_string(),
        SchedulerConfig::Cosine { .. } => "cosine annealing".to_string(),
        SchedulerConfig::Step { .. } => "step decay".to_string(),
    }
}

fn clipping_value(clipping: &GradientClipping) -> String {
    match clipping {
        GradientClipping::None => "none".to_string(),
        GradientClipping::Norm { max_norm } => format!("norm · max {max_norm}"),
        GradientClipping::Value { min, max } => format!("value · min {min} · max {max}"),
    }
}

fn batches_value(batch_size: Option<usize>) -> String {
    match batch_size {
        Some(size) => format!("{size}"),
        None => "full-batch".to_string(),
    }
}

fn checkpoints_value(checkpoint_interval: usize) -> String {
    if checkpoint_interval == 0 {
        "disabled".to_string()
    } else {
        format!("every {checkpoint_interval} epochs")
    }
}

fn early_stopping_value(early_stopping: Option<&EarlyStoppingConfig>) -> String {
    match early_stopping {
        Some(config) if config.restore_best_model() => {
            format!("patience {} · restore best", config.patience())
        }
        Some(config) => format!("patience {}", config.patience()),
        None => "disabled".to_string(),
    }
}
