//! The `TRAINING HYPERPARAMETERS` recap. Unlike the entity verbs, this block is
//! built field-by-field so a resumed run can diff each value against its
//! previous setting and annotate the changed lines with `▲ was …`.

use super::{TRACE_ICON, label_width, row, theme};
use nrn::training::{
    EarlyStoppingConfig, GradientClipping, HyperParameters, LossConfig, OptimizerConfig,
    SchedulerConfig,
};

/// A recap row: a label and a renderer that formats the matching field of a
/// [`HyperParameters`] into its displayed value.
type Row = (&'static str, fn(&HyperParameters) -> String);

/// Recap rows in display order. [`print_recap`] iterates over these to align the
/// leader dots and to diff each field against `previous`.
const ROWS: &[Row] = &[
    ("Seed", |h| h.seed().to_string()),
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

/// Prints the `TRAINING HYPERPARAMETERS` recap: one dotted-leader line per field
/// of `current`. If `previous` is given and a field renders differently from its
/// value in `previous`, the line is annotated `▲ was <previous value>`.
pub(crate) fn print_recap(current: &HyperParameters, previous: Option<&HyperParameters>) {
    let width = label_width(ROWS.iter().map(|(label, _)| *label));

    println!(
        "{} {}",
        theme::icon(TRACE_ICON),
        theme::title("TRAINING HYPERPARAMETERS"),
    );

    for (label, render) in ROWS {
        let value = render(current);
        let was = previous.map(render).filter(|prev| *prev != value);
        println!("{}", row(label, &value, width, was.as_deref()));
    }
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
