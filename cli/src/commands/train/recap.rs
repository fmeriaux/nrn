use crate::console::TRACE_ICON;
use console::style;
use nrn::io::hyperparams::{
    ClippingRecord, EarlyStoppingRecord, HyperParamsRecord, OptimizerRecord, SchedulerRecord,
};
use nrn::training::{EarlyStoppingConfig, GradientClipping, HyperParams};

/// A hyperparameter that was changed at `train resume` relative to the run's
/// stored metadata. Purely informational: it annotates the recap line for
/// `field` with the value that was in the metadata before the override.
pub struct FieldOverride {
    pub field: &'static str,
    pub meta_value: String,
}

/// Recap row labels, in display order. [`print_recap`] iterates over these to
/// compute the leader-dot alignment.
const FIELDS: &[&str] = &[
    "Epochs",
    "Loss",
    "Optimizer",
    "Scheduler",
    "Clipping",
    "Batches",
    "Split",
    "Checkpoints",
    "Early stopping",
];

/// Prints the `TRAINING HYPERPARAMETERS` recap: one bullet-leader line per
/// field, with overridden fields annotated `▲ was <meta_value>`.
pub fn print_recap(hyperparams: &HyperParams, overrides: &[FieldOverride]) {
    let label_width = FIELDS.iter().map(|f| f.len()).max().unwrap_or(0);

    println!(
        "{} {}",
        style(TRACE_ICON).bright().blue(),
        style("TRAINING HYPERPARAMETERS").bold().blue()
    );

    print_row(
        "Epochs",
        &hyperparams.epochs.to_string(),
        overrides,
        label_width,
    );
    print_row("Loss", hyperparams.loss.name(), overrides, label_width);
    print_row(
        "Optimizer",
        &optimizer_value(hyperparams),
        overrides,
        label_width,
    );
    print_row(
        "Scheduler",
        &hyperparams.scheduler.name().to_lowercase(),
        overrides,
        label_width,
    );
    print_row(
        "Clipping",
        &clipping_value(&hyperparams.clipping),
        overrides,
        label_width,
    );
    print_row(
        "Batches",
        &batches_value(hyperparams.batch_size),
        overrides,
        label_width,
    );
    print_row(
        "Split",
        &format!(
            "val {} · test {}",
            hyperparams.val_ratio, hyperparams.test_ratio
        ),
        overrides,
        label_width,
    );
    print_row(
        "Checkpoints",
        &checkpoints_value(hyperparams.checkpoint_interval),
        overrides,
        label_width,
    );
    print_row(
        "Early stopping",
        &early_stopping_value(&hyperparams.early_stopping),
        overrides,
        label_width,
    );
}

fn print_row(field: &'static str, value: &str, overrides: &[FieldOverride], label_width: usize) {
    let leader = ".".repeat(label_width + 3 - field.len());
    let mut line = format!(
        "   {} {} {}",
        style(field).cyan(),
        style(leader).dim(),
        style(value).yellow()
    );

    if let Some(over) = overrides.iter().find(|over| over.field == field) {
        line.push_str(&format!(
            "   {}",
            style(format!("▲ was {}", over.meta_value)).yellow()
        ));
    }

    println!("{line}");
}

pub(crate) fn optimizer_value(hyperparams: &HyperParams) -> String {
    format!(
        "{} · lr {}",
        hyperparams.optimizer.name(),
        hyperparams.optimizer.learning_rate().value()
    )
}

pub(crate) fn clipping_value(clipping: &GradientClipping) -> String {
    match clipping {
        GradientClipping::None => "none".to_string(),
        GradientClipping::Norm { max_norm } => format!("norm · max {max_norm}"),
        GradientClipping::Value { min, max } => format!("value · min {min} · max {max}"),
    }
}

pub(crate) fn batches_value(batch_size: Option<usize>) -> String {
    match batch_size {
        Some(size) => format!("{size}"),
        None => "full-batch".to_string(),
    }
}

pub(crate) fn checkpoints_value(checkpoint_interval: usize) -> String {
    if checkpoint_interval == 0 {
        "disabled".to_string()
    } else {
        format!("every {checkpoint_interval} epochs")
    }
}

pub(crate) fn early_stopping_value(early_stopping: &Option<EarlyStoppingConfig>) -> String {
    match early_stopping {
        Some(config) if config.restore_best_model => {
            format!("patience {} · restore best", config.patience)
        }
        Some(config) => format!("patience {}", config.patience),
        None => "disabled".to_string(),
    }
}

/// Renders the `Optimizer` recap value (`<name> · lr <rate>`) from a stored record,
/// used to capture the metadata value before a `--lr` override is applied.
pub(crate) fn optimizer_record_value(record: &HyperParamsRecord) -> String {
    let name = match record.optimizer {
        OptimizerRecord::Sgd => "Stochastic Gradient Descent (SGD)",
        OptimizerRecord::Adam => "Adam",
    };
    format!("{name} · lr {}", record.lr)
}

/// Renders the `Scheduler` recap value from a stored record, used to capture
/// the metadata value before a `--lr-min` override is applied.
pub(crate) fn scheduler_record_value(scheduler: &SchedulerRecord) -> String {
    match scheduler {
        SchedulerRecord::Constant => "constant".to_string(),
        SchedulerRecord::Cosine { .. } => "cosine annealing".to_string(),
        SchedulerRecord::Step { .. } => "step decay".to_string(),
    }
}

/// Renders the `Clipping` recap value from a stored record, used to capture
/// the metadata value before a `--clip-norm`/`--clip-value`/`--no-clip` override is applied.
pub(crate) fn clipping_record_value(clipping: &ClippingRecord) -> String {
    match clipping {
        ClippingRecord::None => "none".to_string(),
        ClippingRecord::Norm { max_norm } => format!("norm · max {max_norm}"),
        ClippingRecord::Value { min, max } => format!("value · min {min} · max {max}"),
    }
}

/// Renders the `Early stopping` recap value from a stored record, used to
/// capture the metadata value before an `--early-stopping`/`--restore-best-model`
/// override is applied.
pub(crate) fn early_stopping_record_value(early_stopping: &Option<EarlyStoppingRecord>) -> String {
    match early_stopping {
        Some(record) if record.restore_best_model => {
            format!("patience {} · restore best", record.patience)
        }
        Some(record) => format!("patience {}", record.patience),
        None => "disabled".to_string(),
    }
}
