use crate::console::TRACE_ICON;
use console::style;
use nrn::io::hyperparams::{
    ClippingRecord, EarlyStoppingRecord, HyperParamsRecord, LossRecord, OptimizerRecord,
    SchedulerRecord,
};

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
/// field of `current`. If `previous` is given and a field differs from its
/// value in `previous`, the line is annotated `▲ was <previous value>`.
pub fn print_recap(current: &HyperParamsRecord, previous: Option<&HyperParamsRecord>) {
    let label_width = FIELDS.iter().map(|f| f.len()).max().unwrap_or(0);

    println!(
        "{} {}",
        style(TRACE_ICON).bright().blue(),
        style("TRAINING HYPERPARAMETERS").bold().blue()
    );

    print_row(
        "Epochs",
        &current.epochs.to_string(),
        label_width,
        previous
            .filter(|p| p.epochs != current.epochs)
            .map(|p| p.epochs.to_string()),
    );
    print_row("Loss", &loss_value(&current.loss), label_width, None);
    print_row(
        "Optimizer",
        &optimizer_value(current),
        label_width,
        previous
            .filter(|p| p.optimizer != current.optimizer || p.lr != current.lr)
            .map(optimizer_value),
    );
    print_row(
        "Scheduler",
        &scheduler_value(&current.scheduler),
        label_width,
        previous
            .filter(|p| p.scheduler != current.scheduler)
            .map(|p| scheduler_value(&p.scheduler)),
    );
    print_row(
        "Clipping",
        &clipping_value(&current.clipping),
        label_width,
        previous
            .filter(|p| p.clipping != current.clipping)
            .map(|p| clipping_value(&p.clipping)),
    );
    print_row(
        "Batches",
        &batches_value(current.batch_size),
        label_width,
        previous
            .filter(|p| p.batch_size != current.batch_size)
            .map(|p| batches_value(p.batch_size)),
    );
    print_row(
        "Split",
        &format!("val {} · test {}", current.val_ratio, current.test_ratio),
        label_width,
        None,
    );
    print_row(
        "Checkpoints",
        &checkpoints_value(current.checkpoint_interval),
        label_width,
        previous
            .filter(|p| p.checkpoint_interval != current.checkpoint_interval)
            .map(|p| checkpoints_value(p.checkpoint_interval)),
    );
    print_row(
        "Early stopping",
        &early_stopping_value(&current.early_stopping),
        label_width,
        previous
            .filter(|p| p.early_stopping != current.early_stopping)
            .map(|p| early_stopping_value(&p.early_stopping)),
    );
}

fn print_row(field: &'static str, value: &str, label_width: usize, was: Option<String>) {
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

pub(crate) fn loss_value(loss: &LossRecord) -> String {
    match loss {
        LossRecord::CrossEntropy => "Cross-Entropy".to_string(),
    }
}

pub(crate) fn optimizer_value(record: &HyperParamsRecord) -> String {
    let name = match record.optimizer {
        OptimizerRecord::Sgd => "Stochastic Gradient Descent (SGD)",
        OptimizerRecord::Adam => "Adam",
    };
    format!("{name} · lr {}", record.lr)
}

pub(crate) fn scheduler_value(scheduler: &SchedulerRecord) -> String {
    match scheduler {
        SchedulerRecord::Constant => "constant".to_string(),
        SchedulerRecord::Cosine { .. } => "cosine annealing".to_string(),
        SchedulerRecord::Step { .. } => "step decay".to_string(),
    }
}

pub(crate) fn clipping_value(clipping: &ClippingRecord) -> String {
    match clipping {
        ClippingRecord::None => "none".to_string(),
        ClippingRecord::Norm { max_norm } => format!("norm · max {max_norm}"),
        ClippingRecord::Value { min, max } => format!("value · min {min} · max {max}"),
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

pub(crate) fn early_stopping_value(early_stopping: &Option<EarlyStoppingRecord>) -> String {
    match early_stopping {
        Some(record) if record.restore_best_model => {
            format!("patience {} · restore best", record.patience)
        }
        Some(record) => format!("patience {}", record.patience),
        None => "disabled".to_string(),
    }
}
