//! The `TRAINING HYPERPARAMETERS` recap, rendered from a [`HyperParametersView`]:
//! a run's configuration field-by-field, and — when resuming — each value
//! compared against its previous setting, the changed lines annotated `▲ was …`.

use super::{Describe, Named, column_width, row};
use nrn::data::scalers::ScalerKind;
use nrn::training::{
    EarlyStoppingConfig, GradientClipping, HyperParameters, LossConfig, OptimizerConfig,
    SchedulerConfig,
};

/// A recap row: a label and a renderer that formats the matching field of a
/// [`HyperParameters`] into its displayed value.
type Row = (&'static str, fn(&HyperParameters) -> String);

/// Recap rows in display order. [`HyperParametersView::describe`] iterates over
/// these to align the leader dots and to diff each field against `previous`.
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
    ("Scaling", |h| scaling_value(h.scaler())),
    ("Checkpoints", |h| {
        checkpoints_value(h.checkpoint_interval())
    }),
    ("Early stopping", |h| {
        early_stopping_value(h.early_stopping())
    }),
];

/// A run's configuration, optionally paired with the previous one it resumes
/// from. Its description is the `TRAINING HYPERPARAMETERS` recap: a title line
/// then one dotted-leader line per field of `current`; with a `previous`, each
/// field that renders differently is annotated `▲ was <previous value>`.
pub(crate) struct HyperParametersView<'a> {
    pub current: &'a HyperParameters,
    pub previous: Option<&'a HyperParameters>,
}

impl Named for HyperParametersView<'_> {
    const NAME: &'static str = "TRAINING HYPERPARAMETERS";
}

impl Describe for HyperParametersView<'_> {
    fn describe(&self) -> String {
        let width = column_width(ROWS.iter().map(|(label, _)| *label));

        ROWS.iter()
            .map(|(label, render)| {
                let value = render(self.current);
                let was = self.previous.map(render).filter(|prev| *prev != value);
                row(label, &value, width, was.as_deref())
            })
            .collect::<Vec<_>>()
            .join("\n")
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

fn scaling_value(scaler: Option<ScalerKind>) -> String {
    match scaler {
        Some(kind) => kind.name().to_string(),
        None => "none".to_string(),
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
