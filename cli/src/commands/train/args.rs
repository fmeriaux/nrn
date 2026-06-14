use clap::*;
use nrn::io::hyperparams::{
    ClippingRecord, EarlyStoppingRecord, HyperParametersRecord, SchedulerRecord,
};
use nrn::training::{
    EarlyStoppingConfig, EarlyStoppingConfigError, GradientClipping, GradientClippingError,
    HyperParameters, LearningRate, LossConfig, OptimizerConfig, SchedulerConfig,
};
use std::error::Error;

// ─── Value enums ─────────────────────────────────────────────────────────────

#[derive(ValueEnum, Debug, Copy, Clone)]
enum OptimizerType {
    Sgd,
    Adam,
}

#[derive(ValueEnum, Debug, Copy, Clone)]
enum SchedulerType {
    Constant,
    Cosine,
    Step,
}

// ─── Shared hyperparameters ───────────────────────────────────────────────────

#[derive(Args, Debug)]
pub struct TrainArgs {
    /// Number of epochs to train
    #[arg(short, long)]
    pub epochs: usize,

    /// Checkpoint interval (0 = no checkpoints)
    #[arg(short = 'k', long, default_value_t = 10)]
    pub checkpoint_interval: usize,

    /// Optimizer
    #[arg(long, value_enum, default_value_t = OptimizerType::Adam)]
    optimizer: OptimizerType,

    /// Learning rate scheduler
    #[arg(long, value_enum, default_value_t = SchedulerType::Constant)]
    scheduler: SchedulerType,

    /// Enable warm restarts for schedulers that support it (e.g., cosine)
    #[arg(long, requires = "scheduler", default_value_t = false)]
    warm_restarts: bool,

    /// Cycle multiplier for cosine scheduler with warm restarts
    #[arg(long, requires = "warm_restarts")]
    cycle_multiplier: Option<usize>,

    /// Decay factor for step scheduler
    #[arg(long, requires = "scheduler", default_value_t = 0.1)]
    decay_factor: f32,

    /// Step size for cosine/step scheduler (defaults to total epochs)
    #[arg(long, requires = "scheduler")]
    steps: Option<usize>,

    /// Learning rate
    #[arg(long, default_value_t = 0.001)]
    pub lr: f32,

    /// Minimum learning rate for schedulers that support it
    #[arg(long, requires = "scheduler")]
    lr_min: Option<f32>,

    /// Gradient clipping L2 norm
    #[arg(long, default_value_t = 1.0, conflicts_with_all = &["clip_value", "no_clip"])]
    clip_norm: f32,

    /// Element-wise gradient clipping value
    #[arg(long, conflicts_with_all = &["clip_norm", "no_clip"])]
    clip_value: Option<f32>,

    /// Disable gradient clipping
    #[arg(long, conflicts_with_all = &["clip_norm", "clip_value"])]
    no_clip: bool,

    /// Fraction of data for validation
    #[arg(long, default_value_t = 0.1)]
    val_ratio: f32,

    /// Fraction of data for test
    #[arg(long, default_value_t = 0.1)]
    test_ratio: f32,

    /// Early stopping patience (0 = disabled)
    #[arg(long, default_value_t = 0)]
    early_stopping: usize,

    /// Restore the best model when early stopping triggers
    #[arg(long, requires = "early_stopping", default_value_t = true)]
    restore_best_model: bool,

    /// Mini-batch size (omit for full-batch)
    #[arg(long)]
    batch_size: Option<usize>,
}

impl From<OptimizerType> for OptimizerConfig {
    fn from(optimizer: OptimizerType) -> Self {
        match optimizer {
            OptimizerType::Sgd => OptimizerConfig::Sgd,
            OptimizerType::Adam => OptimizerConfig::Adam,
        }
    }
}

impl From<&TrainArgs> for SchedulerConfig {
    fn from(args: &TrainArgs) -> Self {
        let steps = args.steps.unwrap_or(args.epochs);
        match args.scheduler {
            SchedulerType::Constant => SchedulerConfig::Constant,
            SchedulerType::Cosine => SchedulerConfig::Cosine {
                lr_min: args.lr_min.unwrap_or(0.0),
                steps,
                warm_restarts: args.warm_restarts,
                cycle_multiplier: args.cycle_multiplier.unwrap_or(1),
            },
            SchedulerType::Step => SchedulerConfig::Step {
                decay_factor: args.decay_factor,
                steps,
            },
        }
    }
}

impl TryFrom<&TrainArgs> for GradientClipping {
    type Error = GradientClippingError;

    fn try_from(args: &TrainArgs) -> Result<Self, Self::Error> {
        if args.no_clip {
            Ok(GradientClipping::None)
        } else if let Some(value) = args.clip_value {
            GradientClipping::value(-value, value)
        } else {
            GradientClipping::norm(args.clip_norm)
        }
    }
}

impl TrainArgs {
    /// The early-stopping config, or `None` when patience is zero (disabled).
    fn early_stopping(&self) -> Result<Option<EarlyStoppingConfig>, EarlyStoppingConfigError> {
        if self.early_stopping > 0 {
            Ok(Some(EarlyStoppingConfig::new(
                self.early_stopping,
                self.restore_best_model,
            )?))
        } else {
            Ok(None)
        }
    }
}

impl TryFrom<&TrainArgs> for HyperParameters {
    type Error = Box<dyn Error>;

    /// Assembles the CLI arguments into a validated domain spec, surfacing any
    /// invalid value (non-positive learning rate, bad clipping bounds,
    /// cross-field invariant violations) as an error.
    fn try_from(args: &TrainArgs) -> Result<Self, Self::Error> {
        Ok(HyperParameters::new(
            args.epochs,
            args.checkpoint_interval,
            args.batch_size,
            LearningRate::new(args.lr)?,
            args.optimizer.into(),
            SchedulerConfig::from(args),
            GradientClipping::try_from(args)?,
            LossConfig::CrossEntropy,
            args.early_stopping()?,
            args.val_ratio,
            args.test_ratio,
        )?)
    }
}

// ─── Resume overrides ─────────────────────────────────────────────────────────

/// Overridable hyperparameters at `train resume`. Unlike `train start`, the
/// dataset, model architecture, and split ratios are fixed by the run's
/// metadata and cannot be changed here. Changing the optimizer or scheduler
/// *algorithm* (as opposed to its learning rate / decay parameters) also
/// isn't supported on resume — start a new run for that, since the restored
/// optimizer/scheduler state (moment estimates, step counters) is only
/// meaningful for the algorithm that produced it.
#[derive(Args, Debug)]
pub struct ResumeOverrides {
    /// Override: number of epochs to train
    #[arg(short, long)]
    epochs: Option<usize>,

    /// Override: checkpoint interval (0 = no checkpoints)
    #[arg(short = 'k', long)]
    checkpoint_interval: Option<usize>,

    /// Override: learning rate
    #[arg(long)]
    lr: Option<f32>,

    /// Override: minimum learning rate (only applies if the run uses a cosine scheduler)
    #[arg(long)]
    lr_min: Option<f32>,

    /// Override: gradient clipping L2 norm
    #[arg(long, conflicts_with_all = &["clip_value", "no_clip"])]
    clip_norm: Option<f32>,

    /// Override: element-wise gradient clipping value
    #[arg(long, conflicts_with_all = &["clip_norm", "no_clip"])]
    clip_value: Option<f32>,

    /// Override: disable gradient clipping
    #[arg(long, conflicts_with_all = &["clip_norm", "clip_value"])]
    no_clip: bool,

    /// Override: early stopping patience (0 = disabled)
    #[arg(long)]
    early_stopping: Option<usize>,

    /// Override: restore the best model when early stopping triggers
    #[arg(long)]
    restore_best_model: Option<bool>,

    /// Override: mini-batch size
    #[arg(long, conflicts_with = "full_batch")]
    batch_size: Option<usize>,

    /// Override: use full-batch gradient descent
    #[arg(long, conflicts_with = "batch_size")]
    full_batch: bool,
}

impl ResumeOverrides {
    /// Applies the requested overrides onto `record` in place. Overrides operate
    /// on the serialized record (the run's persisted form) before it is
    /// validated back into [`HyperParameters`].
    pub fn apply(&self, record: &mut HyperParametersRecord) {
        if let Some(epochs) = self.epochs {
            record.epochs = epochs;
        }

        if let Some(checkpoint_interval) = self.checkpoint_interval {
            record.checkpoint_interval = checkpoint_interval;
        }

        if let Some(lr) = self.lr {
            record.lr = lr;
        }

        if let Some(lr_min) = self.lr_min
            && let SchedulerRecord::Cosine {
                lr_min: current_lr_min,
                ..
            } = &mut record.scheduler
        {
            *current_lr_min = lr_min;
        }

        let new_clipping = if self.no_clip {
            Some(ClippingRecord::None)
        } else if let Some(value) = self.clip_value {
            Some(ClippingRecord::Value {
                min: -value,
                max: value,
            })
        } else {
            self.clip_norm
                .map(|max_norm| ClippingRecord::Norm { max_norm })
        };
        if let Some(clipping) = new_clipping {
            record.clipping = clipping;
        }

        if self.early_stopping.is_some() || self.restore_best_model.is_some() {
            let patience = self
                .early_stopping
                .unwrap_or_else(|| record.early_stopping.as_ref().map_or(0, |es| es.patience));
            let restore_best_model = self.restore_best_model.unwrap_or_else(|| {
                record
                    .early_stopping
                    .as_ref()
                    .is_none_or(|es| es.restore_best_model)
            });
            record.early_stopping = if patience > 0 {
                Some(EarlyStoppingRecord {
                    patience,
                    restore_best_model,
                })
            } else {
                None
            };
        }

        if self.full_batch || self.batch_size.is_some() {
            record.batch_size = if self.full_batch {
                None
            } else {
                self.batch_size
            };
        }
    }
}
