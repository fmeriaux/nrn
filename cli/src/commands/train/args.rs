use clap::*;
use ndarray_rand::rand::random;
use nrn::data::scalers::ScalerKind;
use nrn::io::model::hyperparams::{
    ClippingRecord, EarlyStoppingRecord, HyperParametersRecord, SchedulerRecord,
};
use nrn::task::Task;
use nrn::training::{
    EarlyStoppingConfig, GradientClipping, GradientClippingError, HyperParameters,
    HyperParametersError, LossConfig, OptimizerConfig, SchedulerConfig,
};

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

#[derive(ValueEnum, Debug, Copy, Clone)]
enum ScaleType {
    MinMax,
    ZScore,
}

impl From<ScaleType> for ScalerKind {
    fn from(scale: ScaleType) -> Self {
        match scale {
            ScaleType::MinMax => ScalerKind::MinMax,
            ScaleType::ZScore => ScalerKind::ZScore,
        }
    }
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

    /// Weight decay (Adam: decoupled AdamW; SGD: L2). 0 = disabled
    #[arg(long, default_value_t = 0.0)]
    weight_decay: f32,

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

    /// Disable restoring the best model when early stopping triggers
    #[arg(long, requires = "early_stopping")]
    no_restore_best_model: bool,

    /// Mini-batch size (omit for full-batch)
    #[arg(long)]
    batch_size: Option<usize>,

    /// Input scaler fitted on the train split and applied to all splits (omit for none)
    #[arg(long, value_enum)]
    scale: Option<ScaleType>,

    /// Seed for weight initialization and mini-batch shuffling (omit for a random,
    /// reported seed). Recorded so the run is reproducible and resumable.
    #[arg(long)]
    pub seed: Option<u64>,
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
    fn early_stopping(&self) -> Option<EarlyStoppingConfig> {
        EarlyStoppingConfig::new(self.early_stopping, !self.no_restore_best_model).ok()
    }

    /// Assembles the CLI arguments into a validated domain spec, surfacing any
    /// invalid value (non-positive learning rate, bad clipping bounds,
    /// cross-field invariant violations) as a single [`HyperParametersError`].
    /// The caller-specific clipping and early-stopping components are built from
    /// the arg shapes here, their errors folding into the same type via `?`. The
    /// loss is derived from `task`.
    pub fn to_hyperparameters(&self, task: &Task) -> Result<HyperParameters, HyperParametersError> {
        HyperParameters::from_values(
            self.epochs,
            self.checkpoint_interval,
            self.batch_size,
            self.lr,
            self.weight_decay,
            self.optimizer.into(),
            SchedulerConfig::from(self),
            GradientClipping::try_from(self)?,
            LossConfig::for_task(task),
            self.early_stopping(),
            self.val_ratio,
            self.test_ratio,
            self.seed.unwrap_or_else(random),
            self.scale.map(Into::into),
        )
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

    /// Override: weight decay (0 = disabled)
    #[arg(long)]
    weight_decay: Option<f32>,

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

        if let Some(weight_decay) = self.weight_decay {
            record.weight_decay = weight_decay;
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

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use nrn::io::model::hyperparams::{
        LossKindRecord, LossRecord, OptimizerRecord, ReductionRecord,
    };

    // ─── TrainArgs conversions ─────────────────────────────────────────────────

    /// Parser wrapper exercising the flattened [`TrainArgs`] through clap so
    /// defaults and the conversion impls are covered in one path.
    #[derive(Parser)]
    struct TrainCli {
        #[command(flatten)]
        args: TrainArgs,
    }

    fn train_args(extra: &[&str]) -> TrainArgs {
        let mut argv = vec!["train", "--epochs", "100"];
        argv.extend_from_slice(extra);
        TrainCli::parse_from(argv).args
    }

    #[test]
    fn optimizer_defaults_to_adam() {
        assert_eq!(
            OptimizerConfig::from(train_args(&[]).optimizer),
            OptimizerConfig::Adam
        );
    }

    #[test]
    fn optimizer_sgd_selected() {
        assert_eq!(
            OptimizerConfig::from(train_args(&["--optimizer", "sgd"]).optimizer),
            OptimizerConfig::Sgd
        );
    }

    #[test]
    fn scheduler_defaults_to_constant() {
        assert_eq!(
            SchedulerConfig::from(&train_args(&[])),
            SchedulerConfig::Constant
        );
    }

    #[test]
    fn cosine_steps_default_to_total_epochs() {
        assert_eq!(
            SchedulerConfig::from(&train_args(&["--scheduler", "cosine"])),
            SchedulerConfig::Cosine {
                lr_min: 0.0,
                steps: 100,
                warm_restarts: false,
                cycle_multiplier: 1,
            }
        );
    }

    #[test]
    fn cosine_carries_warm_restart_knobs() {
        let args = train_args(&[
            "--scheduler",
            "cosine",
            "--lr-min",
            "0.0001",
            "--steps",
            "20",
            "--warm-restarts",
            "--cycle-multiplier",
            "2",
        ]);
        assert_eq!(
            SchedulerConfig::from(&args),
            SchedulerConfig::Cosine {
                lr_min: 0.0001,
                steps: 20,
                warm_restarts: true,
                cycle_multiplier: 2,
            }
        );
    }

    #[test]
    fn step_scheduler_carries_decay_factor() {
        let args = train_args(&[
            "--scheduler",
            "step",
            "--decay-factor",
            "0.5",
            "--steps",
            "30",
        ]);
        assert_eq!(
            SchedulerConfig::from(&args),
            SchedulerConfig::Step {
                decay_factor: 0.5,
                steps: 30,
            }
        );
    }

    #[test]
    fn clipping_defaults_to_norm() {
        assert_eq!(
            GradientClipping::try_from(&train_args(&[])).unwrap(),
            GradientClipping::Norm { max_norm: 1.0 }
        );
    }

    #[test]
    fn clip_value_is_symmetric() {
        assert_eq!(
            GradientClipping::try_from(&train_args(&["--clip-value", "0.5"])).unwrap(),
            GradientClipping::Value {
                min: -0.5,
                max: 0.5
            }
        );
    }

    #[test]
    fn no_clip_disables_clipping() {
        assert_eq!(
            GradientClipping::try_from(&train_args(&["--no-clip"])).unwrap(),
            GradientClipping::None
        );
    }

    #[test]
    fn invalid_clip_norm_surfaces_error() {
        assert!(GradientClipping::try_from(&train_args(&["--clip-norm", "0"])).is_err());
    }

    #[test]
    fn early_stopping_disabled_when_zero() {
        assert_eq!(train_args(&[]).early_stopping(), None);
    }

    #[test]
    fn early_stopping_enabled_with_patience() {
        let config = train_args(&["--early-stopping", "5"])
            .early_stopping()
            .expect("early stopping enabled");
        assert_eq!(config.patience(), 5);
        assert!(config.restore_best_model());
    }

    #[test]
    fn no_restore_best_model_disables_restoration() {
        let config = train_args(&["--early-stopping", "5", "--no-restore-best-model"])
            .early_stopping()
            .expect("early stopping enabled");
        assert!(!config.restore_best_model());
    }

    #[test]
    fn hyperparameters_assembled_from_args() {
        let args = train_args(&["--lr", "0.01", "--batch-size", "32", "--seed", "42"]);
        let hp = args
            .to_hyperparameters(&Task::Binary)
            .expect("valid hyperparameters");
        assert_eq!(hp.epochs(), 100);
        assert_eq!(hp.lr().value(), 0.01);
        assert_eq!(hp.batch_size(), Some(32));
        assert_eq!(hp.seed(), 42);
        assert_eq!(hp.optimizer(), &OptimizerConfig::Adam);
    }

    #[test]
    fn weight_decay_defaults_to_zero_and_is_parsed() {
        assert_eq!(
            train_args(&[])
                .to_hyperparameters(&Task::Binary)
                .unwrap()
                .weight_decay()
                .value(),
            0.0
        );
        assert_eq!(
            train_args(&["--weight-decay", "0.0001"])
                .to_hyperparameters(&Task::Binary)
                .unwrap()
                .weight_decay()
                .value(),
            0.0001
        );
    }

    #[test]
    fn scale_maps_to_the_matching_scaler_kind() {
        assert_eq!(
            train_args(&[])
                .to_hyperparameters(&Task::Binary)
                .unwrap()
                .scaler(),
            None
        );
        assert_eq!(
            train_args(&["--scale", "min-max"])
                .to_hyperparameters(&Task::Binary)
                .unwrap()
                .scaler(),
            Some(ScalerKind::MinMax)
        );
        assert_eq!(
            train_args(&["--scale", "z-score"])
                .to_hyperparameters(&Task::Binary)
                .unwrap()
                .scaler(),
            Some(ScalerKind::ZScore)
        );
    }

    #[test]
    fn hyperparameters_surface_validation_errors() {
        // Validation/test ratios summing past 1.0 is a cross-field invariant violation.
        let args = train_args(&["--val-ratio", "0.8", "--test-ratio", "0.8"]);
        assert!(args.to_hyperparameters(&Task::Binary).is_err());
    }

    // ─── ResumeOverrides ───────────────────────────────────────────────────────

    #[derive(Parser)]
    struct ResumeCli {
        #[command(flatten)]
        overrides: ResumeOverrides,
    }

    fn overrides(extra: &[&str]) -> ResumeOverrides {
        let mut argv = vec!["resume"];
        argv.extend_from_slice(extra);
        ResumeCli::parse_from(argv).overrides
    }

    fn base_record() -> HyperParametersRecord {
        HyperParametersRecord {
            epochs: 100,
            checkpoint_interval: 10,
            batch_size: Some(32),
            lr: 0.001,
            weight_decay: 0.0,
            optimizer: OptimizerRecord::Adam,
            scheduler: SchedulerRecord::Constant,
            clipping: ClippingRecord::Norm { max_norm: 1.0 },
            early_stopping: None,
            val_ratio: 0.1,
            test_ratio: 0.1,
            loss: LossRecord {
                kind: LossKindRecord::BinaryCrossEntropy,
                reduction: ReductionRecord::Mean,
            },
            seed: 42,
            scaler: None,
        }
    }

    #[test]
    fn empty_overrides_leave_record_unchanged() {
        let mut record = base_record();
        overrides(&[]).apply(&mut record);
        assert_eq!(record, base_record());
    }

    #[test]
    fn scalar_overrides_applied() {
        let mut record = base_record();
        overrides(&[
            "--epochs",
            "200",
            "--checkpoint-interval",
            "5",
            "--lr",
            "0.01",
            "--weight-decay",
            "0.0005",
        ])
        .apply(&mut record);
        assert_eq!(record.epochs, 200);
        assert_eq!(record.checkpoint_interval, 5);
        assert_eq!(record.lr, 0.01);
        assert_eq!(record.weight_decay, 0.0005);
    }

    #[test]
    fn lr_min_override_only_affects_cosine() {
        // Constant scheduler: lr_min has nowhere to land, record stays put.
        let mut constant = base_record();
        overrides(&["--lr-min", "0.0001"]).apply(&mut constant);
        assert_eq!(constant.scheduler, SchedulerRecord::Constant);

        // Cosine scheduler: lr_min is patched in place.
        let mut cosine = base_record();
        cosine.scheduler = SchedulerRecord::Cosine {
            lr_min: 0.0,
            steps: 100,
            warm_restarts: false,
            cycle_multiplier: 1,
        };
        overrides(&["--lr-min", "0.0001"]).apply(&mut cosine);
        assert_eq!(
            cosine.scheduler,
            SchedulerRecord::Cosine {
                lr_min: 0.0001,
                steps: 100,
                warm_restarts: false,
                cycle_multiplier: 1,
            }
        );
    }

    #[test]
    fn clipping_overrides_applied() {
        let mut value = base_record();
        overrides(&["--clip-value", "0.5"]).apply(&mut value);
        assert_eq!(
            value.clipping,
            ClippingRecord::Value {
                min: -0.5,
                max: 0.5
            }
        );

        let mut none = base_record();
        overrides(&["--no-clip"]).apply(&mut none);
        assert_eq!(none.clipping, ClippingRecord::None);

        let mut norm = base_record();
        overrides(&["--clip-norm", "2.0"]).apply(&mut norm);
        assert_eq!(norm.clipping, ClippingRecord::Norm { max_norm: 2.0 });
    }

    #[test]
    fn early_stopping_can_be_enabled_and_disabled() {
        let mut enabled = base_record();
        overrides(&["--early-stopping", "5"]).apply(&mut enabled);
        assert_eq!(
            enabled.early_stopping,
            Some(EarlyStoppingRecord {
                patience: 5,
                restore_best_model: true,
            })
        );

        // Patience 0 clears an existing config.
        let mut disabled = base_record();
        disabled.early_stopping = Some(EarlyStoppingRecord {
            patience: 5,
            restore_best_model: true,
        });
        overrides(&["--early-stopping", "0"]).apply(&mut disabled);
        assert_eq!(disabled.early_stopping, None);
    }

    #[test]
    fn restore_best_model_override_reuses_existing_patience() {
        let mut record = base_record();
        record.early_stopping = Some(EarlyStoppingRecord {
            patience: 7,
            restore_best_model: true,
        });
        overrides(&["--restore-best-model", "false"]).apply(&mut record);
        assert_eq!(
            record.early_stopping,
            Some(EarlyStoppingRecord {
                patience: 7,
                restore_best_model: false,
            })
        );
    }

    #[test]
    fn batch_size_and_full_batch_overrides() {
        let mut sized = base_record();
        overrides(&["--batch-size", "64"]).apply(&mut sized);
        assert_eq!(sized.batch_size, Some(64));

        let mut full = base_record();
        overrides(&["--full-batch"]).apply(&mut full);
        assert_eq!(full.batch_size, None);
    }
}
