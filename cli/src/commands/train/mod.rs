mod model_saver;
mod monitor;
mod recap;

use crate::actions::*;
use crate::console::{RUN_ICON, Summary, completed, loaded, recording_at, warning};
use clap::*;
use model_saver::ModelSaver;
use monitor::ConsoleMonitor;
use nrn::data::ModelSplit;
use nrn::io::checkpoint::CheckpointRecorder;
use nrn::io::hyperparams::{
    ClippingRecord, EarlyStoppingRecord, HyperParamsRecord, LossRecord, OptimizerRecord,
    SchedulerRecord,
};
use nrn::io::run::{TrainingMeta, TrainingRun};
use nrn::model::NeuralNetwork;
use nrn::training::{Callbacks, FatalDivergence, HyperParams, TrainingLoop};
use std::error::Error;
use std::fmt::Display;
use std::io::{Error as IoError, ErrorKind, Result as IoResult};
use std::path::{Path, PathBuf};

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

impl Display for OptimizerType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OptimizerType::Sgd => write!(f, "Stochastic Gradient Descent (SGD)"),
            OptimizerType::Adam => write!(f, "Adam"),
        }
    }
}

impl Display for SchedulerType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SchedulerType::Constant => write!(f, "Constant"),
            SchedulerType::Cosine => write!(f, "Cosine Annealing"),
            SchedulerType::Step => write!(f, "Step Decay"),
        }
    }
}

// ─── TrainCommand ─────────────────────────────────────────────────────────────

#[derive(Subcommand, Debug)]
pub enum TrainCommand {
    /// Start a new training run from scratch
    Start(StartArgs),
    /// Resume training from an existing run directory
    Resume(ResumeArgs),
}

impl TrainCommand {
    pub fn run(&self) -> Result<(), Box<dyn Error>> {
        match self {
            TrainCommand::Start(args) => args.run(),
            TrainCommand::Resume(args) => args.run(),
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

impl TrainArgs {
    /// Maps these CLI arguments onto a [`HyperParamsRecord`], the serializable
    /// counterpart reconstructed (and validated) via
    /// [`HyperParamsRecord::into_hyperparams`].
    fn to_record(&self) -> HyperParamsRecord {
        let optimizer = match self.optimizer {
            OptimizerType::Sgd => OptimizerRecord::Sgd,
            OptimizerType::Adam => OptimizerRecord::Adam,
        };

        let steps = self.steps.unwrap_or(self.epochs);

        let scheduler = match self.scheduler {
            SchedulerType::Constant => SchedulerRecord::Constant,
            SchedulerType::Cosine => SchedulerRecord::Cosine {
                lr_min: self.lr_min.unwrap_or(0.0),
                steps,
                warm_restarts: self.warm_restarts,
                cycle_multiplier: self.cycle_multiplier.unwrap_or(1),
            },
            SchedulerType::Step => SchedulerRecord::Step {
                decay_factor: self.decay_factor,
                steps,
            },
        };

        let clipping = if self.no_clip {
            ClippingRecord::None
        } else if let Some(value) = self.clip_value {
            ClippingRecord::Value {
                min: -value,
                max: value,
            }
        } else {
            ClippingRecord::Norm {
                max_norm: self.clip_norm,
            }
        };

        let early_stopping = if self.early_stopping > 0 {
            Some(EarlyStoppingRecord {
                patience: self.early_stopping,
                restore_best_model: self.restore_best_model,
            })
        } else {
            None
        };

        HyperParamsRecord {
            epochs: self.epochs,
            checkpoint_interval: self.checkpoint_interval,
            batch_size: self.batch_size,
            lr: self.lr,
            optimizer,
            scheduler,
            clipping,
            early_stopping,
            val_ratio: self.val_ratio,
            test_ratio: self.test_ratio,
            loss: LossRecord::CrossEntropy,
        }
    }
}

// ─── Start ────────────────────────────────────────────────────────────────────

#[derive(Args, Debug)]
pub struct StartArgs {
    /// Dataset to train on
    dataset: String,

    /// Load a pre-trained model to continue training (checkpoint count resets to 0)
    #[arg(short, long, conflicts_with_all = &["layers", "auto_layers"])]
    model: Option<String>,

    /// Hidden layer sizes (comma-separated)
    #[arg(long, value_delimiter = ',', conflicts_with_all = &["auto_layers", "model"])]
    layers: Option<Vec<usize>>,

    /// Infer hidden layers from dataset characteristics
    #[arg(long, conflicts_with_all = &["layers", "model"], default_value_t = false)]
    auto_layers: bool,

    /// Overwrite an existing training run directory
    #[arg(long, default_value_t = false)]
    overwrite: bool,

    #[command(flatten)]
    hp: TrainArgs,
}

impl StartArgs {
    pub fn run(&self) -> Result<(), Box<dyn Error>> {
        if let Some(ref layers) = self.layers
            && layers.contains(&0)
        {
            return Err("Each hidden layer must have at least one neuron.".into());
        }

        let dataset_name = get_file_stem(Path::new(&self.dataset));
        let dataset_path = Path::new(&self.dataset);
        let dataset = load_dataset(&self.dataset)?;
        dataset.validate()?;

        let record = self.hp.to_record();
        let hyperparams = record.into_hyperparams()?;

        let split = dataset
            .to_model_dataset()
            .split(record.val_ratio, record.test_ratio);
        completed(split.summary().as_str());

        let model = match &self.model {
            Some(path) => load_model(path)?,
            None => initialize_model_with(&dataset, self.layers.clone(), self.auto_layers),
        };

        let run_dir = dataset_path.with_file_name(format!("training-model-{dataset_name}"));
        let model_save_path = dataset_path.with_file_name(format!("model-{dataset_name}"));

        let recorder = if record.checkpoint_interval > 0 {
            let recorder =
                create_checkpoint_recorder(&run_dir, &dataset_name, &record, self.overwrite)?;
            recording_at(RUN_ICON, "TRAINING RUN", &run_dir);
            Some(recorder)
        } else {
            None
        };

        execute_training(
            hyperparams,
            model,
            split,
            0,
            model_save_path,
            recorder,
            record,
            None,
        )
    }
}

// ─── Resume ───────────────────────────────────────────────────────────────────

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
    /// Applies the requested overrides onto `record` in place.
    fn apply(&self, record: &mut HyperParamsRecord) {
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

#[derive(Args, Debug)]
pub struct ResumeArgs {
    /// Training run directory to resume from
    run_dir: String,

    /// Checkpoint index to resume from (default: last checkpoint)
    #[arg(long)]
    from: Option<usize>,

    #[command(flatten)]
    overrides: ResumeOverrides,
}

impl ResumeArgs {
    pub fn run(&self) -> Result<(), Box<dyn Error>> {
        let run_dir = Path::new(&self.run_dir);
        let run = TrainingRun::open(run_dir)?;
        let meta = run.meta();

        let dataset = load_dataset(&meta.dataset)?;
        dataset.validate()?;

        let previous = meta.hyperparams.clone();
        let mut record = meta.hyperparams.clone();
        self.overrides.apply(&mut record);
        let mut hyperparams = record.into_hyperparams()?;

        let split = dataset
            .to_model_dataset()
            .split(record.val_ratio, record.test_ratio);
        completed(split.summary().as_str());

        let archive = run.archive()?;
        if archive.is_empty() {
            return Err(format!(
                "No checkpoints found in '{}'; cannot resume.",
                run_dir.display()
            )
            .into());
        }

        let checkpoint_idx = match self.from {
            Some(idx) => {
                if idx >= archive.len() {
                    return Err(format!(
                        "Checkpoint index {idx} out of range (run has {} checkpoints).",
                        archive.len()
                    )
                    .into());
                }
                idx
            }
            None => archive.len() - 1,
        };

        let model = archive.model_at(checkpoint_idx)?;
        loaded(&model);

        let model_save_path = run_dir
            .parent()
            .unwrap_or(Path::new("."))
            .join(format!("model-{}", meta.dataset));

        let from_epoch = archive
            .epoch_at(checkpoint_idx)
            .expect("checkpoint_idx was just validated against archive.len()");

        if let Some(state) = archive.scheduler_at(checkpoint_idx)? {
            hyperparams.scheduler_mut().restore(&state);
            completed(&format!(
                "Restored {} scheduler state from checkpoint at epoch {from_epoch}",
                hyperparams.scheduler().name()
            ));
        }
        if let Some(state) = archive.optimizer_at(checkpoint_idx)? {
            hyperparams.optimizer_mut().restore(&state)?;
            completed(&format!(
                "Restored {} optimizer state from checkpoint at epoch {from_epoch}",
                hyperparams.optimizer().name()
            ));
        }

        let recorder = if record.checkpoint_interval > 0 {
            let trimmed = run.trim_after(from_epoch)?;
            if trimmed > 0 {
                warning(&format!(
                    "Removed {trimmed} checkpoint(s) after epoch {from_epoch}"
                ));
            }
            recording_at(RUN_ICON, "TRAINING RUN", run_dir);
            Some(run.recorder())
        } else {
            None
        };

        execute_training(
            hyperparams,
            model,
            split,
            from_epoch,
            model_save_path,
            recorder,
            record,
            Some(previous),
        )
    }
}

// ─── Shared execution ─────────────────────────────────────────────────────────

/// Builds the callbacks, then runs the training loop to completion, mapping a
/// fatal divergence to a user-facing error.
#[allow(clippy::too_many_arguments)]
fn execute_training(
    hyperparams: HyperParams,
    model: NeuralNetwork,
    split: ModelSplit,
    epoch_start: usize,
    model_save_path: PathBuf,
    recorder: Option<CheckpointRecorder>,
    current: HyperParamsRecord,
    previous: Option<HyperParamsRecord>,
) -> Result<(), Box<dyn Error>> {
    let callbacks = Callbacks::empty()
        .with(ConsoleMonitor::new(current, previous))
        .with(ModelSaver::new(model_save_path))
        .with_opt(recorder);

    TrainingLoop {
        model,
        callbacks,
        split,
        hyperparams,
        epoch_start,
    }
    .run()?
    .into_result()
    .map_err(divergence_error)?;

    Ok(())
}

// ─── Report handling ──────────────────────────────────────────────────────────

/// Turns a [`FatalDivergence`] into a user-facing error with actionable hints.
fn divergence_error(divergence: FatalDivergence) -> Box<dyn Error> {
    format!(
        "Model diverged at epoch {} (NaN/Inf in weights). \
         Try: --early-stopping with --restore-best-model, --scheduler cosine, \
         a lower --lr, or stronger gradient clipping.",
        divergence.final_epoch
    )
    .into()
}

/// Wraps [`TrainingRun::create`], adding the `--overwrite` remediation
/// hint to an `AlreadyExists` error.
fn create_checkpoint_recorder(
    run_dir: &Path,
    dataset_name: &str,
    hyperparams: &HyperParamsRecord,
    overwrite: bool,
) -> IoResult<CheckpointRecorder> {
    let meta = TrainingMeta {
        dataset: dataset_name.to_string(),
        hyperparams: hyperparams.clone(),
    };
    TrainingRun::create(run_dir, &meta, overwrite)
        .map(|run| run.recorder())
        .map_err(|e| {
            if e.kind() == ErrorKind::AlreadyExists {
                IoError::new(e.kind(), format!("{e}; use --overwrite to replace it"))
            } else {
                e
            }
        })
}
