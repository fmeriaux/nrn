mod model_saver;
mod monitor;

use crate::actions::*;
use crate::console::{RUN_ICON, Summary, completed, loaded, recording_at, warning};
use clap::*;
use model_saver::ModelSaver;
use monitor::ConsoleMonitor;
use nrn::data::ModelSplit;
use nrn::io::checkpoint::{CheckpointArchive, CheckpointRecorder, TrainingMeta, TrainingRun};
use nrn::loss_functions::CROSS_ENTROPY_LOSS;
use nrn::model::NeuralNetwork;
use nrn::optimizers::{Adam, Optimizer, StochasticGradientDescent};
use nrn::schedulers;
use nrn::schedulers::{ConstantScheduler, Scheduler, StepDecay};
use nrn::training::{
    Callbacks, EarlyStopping, FatalDivergence, GradientClipping, LearningRate, TrainingConfig,
    TrainingLoop,
};
use schedulers::CosineAnnealing;
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
pub struct HyperParams {
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

impl HyperParams {
    fn validate(&self) -> Result<(), Box<dyn Error>> {
        if self.epochs < 1 {
            return Err("The number of epochs must be greater than zero.".into());
        }
        if self.lr < 0.0 {
            return Err("The learning rate must be a non-negative value.".into());
        }
        if let Some(lr_min) = self.lr_min
            && (lr_min < 0.0 || lr_min >= self.lr)
        {
            return Err("The minimum learning rate must be non-negative and less than the initial learning rate.".into());
        }
        if self.decay_factor <= 0.0 {
            return Err("The decay factor must be a positive value.".into());
        }
        if let Some(steps) = self.steps
            && steps < 1
        {
            return Err("The step size must be greater than zero.".into());
        }
        if self.clip_norm <= 0.0 {
            return Err("The gradient clipping norm must be a positive value.".into());
        }
        if let Some(clip_value) = self.clip_value
            && clip_value <= 0.0
        {
            return Err("The gradient clipping value must be a positive value.".into());
        }
        if let Some(cycle_multiplier) = self.cycle_multiplier
            && cycle_multiplier < 1
        {
            return Err("The cycle multiplier must be at least 1.".into());
        }
        if self.val_ratio < 0.0 || self.val_ratio >= 1.0 {
            return Err("Validation ratio must be in the range [0.0, 1.0)".into());
        }
        if self.test_ratio <= 0.0 || self.test_ratio >= 1.0 {
            return Err("Test ratio must be in the range [0.0, 1.0)".into());
        }
        if self.val_ratio + self.test_ratio >= 1.0 {
            return Err("The sum of validation and test ratios must be less than 1.0".into());
        }
        Ok(())
    }

    fn make_optimizer(&self) -> Box<dyn Optimizer> {
        match self.optimizer {
            OptimizerType::Sgd => Box::new(StochasticGradientDescent::new(self.lr())),
            OptimizerType::Adam => Box::new(Adam::with_defaults(self.lr())),
        }
    }

    fn make_scheduler(&self) -> Box<dyn Scheduler> {
        match self.scheduler {
            SchedulerType::Constant => Box::new(ConstantScheduler::new(self.lr())),
            SchedulerType::Cosine => {
                let cosine = CosineAnnealing::new(self.lr_min(), self.lr(), self.step_size());
                if self.warm_restarts {
                    Box::new(cosine.with_restarts(true, self.cycle_multiplier.unwrap_or(1)))
                } else {
                    Box::new(cosine)
                }
            }
            SchedulerType::Step => Box::new(StepDecay::new(
                self.lr(),
                self.step_size(),
                self.decay_factor,
            )),
        }
    }

    fn infer_clipping(&self) -> GradientClipping {
        if self.no_clip {
            GradientClipping::None
        } else if let Some(value) = self.clip_value {
            GradientClipping::Value {
                min: -value,
                max: value,
            }
        } else {
            GradientClipping::Norm {
                max_norm: self.clip_norm,
            }
        }
    }

    fn early_stopping(&self) -> Option<EarlyStopping> {
        if self.early_stopping > 0 {
            Some(EarlyStopping::new(
                self.early_stopping,
                self.restore_best_model,
            ))
        } else {
            None
        }
    }

    fn lr(&self) -> LearningRate {
        LearningRate::new(self.lr)
    }

    fn lr_min(&self) -> LearningRate {
        LearningRate::new(self.lr_min.unwrap_or(0.0))
    }

    fn step_size(&self) -> usize {
        self.steps.unwrap_or(self.epochs)
    }

    fn make_config(&self) -> TrainingConfig {
        TrainingConfig {
            epochs: self.epochs,
            eval_interval: self.checkpoint_interval,
            batch_size: self.batch_size,
            loss: CROSS_ENTROPY_LOSS.clone(),
            optimizer: self.make_optimizer(),
            scheduler: self.make_scheduler(),
            clipping: self.infer_clipping(),
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
    hp: HyperParams,
}

impl StartArgs {
    fn validate(&self) -> Result<(), Box<dyn Error>> {
        self.hp.validate()?;
        if let Some(ref layers) = self.layers
            && layers.contains(&0)
        {
            return Err("Each hidden layer must have at least one neuron.".into());
        }
        Ok(())
    }

    pub fn run(&self) -> Result<(), Box<dyn Error>> {
        self.validate()?;

        let dataset_name = get_file_stem(Path::new(&self.dataset));
        let dataset_path = Path::new(&self.dataset);
        let dataset = load_dataset(&self.dataset)?;
        dataset.validate()?;

        let split = dataset
            .to_model_dataset()
            .split(self.hp.val_ratio, self.hp.test_ratio);
        completed(split.summary().as_str());

        let model = match &self.model {
            Some(path) => {
                if matches!(self.hp.optimizer, OptimizerType::Adam) {
                    warning(
                        "Resuming with Adam: optimizer state is not restored, \
                         its moments restart from zero for the first epochs",
                    );
                }
                load_model(path)?
            }
            None => initialize_model_with(&dataset, self.layers.clone(), self.auto_layers),
        };

        let run_dir = dataset_path.with_file_name(format!("training-model-{dataset_name}"));
        let model_save_path = dataset_path.with_file_name(format!("model-{dataset_name}"));

        let recorder = if self.hp.checkpoint_interval > 0 {
            let recorder = create_checkpoint_recorder(&run_dir, &dataset_name, self.overwrite)?;
            recording_at(RUN_ICON, "TRAINING RUN", &run_dir);
            Some(recorder)
        } else {
            None
        };

        execute_training(&self.hp, model, split, 0, model_save_path, recorder)
    }
}

// ─── Resume ───────────────────────────────────────────────────────────────────

#[derive(Args, Debug)]
pub struct ResumeArgs {
    /// Training run directory to resume from
    run_dir: String,

    /// Checkpoint index to resume from (default: last checkpoint)
    #[arg(long)]
    from: Option<usize>,

    #[command(flatten)]
    hp: HyperParams,
}

impl ResumeArgs {
    pub fn run(&self) -> Result<(), Box<dyn Error>> {
        self.hp.validate()?;

        let run_dir = Path::new(&self.run_dir);
        let meta = TrainingMeta::load(run_dir)?;

        let dataset = load_dataset(&meta.dataset)?;
        dataset.validate()?;

        let split = dataset
            .to_model_dataset()
            .split(self.hp.val_ratio, self.hp.test_ratio);
        completed(split.summary().as_str());

        let archive = CheckpointArchive::load(run_dir)?;
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

        if matches!(self.hp.optimizer, OptimizerType::Adam) {
            warning(
                "Resuming with Adam: optimizer state is not restored, \
                 its moments restart from zero for the first epochs",
            );
        }

        let model_save_path = run_dir
            .parent()
            .unwrap_or(Path::new("."))
            .join(format!("model-{}", meta.dataset));

        let from_epoch = archive
            .epoch_at(checkpoint_idx)
            .expect("checkpoint_idx was just validated against archive.len()");

        let recorder = if self.hp.checkpoint_interval > 0 {
            let (recorder, trimmed) = TrainingRun::resume(run_dir, from_epoch)?;
            if trimmed > 0 {
                warning(&format!(
                    "Removed {trimmed} checkpoint(s) after epoch {from_epoch}"
                ));
            }
            recording_at(RUN_ICON, "TRAINING RUN", run_dir);
            Some(recorder)
        } else {
            None
        };

        execute_training(
            &self.hp,
            model,
            split,
            from_epoch,
            model_save_path,
            recorder,
        )
    }
}

// ─── Shared execution ─────────────────────────────────────────────────────────

/// Builds the callbacks and config from `hp`, then runs the training loop to
/// completion, mapping a fatal divergence to a user-facing error.
fn execute_training(
    hp: &HyperParams,
    model: NeuralNetwork,
    split: ModelSplit,
    epoch_start: usize,
    model_save_path: PathBuf,
    recorder: Option<CheckpointRecorder>,
) -> Result<(), Box<dyn Error>> {
    let callbacks = Callbacks::empty()
        .with(ConsoleMonitor::new())
        .with(ModelSaver::new(model_save_path))
        .with_opt(recorder);

    TrainingLoop {
        model,
        callbacks,
        split,
        config: hp.make_config(),
        early_stopping: hp.early_stopping(),
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
    overwrite: bool,
) -> IoResult<CheckpointRecorder> {
    let meta = TrainingMeta {
        dataset: dataset_name.to_string(),
    };
    TrainingRun::create(run_dir, &meta, overwrite).map_err(|e| {
        if e.kind() == ErrorKind::AlreadyExists {
            IoError::new(e.kind(), format!("{e}; use --overwrite to replace it"))
        } else {
            e
        }
    })
}
