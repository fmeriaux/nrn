mod args;
mod callbacks;

use crate::actions::*;
use crate::display::{loaded, recording_at, warning};
use crate::path::PathExt;
use args::{ResumeOverrides, TrainArgs};
use callbacks::{ConsoleMonitor, ModelSaver};
use clap::*;
use nrn::data::ModelDataset;
use nrn::io::checkpoint::CheckpointRecorder;
use nrn::io::hyperparams::HyperParametersRecord;
use nrn::io::run::{TrainingMeta, TrainingRun};
use nrn::model::{LayerPlan, NeuralNetwork};
use nrn::optimizers::OptimizerState;
use nrn::schedulers::SchedulerState;
use nrn::training::{Callbacks, FatalDivergence, HyperParameters};
use std::error::Error;
use std::io::{Error as IoError, ErrorKind, Result as IoResult};
use std::path::{Path, PathBuf};

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

impl From<&StartArgs> for LayerPlan {
    fn from(args: &StartArgs) -> Self {
        if args.auto_layers {
            LayerPlan::Auto
        } else {
            LayerPlan::Explicit(args.layers.clone().unwrap_or_default())
        }
    }
}

impl StartArgs {
    pub fn run(&self) -> Result<(), Box<dyn Error>> {
        if let Some(ref layers) = self.layers
            && layers.contains(&0)
        {
            return Err("Each hidden layer must have at least one neuron.".into());
        }

        let dataset_name = Path::new(&self.dataset).file_stem_string();
        let dataset_path = Path::new(&self.dataset);
        let dataset = load_dataset(&self.dataset)?;

        let hyperparameters = HyperParameters::try_from(&self.hp)?;

        let model = match &self.model {
            Some(path) => load_model(path)?,
            None => initialize_model_with(&dataset, LayerPlan::from(self), hyperparameters.seed()),
        };

        let run_dir = dataset_path.with_file_name(format!("training-model-{dataset_name}"));
        let model_save_path = dataset_path.with_file_name(format!("model-{dataset_name}"));

        let recorder = if hyperparameters.checkpoint_interval() > 0 {
            let record = HyperParametersRecord::from(&hyperparameters);
            let recorder =
                create_checkpoint_recorder(&run_dir, &dataset_name, &record, self.overwrite)?;
            recording_at("TRAINING RUN", &run_dir);
            Some(recorder)
        } else {
            None
        };

        execute_training(
            hyperparameters,
            model,
            dataset.to_model_dataset(),
            model_save_path,
            recorder,
            None,
            None,
        )
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
    overrides: ResumeOverrides,
}

impl ResumeArgs {
    pub fn run(&self) -> Result<(), Box<dyn Error>> {
        let run_dir = Path::new(&self.run_dir);
        let run = TrainingRun::open(run_dir)?;
        let meta = run.meta();

        let dataset = load_dataset(&meta.dataset)?;

        let previous = HyperParameters::try_from(meta.hyperparams.clone())?;

        let mut record = meta.hyperparams.clone();
        self.overrides.apply(&mut record);
        let hyperparameters = HyperParameters::try_from(record)?;

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

        // Read the checkpointed optimizer/scheduler state now; it is restored on
        // the built `Trainer` just before training (see the `prepare` closure).
        let scheduler_state = archive.scheduler_at(checkpoint_idx)?;
        let optimizer_state = archive.optimizer_at(checkpoint_idx)?;

        let recorder = if hyperparameters.checkpoint_interval() > 0 {
            let trimmed = run.trim_after(from_epoch)?;
            if trimmed > 0 {
                warning(&format!(
                    "Removed {trimmed} checkpoint(s) after epoch {from_epoch}"
                ));
            }
            recording_at("TRAINING RUN", run_dir);
            Some(run.recorder())
        } else {
            None
        };

        execute_training(
            hyperparameters,
            model,
            dataset.to_model_dataset(),
            model_save_path,
            recorder,
            Some(previous),
            Some(ResumeState {
                epoch_start: from_epoch,
                optimizer: optimizer_state,
                scheduler: scheduler_state,
            }),
        )
    }
}

// ─── Shared execution ─────────────────────────────────────────────────────────

/// The checkpointed state a resumed run restores onto its [`Trainer`] before
/// training: the epoch to resume from plus any persisted optimizer/scheduler state.
struct ResumeState {
    epoch_start: usize,
    optimizer: Option<OptimizerState>,
    scheduler: Option<SchedulerState>,
}

/// Builds the callbacks and the `Trainer`, restores checkpointed state when
/// resuming, then trains to completion, mapping a fatal divergence to a
/// user-facing error.
fn execute_training(
    hyperparameters: HyperParameters,
    model: NeuralNetwork,
    dataset: ModelDataset,
    model_save_path: PathBuf,
    recorder: Option<CheckpointRecorder>,
    previous: Option<HyperParameters>,
    resume: Option<ResumeState>,
) -> Result<(), Box<dyn Error>> {
    let callbacks = Callbacks::empty()
        .with(ConsoleMonitor::new(hyperparameters.clone(), previous))
        .with(ModelSaver::new(model_save_path))
        .with_opt(recorder);

    let mut trainer = hyperparameters.build(model, dataset, callbacks);

    // Restoring fires `on_restore`, which the console monitor narrates.
    if let Some(resume) = resume {
        trainer.restore(resume.epoch_start, resume.optimizer, resume.scheduler)?;
    }

    trainer.train()?.into_result().map_err(divergence_error)?;

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
    hyperparams: &HyperParametersRecord,
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
