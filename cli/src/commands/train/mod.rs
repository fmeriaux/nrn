mod args;
mod callbacks;

use crate::display::{initialized, loaded, recording_at, warning};
use crate::path::PathExt;
use args::{ResumeOverrides, TrainArgs};
use callbacks::{ConsoleMonitor, ModelSaver};
use clap::*;
use nrn::activations::RELU;
use nrn::data::Dataset;
use nrn::io::hyperparams::HyperParametersRecord;
use nrn::io::run::{TrainingMeta, TrainingRun};
use nrn::model::{LayerPlan, NeuralNetwork, NeuronLayerSpec};
use nrn::training::{Callbacks, FatalDivergence, HyperParameters};
use std::error::Error;
use std::fmt;
use std::io::{Error as IoError, ErrorKind};
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

impl StartArgs {
    pub fn run(&self) -> Result<(), Box<dyn Error>> {
        let dataset_path = Path::new(&self.dataset);
        let dataset_name = dataset_path.file_stem_string();
        let run_dir = dataset_path.sibling("training-model");
        let model_name = format!("model-{dataset_name}");

        let dataset = Dataset::load(&self.dataset)?;
        loaded(&dataset);

        let hyperparameters = HyperParameters::try_from(&self.hp)?;

        let model = match &self.model {
            Some(path) => {
                let model = NeuralNetwork::load(path)?;
                loaded(&model);
                model
            }
            None => {
                let plan = if self.auto_layers {
                    LayerPlan::Auto {
                        n_features: dataset.n_features(),
                        n_samples: dataset.n_samples(),
                    }
                } else {
                    LayerPlan::Explicit(self.layers.clone().unwrap_or_default())
                };
                let layer_specs = NeuronLayerSpec::plan(plan, dataset.n_classes(), &*RELU)?;
                let model = NeuralNetwork::initialization(
                    dataset.n_features(),
                    &layer_specs,
                    hyperparameters.seed(),
                );
                initialized(&model);
                model
            }
        };

        let recorder = if hyperparameters.checkpoint_interval() > 0 {
            let meta = TrainingMeta {
                dataset: dataset_name,
                model: model_name.clone(),
                hyperparams: HyperParametersRecord::from(&hyperparameters),
            };
            let recorder = TrainingRun::create(&run_dir, &meta, self.overwrite)
                .map_err(overwrite_hint)?
                .recorder();
            recording_at("TRAINING RUN", &run_dir);
            Some(recorder)
        } else {
            None
        };

        let callbacks = Callbacks::empty()
            .with(ConsoleMonitor::new(hyperparameters.clone(), None))
            .with(ModelSaver::new(&run_dir, &model_name))
            .with_opt(recorder);

        hyperparameters
            .build(model, dataset.to_model_dataset(), callbacks)
            .train()?
            .into_result()
            .map_err(DivergedRun::from)?;

        Ok(())
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

        let dataset = Dataset::load(&meta.dataset)?;
        loaded(&dataset);

        let previous = HyperParameters::try_from(meta.hyperparams.clone())?;

        let mut record = meta.hyperparams.clone();
        self.overrides.apply(&mut record);
        let hyperparameters = HyperParameters::try_from(record)?;

        let archive = run.archive()?;
        if archive.is_empty() {
            return Err(ResumeError::NoCheckpoints(run_dir.to_path_buf()).into());
        }

        let checkpoint_idx = match self.from {
            Some(idx) => {
                if idx >= archive.len() {
                    return Err(ResumeError::CheckpointOutOfRange {
                        index: idx,
                        available: archive.len(),
                    }
                    .into());
                }
                idx
            }
            None => archive.len() - 1,
        };

        let model = archive.model_at(checkpoint_idx)?;
        loaded(&model);

        let from_epoch = archive
            .epoch_at(checkpoint_idx)
            .expect("checkpoint_idx was just validated against archive.len()");

        let scheduler_state = archive.scheduler_at(checkpoint_idx)?;
        let optimizer_state = archive.optimizer_at(checkpoint_idx)?;

        let recorder = if hyperparameters.checkpoint_interval() > 0 {
            let trimmed = run.trim_after(from_epoch)?;
            if trimmed > 0 {
                warning!("Removed {trimmed} checkpoint(s) after epoch {from_epoch}");
            }
            recording_at("TRAINING RUN", run_dir);
            Some(run.recorder())
        } else {
            None
        };

        let callbacks = Callbacks::empty()
            .with(ConsoleMonitor::new(hyperparameters.clone(), Some(previous)))
            .with(ModelSaver::new(run_dir, &meta.model))
            .with_opt(recorder);

        let mut trainer = hyperparameters.build(model, dataset.to_model_dataset(), callbacks);
        trainer.restore(from_epoch, optimizer_state, scheduler_state)?;
        trainer.train()?.into_result().map_err(DivergedRun::from)?;

        Ok(())
    }
}

/// Errors raised while validating a resume request against a run's checkpoints.
#[derive(Debug)]
enum ResumeError {
    /// The run directory holds no checkpoints to resume from.
    NoCheckpoints(PathBuf),
    /// The requested checkpoint index is past the last checkpoint.
    CheckpointOutOfRange { index: usize, available: usize },
}

impl fmt::Display for ResumeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ResumeError::NoCheckpoints(dir) => {
                write!(
                    f,
                    "no checkpoints found in '{}'; cannot resume",
                    dir.display()
                )
            }
            ResumeError::CheckpointOutOfRange { index, available } => write!(
                f,
                "checkpoint index {index} out of range (run has {available} checkpoints)"
            ),
        }
    }
}

impl Error for ResumeError {}

/// Reported when training ends in an unrecovered divergence, with actionable hints.
#[derive(Debug)]
struct DivergedRun {
    final_epoch: usize,
}

impl From<FatalDivergence> for DivergedRun {
    fn from(divergence: FatalDivergence) -> Self {
        Self {
            final_epoch: divergence.final_epoch,
        }
    }
}

impl fmt::Display for DivergedRun {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "model diverged at epoch {} (NaN/Inf in weights). \
             Try: --early-stopping with --restore-best-model, --scheduler cosine, \
             a lower --lr, or stronger gradient clipping.",
            self.final_epoch
        )
    }
}

impl Error for DivergedRun {}

/// Adds the `--overwrite` remediation hint to an `AlreadyExists` error.
fn overwrite_hint(error: IoError) -> IoError {
    if error.kind() == ErrorKind::AlreadyExists {
        IoError::new(
            error.kind(),
            format!("{error}; use --overwrite to replace it"),
        )
    } else {
        error
    }
}
