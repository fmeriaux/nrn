use crate::actions::*;
use crate::display::{HISTORY_ICON, Summary, completed, loaded, saved_at, trace, warning};
use crate::progression::Progression;
use clap::*;
use console::style;
use nrn::accuracies::{Accuracy, accuracy_for};
use nrn::data::ModelSplit;
use nrn::evaluation::EvaluationSet;
use nrn::io::training_history::{SnapshotMeta, SnapshotRecorder};
use nrn::loss_functions::{CROSS_ENTROPY_LOSS, LossFunction};
use nrn::model::NeuralNetwork;
use nrn::optimizers::{Adam, Optimizer, StochasticGradientDescent};
use nrn::recorders::{Checkpoints, NoOpRecorder, Recorder};
use nrn::schedulers;
use nrn::schedulers::{ConstantScheduler, Scheduler, StepDecay};
use nrn::training::{EarlyStopping, GradientClipping, LearningRate};
use nrn::training_history::TrainingHistory;
use schedulers::CosineAnnealing;
use std::error::Error;
use std::fmt::Display;
use std::path::{Path, PathBuf};
use std::sync::Arc;

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
    /// Resume training from an existing history directory
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
}

// ─── Start ────────────────────────────────────────────────────────────────────

#[derive(Args, Debug)]
pub struct StartArgs {
    /// Dataset to train on
    dataset: String,

    /// Load a pre-trained model to continue training (snapshot count resets to 0)
    #[arg(short, long, conflicts_with_all = &["layers", "auto_layers"])]
    model: Option<String>,

    /// Hidden layer sizes (comma-separated)
    #[arg(long, value_delimiter = ',', conflicts_with_all = &["auto_layers", "model"])]
    layers: Option<Vec<usize>>,

    /// Infer hidden layers from dataset characteristics
    #[arg(long, conflicts_with_all = &["layers", "model"], default_value_t = false)]
    auto_layers: bool,

    /// Overwrite an existing training history directory
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

        let history_dir = dataset_path.with_file_name(format!("training-model-{dataset_name}"));
        let model_save_path = dataset_path.with_file_name(format!("model-{dataset_name}"));

        let interval = self.hp.checkpoint_interval;
        let mut checkpoints = if interval > 0 {
            trace(&format!(
                "Recording a checkpoint every {} epochs",
                style(interval).yellow()
            ));
            let mut recorder =
                SnapshotRecorder::create(&history_dir, interval, &dataset_name, self.overwrite)?;
            let loss_fn: Arc<dyn LossFunction> = CROSS_ENTROPY_LOSS.clone();
            let evals = EvaluationSet::using_model(
                &model,
                &loss_fn,
                &accuracy_for(dataset.n_classes()),
                &split,
                None,
            );
            recorder.record(&model, &evals)?;
            Checkpoints::new(Box::new(recorder), interval, self.hp.epochs)
        } else {
            Checkpoints::new(Box::new(NoOpRecorder), 0, self.hp.epochs)
        };

        TrainingLoop {
            model,
            checkpoints,
            history_dir: (interval > 0).then(|| history_dir),
            split,
            accuracy: accuracy_for(dataset.n_classes()),
            optimizer: self.hp.make_optimizer(),
            scheduler: self.hp.make_scheduler(),
            clipping: self.hp.infer_clipping(),
            early_stopping: self.hp.early_stopping(),
            epochs: self.hp.epochs,
            restore_best_model: self.hp.restore_best_model,
            batch_size: self.hp.batch_size,
            model_save_path,
        }
        .run()
    }
}

// ─── Resume ───────────────────────────────────────────────────────────────────

#[derive(Args, Debug)]
pub struct ResumeArgs {
    /// Training history directory to resume from
    history_dir: String,

    /// Snapshot index to resume from (default: last snapshot)
    #[arg(long)]
    from: Option<usize>,

    #[command(flatten)]
    hp: HyperParams,
}

impl ResumeArgs {
    pub fn run(&self) -> Result<(), Box<dyn Error>> {
        self.hp.validate()?;

        let history_dir = Path::new(&self.history_dir);
        let meta = SnapshotMeta::load(history_dir)?;

        let dataset = load_dataset(&meta.dataset)?;
        dataset.validate()?;

        let split = dataset
            .to_model_dataset()
            .split(self.hp.val_ratio, self.hp.test_ratio);
        completed(split.summary().as_str());

        let history = TrainingHistory::load(history_dir)?;
        if history.is_empty() {
            return Err(format!(
                "No snapshots found in '{}'; cannot resume.",
                history_dir.display()
            )
            .into());
        }

        let snapshot_idx = match self.from {
            Some(idx) => {
                if idx >= history.len() {
                    return Err(format!(
                        "Snapshot index {idx} out of range (history has {} snapshots).",
                        history.len()
                    )
                    .into());
                }
                idx
            }
            None => history.len() - 1,
        };

        let model = history.model_at(snapshot_idx)?;
        loaded(&model);

        if matches!(self.hp.optimizer, OptimizerType::Adam) {
            warning(
                "Resuming with Adam: optimizer state is not restored, \
                 its moments restart from zero for the first epochs",
            );
        }

        let model_save_path = history_dir
            .parent()
            .unwrap_or(Path::new("."))
            .join(format!("model-{}", meta.dataset));

        let interval = meta.interval;
        let checkpoints = if interval > 0 {
            trace(&format!(
                "Recording a checkpoint every {} epochs",
                style(interval).yellow()
            ));
            Checkpoints::new(
                Box::new(SnapshotRecorder::resume(
                    history_dir,
                    interval,
                    snapshot_idx,
                )?),
                interval,
                self.hp.epochs,
            )
        } else {
            Checkpoints::new(Box::new(NoOpRecorder), 0, self.hp.epochs)
        };

        TrainingLoop {
            model,
            checkpoints,
            history_dir: (interval > 0).then(|| history_dir.to_path_buf()),
            split,
            accuracy: accuracy_for(dataset.n_classes()),
            optimizer: self.hp.make_optimizer(),
            scheduler: self.hp.make_scheduler(),
            clipping: self.hp.infer_clipping(),
            early_stopping: self.hp.early_stopping(),
            epochs: self.hp.epochs,
            restore_best_model: self.hp.restore_best_model,
            batch_size: self.hp.batch_size,
            model_save_path,
        }
        .run()
    }
}

// ─── Training loop ────────────────────────────────────────────────────────────

struct TrainingLoop {
    model: NeuralNetwork,
    checkpoints: Checkpoints,
    history_dir: Option<PathBuf>,
    split: ModelSplit,
    accuracy: Arc<dyn Accuracy>,
    optimizer: Box<dyn Optimizer>,
    scheduler: Box<dyn Scheduler>,
    clipping: GradientClipping,
    early_stopping: Option<EarlyStopping>,
    epochs: usize,
    restore_best_model: bool,
    batch_size: Option<usize>,
    model_save_path: PathBuf,
}

impl TrainingLoop {
    fn run(mut self) -> Result<(), Box<dyn Error>> {
        let loss_function: Arc<dyn LossFunction> = CROSS_ENTROPY_LOSS.clone();
        trace(&format!(
            "Using {} loss function",
            style("Cross-Entropy").bold().blue()
        ));

        let mut early_stopping = self.early_stopping;
        // Seed best model before epoch 1 so divergence at the first epoch can recover.
        if let Some(ref mut es) = early_stopping
            && self.split.validation.is_some()
        {
            es.seed_best_model(&self.model);
        }

        let mut final_evaluations: Option<EvaluationSet> = None;
        let progression = Progression::new(self.epochs, "Training");

        for epoch in progression.iter() {
            self.model.train(
                &self.split.train,
                &loss_function,
                self.optimizer.as_mut(),
                self.scheduler.as_mut(),
                &self.clipping,
                self.batch_size,
            );

            if !self.model.is_finite() {
                let recovered = early_stopping.as_mut().and_then(|es| es.best_model.take());

                if let Some(best) = recovered {
                    progression.done();
                    warning(&format!(
                        "Model diverged at epoch {} (NaN/Inf); recovered best model from early stopping.",
                        epoch + 1
                    ));
                    self.model = best;
                    let evals = EvaluationSet::using_model(
                        &self.model,
                        &loss_function,
                        &self.accuracy,
                        &self.split,
                        None,
                    );
                    self.checkpoints.record(&self.model, &evals)?;
                    final_evaluations = Some(evals);
                    break;
                }

                if let Some(ref dir) = self.history_dir {
                    saved_at(HISTORY_ICON, "TRAINING HISTORY", dir);
                }
                return Err(format!(
                    "Model diverged at epoch {} (NaN/Inf in weights). \
                     Try: --early-stopping with --restore-best-model, --scheduler cosine, \
                     a lower --lr, or stronger gradient clipping.",
                    epoch + 1
                )
                .into());
            }

            let wrote_this_epoch = if self.checkpoints.is_due(epoch) {
                let train_preds = self.model.predict(self.split.train.inputs.view());
                let evals = EvaluationSet::using_model(
                    &self.model,
                    &loss_function,
                    &self.accuracy,
                    &self.split,
                    Some(train_preds.view()),
                );
                self.checkpoints.record(&self.model, &evals)?;
                true
            } else {
                false
            };

            if let Some(ref mut es) = early_stopping
                && let Some(validation) = &self.split.validation
            {
                let preds = self.model.predict(validation.inputs.view());
                let loss = loss_function.compute(preds.view(), validation.targets.view());

                if es.check(loss, &self.model) {
                    progression.done();
                    completed(&format!(
                        "Early stopping triggered at epoch {}",
                        style(epoch + 1).yellow()
                    ));
                    if self.restore_best_model {
                        self.model = es
                            .best_model
                            .as_ref()
                            .expect("Best model should be available")
                            .clone();
                        trace("Restored the best model observed during training");
                    }
                    let stop_evals = EvaluationSet::using_model(
                        &self.model,
                        &loss_function,
                        &self.accuracy,
                        &self.split,
                        None,
                    );
                    if !wrote_this_epoch {
                        self.checkpoints.record(&self.model, &stop_evals)?;
                    }
                    final_evaluations = Some(stop_evals);
                    break;
                }
            }
        }

        let evaluations = final_evaluations.unwrap_or_else(|| {
            EvaluationSet::using_model(
                &self.model,
                &loss_function,
                &self.accuracy,
                &self.split,
                None,
            )
        });

        completed(&format!(
            "{} | {}",
            style("Training completed").bright().green(),
            evaluations.summary()
        ));

        save_model(&self.model_save_path, &self.model)?;

        if let Some(ref dir) = self.history_dir {
            saved_at(HISTORY_ICON, "TRAINING HISTORY", dir);
        }

        Ok(())
    }
}
