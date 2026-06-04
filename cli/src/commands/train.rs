use crate::actions::*;
use crate::display::{HISTORY_ICON, Summary, completed, saved_at, trace, warning};
use crate::progression::Progression;
use clap::*;
use console::style;
use nrn::accuracies::{Accuracy, accuracy_for};
use nrn::evaluation::EvaluationSet;
use nrn::io::training_history::TrainingHistoryWriter;
use nrn::loss_functions::{CROSS_ENTROPY_LOSS, LossFunction};
use nrn::optimizers::{Adam, Optimizer, StochasticGradientDescent};
use nrn::schedulers;
use nrn::schedulers::{ConstantScheduler, Scheduler, StepDecay};
use nrn::training::{EarlyStopping, GradientClipping, LearningRate};
use schedulers::CosineAnnealing;
use std::error::Error;
use std::fmt::Display;
use std::path::Path;
use std::sync::Arc;

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

#[derive(Args, Debug)]
pub struct TrainArgs {
    /// Name of the dataset to train on
    dataset: String,

    /// Provide a pre-trained model to continue training, if not provided, a new model will be initialized
    #[arg(short, long)]
    model: Option<String>,

    /// The number of epochs to train the model
    #[arg(short, long)]
    epochs: usize,

    #[arg(short = 'k', long, default_value_t = 10)]
    /// Specify the checkpoint interval for saving the model state,
    /// if set to 0, no checkpoints will be saved
    checkpoint_interval: usize,

    /// Specify the hidden layers of the model when a new model is initialized
    #[arg(long, value_delimiter = ',', conflicts_with_all = &["auto_layers", "model"])]
    layers: Option<Vec<usize>>,

    /// Automatically infer the hidden layers based on the dataset characteristics
    #[arg(long, conflicts_with_all = &["layers", "model"], default_value_t = false)]
    auto_layers: bool,

    /// Specify the optimizer to use for training
    #[arg(long, value_enum, default_value_t = OptimizerType::Adam)]
    optimizer: OptimizerType,

    /// Specify the scheduler to use for adjusting the learning rate during training
    #[arg(long, value_enum, default_value_t = SchedulerType::Constant)]
    scheduler: SchedulerType,

    /// Enable warm restarts for schedulers that support it (e.g., cosine)
    #[arg(long, requires = "scheduler", default_value_t = false)]
    warm_restarts: bool,

    /// Specify the cycle multiplier for schedulers that support it (e.g., cosine with warm-restarts)
    #[arg(long, requires = "warm_restarts")]
    cycle_multiplier: Option<usize>,

    /// Specify the decay factor for schedulers that require it (e.g., step)
    #[arg(long, requires = "scheduler", default_value_t = 0.1)]
    decay_factor: f32,

    /// Specify the step size for learning rate schedulers that require it:
    /// - cosine: number of epochs to complete a full cosine cycle
    /// - step: number of epochs between each learning rate decay
    ///   By default, this is set to the total number of epochs
    #[arg(long, requires = "scheduler")]
    steps: Option<usize>,

    /// Specify the learning rate for the training process
    #[arg(long, default_value_t = 0.001)]
    lr: f32,

    /// Specify the minimum learning rate for schedulers that support it (e.g., cosine)
    #[arg(long, requires = "scheduler")]
    lr_min: Option<f32>,

    /// Specify the gradient clipping norm to prevent exploding gradients
    #[arg(long, default_value_t = 1.0, conflicts_with_all = &["clip_value", "no_clip"])]
    clip_norm: f32,

    /// Specify the gradient clipping value to prevent exploding gradients.
    /// This performs element-wise clipping: each gradient component is clipped individually to the symmetric range [-value, value].
    #[arg(long, conflicts_with_all = &["clip_norm", "no_clip"])]
    clip_value: Option<f32>,

    /// Disable gradient clipping
    #[arg(long, conflicts_with_all = &["clip_norm", "clip_value"])]
    no_clip: bool,

    /// Specify the validation ratio for the dataset split
    #[arg(long, default_value_t = 0.1)]
    val_ratio: f32,

    /// Specify the test ratio for the dataset split
    #[arg(long, default_value_t = 0.1)]
    test_ratio: f32,

    /// Define patience for early stopping based on validation loss, put to 0 to disable early stopping
    #[arg(long, default_value_t = 0)]
    early_stopping: usize,

    /// Restore the best model observed during training when using early stopping
    #[arg(long, requires = "early_stopping", default_value_t = true)]
    restore_best_model: bool,

    /// Batch size for mini-batch SGD. If not set, full-batch gradient descent is used.
    #[arg(long)]
    batch_size: Option<usize>,
}

impl TrainArgs {
    fn validate(&self) -> Result<(), Box<dyn Error>> {
        if self.epochs < 1 {
            return Err("The number of epochs must be greater than zero.".into());
        }

        if let Some(ref layers) = self.layers
            && layers.contains(&0)
        {
            return Err("Each hidden layer must have at least one neuron.".into());
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

    pub fn run(&self) -> Result<(), Box<dyn Error>> {
        self.validate()?;

        let loss_function: Arc<dyn LossFunction> = CROSS_ENTROPY_LOSS.clone();
        trace(&format!(
            "Using {} loss function",
            style("Cross-Entropy").bold().blue()
        ));

        let mut optimizer = self.make_optimizer();
        trace(&format!(
            "Using {} optimizer with learning rate {}",
            style(self.optimizer).bold().blue(),
            style(self.lr).yellow()
        ));

        let mut scheduler = self.make_scheduler();
        if !matches!(self.scheduler, SchedulerType::Constant) {
            trace(&format!(
                "Learning rate scheduled by {}",
                style(self.scheduler).bold().blue()
            ));
        }

        let clipping = self.infer_clipping();
        trace(&format!("Using {}", clipping.summary()));

        let dataset = load_dataset(&self.dataset)?;
        dataset.validate()?;
        let split = dataset
            .to_model_dataset()
            .split(self.val_ratio, self.test_ratio);
        completed(split.summary().as_str());

        let accuracy: Arc<dyn Accuracy> = accuracy_for(dataset.n_classes());

        // 🧠 NEURAL NETWORK INITIALIZATION
        let mut model = match &self.model {
            Some(file) => load_model(file)?,
            None => initialize_model_with(&dataset, self.layers.clone(), self.auto_layers),
        };

        // Optimizer state is not persisted: a stateful optimizer resets its moments and
        // step counter when resuming from a saved model, so its adaptive steps warm up
        // again over the first epochs. See `make_optimizer`.
        if self.model.is_some() && matches!(self.optimizer, OptimizerType::Adam) {
            warning(
                "Resuming with Adam: optimizer state is not restored, \
                 its moments restart from zero for the first epochs",
            );
        }

        // Compute names up front so the writer can be created before the loop.
        let path = Path::new(&self.dataset);
        let dataset_name = get_file_stem(path);
        let model_name = format!("model-{}", dataset_name);
        let history_dir = path.with_file_name(format!("training-{}", model_name));

        // 👨‍🎓 TRAINING LOOP
        let mut writer: Option<TrainingHistoryWriter> = if self.checkpoint_interval > 0 {
            trace(&format!(
                "Recording a checkpoint every {} epochs",
                style(self.checkpoint_interval).yellow()
            ));
            let mut w = create_history_writer(&history_dir, self.checkpoint_interval)?;
            let evaluations =
                EvaluationSet::using_model(&model, &loss_function, &accuracy, &split, None);
            w.record(&model, &evaluations)?;
            Some(w)
        } else {
            None
        };

        let mut early_stopping = self.early_stopping();
        // Seed the best model with the pre-training state so that divergence at epoch 1
        // (before any es.check() call) can still recover instead of erroring out.
        // Only seed when a validation split exists: without one es.check() is never called,
        // so best_model would stay at the untrained initial state and silently "recover" to it.
        if let Some(ref mut es) = early_stopping
            && split.validation.is_some()
        {
            es.seed_best_model(&model);
        }
        // Holds the final evaluations when early stopping fires, so the post-loop
        // summary can reuse them instead of running a second forward pass.
        let mut final_evaluations: Option<EvaluationSet> = None;

        let progression = Progression::new(self.epochs, "Training");
        for epoch in progression.iter() {
            model.train(
                &split.train,
                &loss_function,
                optimizer.as_mut(),
                scheduler.as_mut(),
                &clipping,
                self.batch_size,
            );

            if !model.is_finite() {
                let recovered = early_stopping
                    .as_mut()
                    .and_then(|es| es.best_model.take());

                if let Some(best) = recovered {
                    progression.done();
                    warning(&format!(
                        "Model diverged at epoch {} (NaN/Inf); recovered best model from early stopping.",
                        epoch + 1
                    ));
                    model = best;
                    let evals =
                        EvaluationSet::using_model(&model, &loss_function, &accuracy, &split, None);
                    // No checkpoint was written for this epoch yet (divergence precedes the
                    // checkpoint block), so always record the recovered model.
                    if let Some(ref mut w) = writer {
                        w.record(&model, &evals)?;
                    }
                    final_evaluations = Some(evals);
                    break;
                }

                if let Some(ref writer) = writer {
                    saved_at(HISTORY_ICON, "TRAINING HISTORY", writer.dir());
                }
                return Err(format!(
                    "Model diverged at epoch {} (NaN/Inf in weights). \
                     Try: --early-stopping with --restore-best-model, --scheduler cosine, \
                     a lower --lr, or stronger gradient clipping.",
                    epoch + 1
                )
                .into());
            }

            let mut wrote_this_epoch = false;
            if let Some(ref mut writer) = writer {
                let epoch_number = epoch + 1;
                if epoch_number % self.checkpoint_interval == 0 || epoch_number == self.epochs {
                    let train_predictions = model.predict(split.train.inputs.view());
                    let evaluations = EvaluationSet::using_model(
                        &model,
                        &loss_function,
                        &accuracy,
                        &split,
                        Some(train_predictions.view()),
                    );
                    writer.record(&model, &evaluations)?;
                    wrote_this_epoch = true;
                }
            }

            if let Some(ref mut early_stopping) = early_stopping
                && let Some(validation) = &split.validation
            {
                let predictions = model.predict(validation.inputs.view());
                let loss = loss_function.compute(predictions.view(), validation.targets.view());

                if early_stopping.check(loss, &model) {
                    // Clear the progress bar before printing so it doesn't overwrite messages.
                    progression.done();
                    completed(&format!(
                        "Early stopping triggered at epoch {}",
                        style(epoch + 1).yellow()
                    ));
                    if self.restore_best_model {
                        model = early_stopping
                            .best_model
                            .as_ref()
                            .expect("Best model should be available")
                            .clone();
                        trace("Restored the best model observed during training");
                    }
                    // Compute once; reused for both the snapshot and the post-loop summary.
                    let stop_evals =
                        EvaluationSet::using_model(&model, &loss_function, &accuracy, &split, None);
                    // Write to history only when the interval block didn't already
                    // capture this epoch (avoids a duplicate snapshot).
                    if !wrote_this_epoch && let Some(ref mut writer) = writer {
                        writer.record(&model, &stop_evals)?;
                    }
                    final_evaluations = Some(stop_evals);
                    break;
                }
            }
        }

        let evaluations = final_evaluations.unwrap_or_else(|| {
            EvaluationSet::using_model(&model, &loss_function, &accuracy, &split, None)
        });

        completed(&format!(
            "{} | {}",
            style("Training completed").bright().green(),
            evaluations.summary()
        ));

        // 🗂️ SAVE THE TRAINED NETWORK
        save_model(path.with_file_name(&model_name), &model)?;

        // 🗂️ DISPLAY TRAINING HISTORY LOCATION
        if let Some(ref writer) = writer {
            saved_at(HISTORY_ICON, "TRAINING HISTORY", writer.dir());
        }

        Ok(())
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

    fn lr(&self) -> LearningRate {
        LearningRate::new(self.lr)
    }

    fn lr_min(&self) -> LearningRate {
        LearningRate::new(self.lr_min.unwrap_or(0.0))
    }

    fn step_size(&self) -> usize {
        self.steps.unwrap_or(self.epochs)
    }

    fn cycle_multiplier(&self) -> usize {
        self.cycle_multiplier.unwrap_or(1)
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

    fn make_scheduler(&self) -> Box<dyn Scheduler> {
        match self.scheduler {
            SchedulerType::Constant => Box::new(ConstantScheduler::new(self.lr())),
            SchedulerType::Cosine => {
                let cosine = CosineAnnealing::new(self.lr_min(), self.lr(), self.step_size());

                if self.warm_restarts {
                    Box::new(cosine.with_restarts(true, self.cycle_multiplier()))
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

    /// Builds the optimizer for this run.
    ///
    /// Note: optimizer state (e.g. Adam's moment estimates and step counter) is not
    /// persisted across runs. When resuming from a saved model with `--model`, a stateful
    /// optimizer starts fresh; the user is warned in [`Self::run`].
    fn make_optimizer(&self) -> Box<dyn Optimizer> {
        match self.optimizer {
            OptimizerType::Sgd => Box::new(StochasticGradientDescent::new(self.lr())),
            OptimizerType::Adam => Box::new(Adam::with_defaults(self.lr())),
        }
    }
}
