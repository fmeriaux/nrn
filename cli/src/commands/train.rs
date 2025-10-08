use crate::actions::*;
use crate::display::{Summary, completed, trace};
use crate::progression::Progression;
use clap::*;
use console::style;
use nrn::accuracies::{Accuracy, accuracy_for};
use nrn::checkpoints::Checkpoints;
use nrn::evaluation::EvaluationSet;
use nrn::loss_functions::{CROSS_ENTROPY_LOSS, LossFunction};
use nrn::optimizers::{Adam, Optimizer, StochasticGradientDescent};
use nrn::schedulers;
use nrn::schedulers::{ConstantScheduler, Scheduler, StepDecay};
use nrn::training::{EarlyStopping, GradientClipping, LearningRate};
use schedulers::CosineAnnealing;
use std::error::Error;
use std::fmt::Display;
use std::path::Path;
use std::sync::{Arc, Mutex};

#[derive(ValueEnum, Debug, Copy, Clone)]
enum OptimizerType {
    SGD,
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
            OptimizerType::SGD => write!(f, "Stochastic Gradient Descent (SGD)"),
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
    /// By default, this is set to the total number of epochs
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
}

impl TrainArgs {
    fn validate(&self) -> Result<(), Box<dyn Error>> {
        if self.epochs < 1 {
            return Err("The number of epochs must be greater than zero.".into());
        }

        if let Some(ref layers) = self.layers {
            if layers.iter().any(|&n| n == 0) {
                return Err("Each hidden layer must have at least one neuron.".into());
            }
        }

        if self.lr < 0.0 {
            return Err("The learning rate must be a non-negative value.".into());
        }

        if let Some(lr_min) = self.lr_min {
            if lr_min < 0.0 || lr_min >= self.lr {
                return Err("The minimum learning rate must be non-negative and less than the initial learning rate.".into());
            }
        }

        if self.decay_factor <= 0.0 {
            return Err("The decay factor must be a positive value.".into());
        }

        if let Some(steps) = self.steps {
            if steps < 1 {
                return Err("The step size must be greater than zero.".into());
            }
        }

        if self.clip_norm <= 0.0 {
            return Err("The gradient clipping norm must be a positive value.".into());
        }

        if let Some(clip_value) = self.clip_value {
            if clip_value <= 0.0 {
                return Err("The gradient clipping value must be a positive value.".into());
            }
        }

        if let Some(cycle_multiplier) = self.cycle_multiplier {
            if cycle_multiplier < 1 {
                return Err("The cycle multiplier must be at least 1.".into());
            }
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

        let optimizer: Arc<Mutex<dyn Optimizer>> = self.make_optimizer();
        trace(&format!(
            "Using {} optimizer with learning rate {}",
            style(self.optimizer).bold().blue(),
            style(self.lr).yellow()
        ));

        let scheduler: Arc<Mutex<dyn Scheduler>> = self.make_scheduler();
        if !matches!(self.scheduler, SchedulerType::Constant) {
            trace(&format!(
                "Learning rate scheduled by {}",
                style(self.scheduler).bold().blue()
            ));
        }

        let clipping = self.infer_clipping();
        trace(&format!("Using {}", clipping.summary()));

        let dataset = load_dataset(&self.dataset)?;
        let split = dataset
            .to_model_dataset()
            .split(self.val_ratio, self.test_ratio);
        completed(split.summary().as_str());

        let accuracy: Arc<dyn Accuracy> = accuracy_for(dataset.n_classes());

        // 🧠 NEURAL NETWORK INITIALIZATION
        let mut model = self
            .model
            .iter()
            .find_map(|file| load_model(file).ok())
            .unwrap_or_else(|| initialize_model_with(&dataset, self.layers.clone(), self.auto_layers));

        // 👨‍🎓 TRAINING LOOP
        let mut checkpoints: Option<Checkpoints> = Checkpoints::by_interval(self.checkpoint_interval, self.epochs);

        if let Some(ref mut checkpoints) = checkpoints {
            trace(&format!(
                "Recording a checkpoint every {} epochs",
                style(checkpoints.interval).yellow()
            ));

            let evaluations = EvaluationSet::using_model(&model, &loss_function, &accuracy, &split, None);

            checkpoints.record(&model, &evaluations);
        };

        let mut early_stopping = self.early_stopping();

        let progression = Progression::new(self.epochs, "Training");
        for epoch in progression.iter() {
            let train_predictions = model.train(
                &split.train,
                &loss_function,
                &optimizer,
                &scheduler,
                &clipping,
            );

            if let Some(ref mut checkpoints) = checkpoints {
                let epoch_number = epoch + 1;
                if epoch_number % checkpoints.interval == 0 || epoch_number == self.epochs {
                    let evaluations = EvaluationSet::using_model(&model, &loss_function, &accuracy, &split, Some(train_predictions.view()));
                    checkpoints.record(&model, &evaluations);
                }
            }

            if let Some(ref mut early_stopping) = early_stopping {
                if let Some(validation) = &split.validation {
                    let predictions = model.predict(validation.inputs.view());
                    let loss = loss_function.compute(predictions.view(), validation.targets.view());

                    if early_stopping.check(loss, &model) {
                        trace(&format!(
                            "Early stopping triggered at epoch {}",
                            style(epoch + 1).yellow()
                        ));
                        progression.done();
                        break;
                    }
                }
            }
        }

        let evaluations = EvaluationSet::using_model(&model, &loss_function, &accuracy, &split, None);

        completed(&format!(
            "{} | {}",
            style("Training completed").bright().green(),
            evaluations.summary()
        ));

        // 🗂️ SAVE THE TRAINED NETWORK
        let path = Path::new(&self.dataset);
        let dataset_name = get_file_stem(&path);
        let model_name = format!("model-{}", dataset_name);
        save_model(path.with_file_name(&model_name), &model)?;

        // 🗂️ SAVE THE CHECKPOINTS
        if let Some(ref checkpoints) = checkpoints {
            save_checkpoints(
                path.with_file_name(format!("training-{}", model_name)),
                checkpoints,
            )?;
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

    fn make_scheduler(&self) -> Arc<Mutex<dyn Scheduler>> {
        match self.scheduler {
            SchedulerType::Constant => Arc::new(Mutex::new(ConstantScheduler::new(self.lr()))),
            SchedulerType::Cosine => {
                let cosine = CosineAnnealing::new(self.lr_min(), self.lr(), self.step_size());

                if self.warm_restarts {
                    Arc::new(Mutex::new(
                        cosine.with_restarts(true, self.cycle_multiplier()),
                    ))
                } else {
                    Arc::new(Mutex::new(cosine))
                }
            }
            SchedulerType::Step => Arc::new(Mutex::new(StepDecay::new(
                self.lr(),
                self.step_size(),
                self.decay_factor,
            ))),
        }
    }

    fn make_optimizer(&self) -> Arc<Mutex<dyn Optimizer>> {
        match self.optimizer {
            OptimizerType::SGD => Arc::new(Mutex::new(StochasticGradientDescent::new(self.lr()))),
            OptimizerType::Adam => Arc::new(Mutex::new(Adam::with_defaults(self.lr()))),
        }
    }
}
