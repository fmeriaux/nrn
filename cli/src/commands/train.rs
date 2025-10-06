use crate::actions;
use crate::display::{Summary, completed, trace};
use crate::progression::Progression;
use clap::*;
use console::style;
use ndarray::ArrayView2;
use nrn::accuracies::{Accuracy, BINARY_ACCURACY, MULTI_CLASS_ACCURACY};
use nrn::activations::{RELU, SIGMOID, SOFTMAX};
use nrn::data::SplitDataset;
use nrn::loss_functions::{CROSS_ENTROPY_LOSS, LossFunction};
use nrn::model::{NeuralNetwork, NeuronLayerSpec};
use nrn::optimizers::{Adam, Optimizer, StochasticGradientDescent};
use nrn::schedulers;
use nrn::schedulers::{ConstantScheduler, Scheduler, StepDecay};
use nrn::training::{GradientClipping, History, LearningRate};
use schedulers::CosineAnnealing;
use std::error::Error;
use std::fmt::Display;
use std::iter::once;
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
    #[arg(short, long, value_parser=1..)]
    epochs: usize,

    #[arg(short = 'k', long, default_value_t = 10, value_parser=0..)]
    /// Specify the checkpoint interval for saving the model state,
    /// if set to 0, no checkpoints will be saved
    checkpoint_interval: usize,

    /// Specify the hidden layers of the model when a new model is initialized
    #[arg(long, value_delimiter = ',', conflicts_with = "model")]
    layers: Option<Vec<usize>>,

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
    #[arg(long, requires = "warm_restarts", value_parser = 1..10)]
    cycle_multiplier: Option<usize>,

    /// Specify the decay factor for schedulers that require it (e.g., step)
    #[arg(long, requires = "scheduler", value_parser = 0..1, default_value_t = 0.1)]
    decay_factor: f32,

    /// Specify the step size for learning rate schedulers that require it:
    /// - cosine: number of epochs to complete a full cosine cycle
    /// - step: number of epochs between each learning rate decay
    /// By default, this is set to the total number of epochs
    #[arg(long, requires = "scheduler", value_parser = 1..)]
    steps: Option<usize>,

    /// Specify the learning rate for the training process
    #[arg(long, default_value_t = 0.001, value_parser = 0..=1)]
    lr: f32,

    /// Specify the minimum learning rate for schedulers that support it (e.g., cosine)
    #[arg(long, requires = "scheduler")]
    lr_min: Option<f32>,

    /// Specify the gradient clipping norm to prevent exploding gradients
    #[arg(long, default_value_t = 1.0, conflicts_with_all = &["clip_value", "no_clip"], value_parser = 0..
    )]
    clip_norm: f32,

    /// Specify the gradient clipping value to prevent exploding gradients.
    /// This performs element-wise clipping: each gradient component is clipped individually to the symmetric range [-value, value].
    #[arg(long, conflicts_with_all = &["clip_norm", "no_clip"], value_parser = 0..)]
    clip_value: Option<f32>,

    /// Disable gradient clipping
    #[arg(long, conflicts_with_all = &["clip_norm", "clip_value"])]
    no_clip: bool,
}

impl TrainArgs {
    pub fn run(self) -> Result<(), Box<dyn Error>> {
        let loss_function: Arc<dyn LossFunction> = CROSS_ENTROPY_LOSS.clone();

        trace(&format!(
            "Using {} loss function",
            style("Cross-Entropy").bold().blue()
        ));

        let optimizer: Arc<Mutex<dyn Optimizer>> = self.make_optimizer();
        let scheduler: Arc<Mutex<dyn Scheduler>> = self.make_scheduler();

        trace(&format!(
            "Using {} optimizer with learning rate {}",
            style(self.optimizer).bold().blue(),
            style(self.lr).yellow()
        ));

        if !matches!(self.scheduler, SchedulerType::Constant) {
            trace(&format!(
                "Learning rate scheduled by {}",
                style(self.scheduler).bold().blue()
            ));
        }

        let clipping = self.infer_clipping();
        trace(&format!("Using {}", clipping.summary()));

        let SplitDataset { train, test } = actions::load_dataset(&self.dataset)?;

        let accuracy: Arc<dyn Accuracy> = select_accuracy(train.n_classes());

        // 🧠 NEURAL NETWORK INITIALIZATION
        let (train_inputs, train_targets) = train.to_model_shape();
        let (test_inputs, test_targets) = test.to_model_shape();
        let train_targets = train_targets.view();
        let test_targets = test_targets.view();

        let layer_specs: Vec<NeuronLayerSpec> = self
            .layers
            .unwrap_or_default()
            .into_iter()
            .map(|neurons| NeuronLayerSpec {
                neurons,
                activation: RELU.clone(),
            })
            .chain(once(create_output_layer(train.n_classes())))
            .collect();

        let mut model = self
            .model
            .iter()
            .find_map(|file| actions::load_model(file).ok())
            .unwrap_or_else(|| actions::initialize_model(train_inputs.nrows(), &layer_specs));

        // 👨‍🎓 TRAINING LOOP
        let mut history: Option<History> =
            History::by_interval(self.checkpoint_interval, self.epochs);

        // Closure to record the training history at each checkpoint
        let record_history =
            |model: &NeuralNetwork, history: &mut History, train_predictions: ArrayView2<f32>| {
                let test_predictions = model.predict(test_inputs);
                history.checkpoint(
                    model,
                    &loss_function,
                    &accuracy,
                    train_predictions,
                    train_targets,
                    test_predictions.view(),
                    test_targets,
                );
            };

        if let Some(ref mut history) = history {
            trace(&format!(
                "Recording {} checkpoints, one every {} epochs",
                style(history.model.capacity()).yellow(),
                style(history.interval).yellow()
            ));

            let train_activations = model.predict(train_inputs);
            record_history(&model, history, train_activations.view());
        };

        let progression = Progression::new(self.epochs, "Training");
        for epoch in progression.iter() {
            let train_predictions = model.train(
                train_inputs,
                train_targets,
                &loss_function,
                &optimizer,
                &scheduler,
                &clipping,
            );

            if let Some(ref mut history) = history {
                let epoch_number = epoch + 1;
                if epoch_number % history.interval == 0 || epoch_number == self.epochs {
                    record_history(&model, history, train_predictions.view());
                }
            }
        }

        let train_predictions = model.predict(train_inputs);
        let train_predictions = train_predictions.view();

        completed(&format!(
            "{} | Loss: {} | Accuracy: Train={}, Test: {}",
            style("Training completed").bright().green(),
            style(loss_function.compute(train_predictions, train_targets)).yellow(),
            style(accuracy.compute(train_predictions, train_targets)).yellow(),
            style(accuracy.compute(model.predict(test_inputs).view(), test_targets)).yellow()
        ));

        // 🗂️ SAVE THE TRAINED NETWORK
        let model_file = format!("model-{}", self.dataset);
        actions::save_model(&model_file, &model)?;

        // 🗂️ SAVE THE TRAINING HISTORY
        if let Some(ref history) = history {
            actions::save_training_history(&format!("training-{}", model_file), history)?;
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

fn select_accuracy(n_classes: usize) -> Arc<dyn Accuracy> {
    if n_classes > 2 {
        MULTI_CLASS_ACCURACY.clone()
    } else {
        BINARY_ACCURACY.clone()
    }
}

fn create_output_layer(n_classes: usize) -> NeuronLayerSpec {
    if n_classes > 2 {
        NeuronLayerSpec {
            neurons: n_classes,
            activation: SOFTMAX.clone(),
        }
    } else {
        NeuronLayerSpec {
            neurons: 1,
            activation: SIGMOID.clone(),
        }
    }
}
