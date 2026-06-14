use super::callbacks::Callbacks;
use super::early_stopping::EarlyStoppingConfig;
use super::trainer::Trainer;
use crate::data::ModelSplit;
use crate::gradients::GradientClipping;
use crate::learning_rate::LearningRate;
use crate::loss_functions::{CROSS_ENTROPY_LOSS, LossFunction};
use crate::model::NeuralNetwork;
use crate::optimizers::{Adam, Optimizer, StochasticGradientDescent};
use crate::schedulers::{
    ConstantScheduler, CosineAnnealing, CosineAnnealingError, Scheduler, StepDecay, StepDecayError,
};
use std::fmt;
use std::sync::Arc;

/// Declarative choice of parameter-update rule. Turned into a concrete
/// [`Optimizer`] by [`HyperParameters::build`].
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizerConfig {
    Sgd,
    Adam,
}

/// Declarative learning-rate schedule. The maximum learning rate is the run's
/// shared [`HyperParameters::lr`]; the variants carry only the schedule-specific
/// parameters. Turned into a concrete [`Scheduler`] by [`HyperParameters::build`].
#[derive(Debug, Clone, PartialEq)]
pub enum SchedulerConfig {
    /// Keeps the learning rate fixed at `lr`.
    Constant,
    /// Cosine annealing from `lr` down to `lr_min` over `steps` epochs.
    Cosine {
        lr_min: f32,
        steps: usize,
        warm_restarts: bool,
        cycle_multiplier: usize,
    },
    /// Step decay: multiply `lr` by `decay_factor` every `steps` epochs.
    Step { decay_factor: f32, steps: usize },
}

/// Declarative choice of loss function. Turned into a concrete [`LossFunction`]
/// by [`HyperParameters::build`].
#[derive(Debug, Clone, PartialEq)]
pub enum LossConfig {
    CrossEntropy,
}

impl OptimizerConfig {
    /// Instantiates the concrete optimizer for the shared learning rate `lr`.
    fn instantiate(&self, lr: LearningRate) -> Box<dyn Optimizer> {
        match self {
            OptimizerConfig::Sgd => Box::new(StochasticGradientDescent::new(lr)),
            OptimizerConfig::Adam => Box::new(Adam::with_defaults(lr)),
        }
    }
}

impl SchedulerConfig {
    /// Instantiates the concrete scheduler, using `lr` as the (maximum) learning rate.
    /// # Errors
    /// Returns [`HyperParametersError`] when the schedule parameters are invalid
    /// (e.g. `lr_min >= lr`, zero `steps`, or an out-of-range `decay_factor`).
    fn instantiate(&self, lr: LearningRate) -> Result<Box<dyn Scheduler>, HyperParametersError> {
        match self {
            SchedulerConfig::Constant => Ok(Box::new(ConstantScheduler::new(lr))),
            SchedulerConfig::Cosine {
                lr_min,
                steps,
                warm_restarts,
                cycle_multiplier,
            } => Ok(Box::new(CosineAnnealing::from_values(
                *lr_min,
                lr.value(),
                *steps,
                *warm_restarts,
                *cycle_multiplier,
            )?)),
            SchedulerConfig::Step {
                decay_factor,
                steps,
            } => Ok(Box::new(StepDecay::new(lr, *steps, *decay_factor)?)),
        }
    }
}

impl LossConfig {
    /// Instantiates the concrete loss function.
    fn instantiate(&self) -> Arc<dyn LossFunction> {
        match self {
            LossConfig::CrossEntropy => CROSS_ENTROPY_LOSS.clone(),
        }
    }
}

/// The declarative specification of a training run: the single source of truth
/// for what to train and how. Holds plain configuration (no instantiated trait
/// objects), validates its cross-field invariants on construction, and builds
/// the runtime [`Trainer`] via [`build`](HyperParameters::build).
///
/// Its serializable mirror is [`crate::io::hyperparams::HyperParametersRecord`].
#[derive(Debug, Clone)]
pub struct HyperParameters {
    /// Number of epochs to train; must be at least `1`.
    epochs: usize,
    /// Interval (in epochs) between scheduled evaluations/checkpoints; `0` disables them.
    checkpoint_interval: usize,
    /// Mini-batch size; `None` runs full-batch gradient descent each epoch.
    batch_size: Option<usize>,
    /// Shared (maximum) learning rate, used by both the optimizer and the scheduler.
    lr: LearningRate,
    /// Parameter-update rule.
    optimizer: OptimizerConfig,
    /// Learning-rate schedule, stepped once per epoch.
    scheduler: SchedulerConfig,
    /// Gradient-clipping strategy applied before each parameter update.
    clipping: GradientClipping,
    /// Loss function minimized during training.
    loss: LossConfig,
    /// Early-stopping policy; `None` always trains for the full `epochs` count.
    early_stopping: Option<EarlyStoppingConfig>,
    /// Fraction of the dataset held out for validation. Part of the run's
    /// identity (the resulting split), consumed when building the split rather
    /// than by the [`Trainer`].
    val_ratio: f32,
    /// Fraction of the dataset held out for testing. Part of the run's identity
    /// (the resulting split), consumed when building the split rather than by
    /// the [`Trainer`].
    test_ratio: f32,
}

/// Returned by [`HyperParameters::new`] when an invariant is violated, either a
/// cross-field one or an invalid schedule parameter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HyperParametersError {
    /// `epochs` was zero.
    ZeroEpochs,
    /// `batch_size` was `Some(0)`.
    ZeroBatchSize,
    /// `val_ratio` was outside `[0.0, 1.0)`.
    InvalidValRatio(f32),
    /// `test_ratio` was outside `(0.0, 1.0)`.
    InvalidTestRatio(f32),
    /// `val_ratio + test_ratio` was not less than `1.0`.
    SplitRatiosTooLarge { val_ratio: f32, test_ratio: f32 },
    /// The cosine schedule parameters were invalid.
    Cosine(CosineAnnealingError),
    /// The step-decay schedule parameters were invalid.
    Step(StepDecayError),
}

impl fmt::Display for HyperParametersError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HyperParametersError::ZeroEpochs => {
                write!(f, "the number of epochs must be greater than zero")
            }
            HyperParametersError::ZeroBatchSize => {
                write!(f, "the batch size must be greater than zero")
            }
            HyperParametersError::InvalidValRatio(ratio) => {
                write!(f, "the validation ratio must be in [0.0, 1.0), got {ratio}")
            }
            HyperParametersError::InvalidTestRatio(ratio) => {
                write!(f, "the test ratio must be in (0.0, 1.0), got {ratio}")
            }
            HyperParametersError::SplitRatiosTooLarge {
                val_ratio,
                test_ratio,
            } => write!(
                f,
                "the sum of validation and test ratios must be less than 1.0, got val={val_ratio}, test={test_ratio}"
            ),
            HyperParametersError::Cosine(e) => write!(f, "{e}"),
            HyperParametersError::Step(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for HyperParametersError {}

impl From<CosineAnnealingError> for HyperParametersError {
    fn from(e: CosineAnnealingError) -> Self {
        HyperParametersError::Cosine(e)
    }
}

impl From<StepDecayError> for HyperParametersError {
    fn from(e: StepDecayError) -> Self {
        HyperParametersError::Step(e)
    }
}

impl HyperParameters {
    /// Creates a fully validated specification.
    ///
    /// Every invariant is checked up front — including that the schedule
    /// parameters are constructible against `lr` — so an existing
    /// [`HyperParameters`] is always buildable.
    /// # Errors
    /// Returns [`HyperParametersError`] if any invariant is violated.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        epochs: usize,
        checkpoint_interval: usize,
        batch_size: Option<usize>,
        lr: LearningRate,
        optimizer: OptimizerConfig,
        scheduler: SchedulerConfig,
        clipping: GradientClipping,
        loss: LossConfig,
        early_stopping: Option<EarlyStoppingConfig>,
        val_ratio: f32,
        test_ratio: f32,
    ) -> Result<Self, HyperParametersError> {
        if epochs < 1 {
            return Err(HyperParametersError::ZeroEpochs);
        }
        if batch_size == Some(0) {
            return Err(HyperParametersError::ZeroBatchSize);
        }
        if !(0.0..1.0).contains(&val_ratio) {
            return Err(HyperParametersError::InvalidValRatio(val_ratio));
        }
        if test_ratio <= 0.0 || test_ratio >= 1.0 {
            return Err(HyperParametersError::InvalidTestRatio(test_ratio));
        }
        if val_ratio + test_ratio >= 1.0 {
            return Err(HyperParametersError::SplitRatiosTooLarge {
                val_ratio,
                test_ratio,
            });
        }

        // Validate the schedule parameters against `lr` up front; the resulting
        // scheduler is rebuilt (infallibly) in `build`.
        scheduler.instantiate(lr)?;

        Ok(HyperParameters {
            epochs,
            checkpoint_interval,
            batch_size,
            lr,
            optimizer,
            scheduler,
            clipping,
            loss,
            early_stopping,
            val_ratio,
            test_ratio,
        })
    }

    pub fn epochs(&self) -> usize {
        self.epochs
    }

    pub fn checkpoint_interval(&self) -> usize {
        self.checkpoint_interval
    }

    pub fn batch_size(&self) -> Option<usize> {
        self.batch_size
    }

    pub fn lr(&self) -> LearningRate {
        self.lr
    }

    pub fn optimizer(&self) -> &OptimizerConfig {
        &self.optimizer
    }

    pub fn scheduler(&self) -> &SchedulerConfig {
        &self.scheduler
    }

    pub fn clipping(&self) -> &GradientClipping {
        &self.clipping
    }

    pub fn loss(&self) -> &LossConfig {
        &self.loss
    }

    pub fn early_stopping(&self) -> Option<&EarlyStoppingConfig> {
        self.early_stopping.as_ref()
    }

    pub fn val_ratio(&self) -> f32 {
        self.val_ratio
    }

    pub fn test_ratio(&self) -> f32 {
        self.test_ratio
    }

    /// Instantiates the runtime [`Trainer`] from this specification, binding it
    /// to the `model`, `split`, `callbacks`, and starting epoch. This is the one
    /// place declarative configs become concrete optimizer/scheduler/loss objects.
    pub fn build(
        self,
        model: NeuralNetwork,
        split: ModelSplit,
        callbacks: Callbacks,
        epoch_start: usize,
    ) -> Trainer {
        let optimizer = self.optimizer.instantiate(self.lr);
        let scheduler = self
            .scheduler
            .instantiate(self.lr)
            .expect("schedule parameters were validated in HyperParameters::new");
        let loss = self.loss.instantiate();

        Trainer {
            model,
            callbacks,
            split,
            loss,
            optimizer,
            scheduler,
            clipping: self.clipping,
            batch_size: self.batch_size,
            epochs: self.epochs,
            checkpoint_interval: self.checkpoint_interval,
            early_stopping: self.early_stopping,
            epoch_start,
        }
    }
}
