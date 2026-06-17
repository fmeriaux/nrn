use super::callbacks::Callbacks;
use super::early_stopping::{EarlyStoppingConfig, EarlyStoppingConfigError};
use super::trainer::Trainer;
use crate::data::ModelDataset;
use crate::gradients::{GradientClipping, GradientClippingError};
use crate::learning_rate::{LearningRate, LearningRateError};
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
    /// identity: [`build`](HyperParameters::build) applies it (with `test_ratio`)
    /// to split the dataset.
    val_ratio: f32,
    /// Fraction of the dataset held out for testing. Part of the run's identity:
    /// [`build`](HyperParameters::build) applies it (with `val_ratio`) to split
    /// the dataset.
    test_ratio: f32,
    /// Seed for the run's randomness — the mini-batch shuffle here, and (by
    /// convention) the model's weight initialization at the call site. Part of the
    /// run's identity so a run is always reproducible and the seed is recorded.
    seed: u64,
}

/// The single error type for constructing a [`HyperParameters`] spec, whether
/// from already-validated components ([`HyperParameters::new`]) or from raw
/// values ([`HyperParameters::from_values`]). It aggregates both the cross-field
/// invariants and every component-level validation failure (learning rate,
/// clipping, schedule, early stopping), so callers that assemble a spec from raw
/// inputs need a single error rather than their own union.
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
    /// The learning rate was negative or non-finite.
    LearningRate(LearningRateError),
    /// The gradient-clipping bounds were invalid.
    Clipping(GradientClippingError),
    /// The early-stopping parameters were invalid.
    EarlyStopping(EarlyStoppingConfigError),
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
            HyperParametersError::LearningRate(e) => write!(f, "{e}"),
            HyperParametersError::Clipping(e) => write!(f, "{e}"),
            HyperParametersError::EarlyStopping(e) => write!(f, "{e}"),
            HyperParametersError::Cosine(e) => write!(f, "{e}"),
            HyperParametersError::Step(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for HyperParametersError {}

impl From<LearningRateError> for HyperParametersError {
    fn from(e: LearningRateError) -> Self {
        HyperParametersError::LearningRate(e)
    }
}

impl From<GradientClippingError> for HyperParametersError {
    fn from(e: GradientClippingError) -> Self {
        HyperParametersError::Clipping(e)
    }
}

impl From<EarlyStoppingConfigError> for HyperParametersError {
    fn from(e: EarlyStoppingConfigError) -> Self {
        HyperParametersError::EarlyStopping(e)
    }
}

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
        seed: u64,
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
            seed,
        })
    }

    /// Creates a fully validated specification from a raw learning rate.
    ///
    /// Convenience constructor for callers that hold an unvalidated `f32` learning
    /// rate (the CLI, a persisted record): it builds the [`LearningRate`] and
    /// delegates to [`new`](HyperParameters::new), folding the learning-rate
    /// validation into the same [`HyperParametersError`]. The `clipping` and
    /// `early_stopping` components are taken already built, since their raw form
    /// is caller-specific; their construction errors convert into
    /// [`HyperParametersError`] via `?` at the call site.
    /// # Errors
    /// Returns [`HyperParametersError`] if any invariant is violated, including a
    /// negative or non-finite `lr`.
    #[allow(clippy::too_many_arguments)]
    pub fn from_values(
        epochs: usize,
        checkpoint_interval: usize,
        batch_size: Option<usize>,
        lr: f32,
        optimizer: OptimizerConfig,
        scheduler: SchedulerConfig,
        clipping: GradientClipping,
        loss: LossConfig,
        early_stopping: Option<EarlyStoppingConfig>,
        val_ratio: f32,
        test_ratio: f32,
        seed: u64,
    ) -> Result<Self, HyperParametersError> {
        HyperParameters::new(
            epochs,
            checkpoint_interval,
            batch_size,
            LearningRate::new(lr)?,
            optimizer,
            scheduler,
            clipping,
            loss,
            early_stopping,
            val_ratio,
            test_ratio,
            seed,
        )
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

    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Instantiates the runtime [`Trainer`] from this specification, binding it
    /// to the `model` and `callbacks`. This is the one place declarative configs
    /// become concrete objects: the optimizer/scheduler/loss are instantiated, and
    /// `dataset` is split into train/validation/test using this spec's own
    /// `val_ratio`/`test_ratio` (validated in [`new`](HyperParameters::new), so the
    /// split is always well-formed).
    ///
    /// The trainer starts from epoch 0; to resume a run, call
    /// [`Trainer::restore`](crate::training::Trainer::restore) on the result.
    pub fn build(
        self,
        model: NeuralNetwork,
        dataset: ModelDataset,
        callbacks: Callbacks,
    ) -> Trainer {
        let split = dataset.split(self.val_ratio, self.test_ratio);
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
            epoch_start: 0,
            seed: self.seed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a spec varying only the fields a validation test cares about; the
    /// rest are fixed to valid defaults (Adam optimizer, no clipping, cross-entropy).
    fn try_build(
        epochs: usize,
        batch_size: Option<usize>,
        lr: f32,
        scheduler: SchedulerConfig,
        val_ratio: f32,
        test_ratio: f32,
    ) -> Result<HyperParameters, HyperParametersError> {
        HyperParameters::from_values(
            epochs,
            1,
            batch_size,
            lr,
            OptimizerConfig::Adam,
            scheduler,
            GradientClipping::None,
            LossConfig::CrossEntropy,
            None,
            val_ratio,
            test_ratio,
            0,
        )
    }

    fn message(result: Result<HyperParameters, HyperParametersError>) -> String {
        result.unwrap_err().to_string()
    }

    #[test]
    fn rejects_zero_epochs() {
        assert_eq!(
            message(try_build(
                0,
                None,
                0.01,
                SchedulerConfig::Constant,
                0.1,
                0.1
            )),
            "the number of epochs must be greater than zero"
        );
    }

    #[test]
    fn rejects_zero_batch_size() {
        assert_eq!(
            message(try_build(
                10,
                Some(0),
                0.01,
                SchedulerConfig::Constant,
                0.1,
                0.1
            )),
            "the batch size must be greater than zero"
        );
    }

    #[test]
    fn rejects_out_of_range_val_ratio() {
        assert_eq!(
            message(try_build(
                10,
                None,
                0.01,
                SchedulerConfig::Constant,
                1.0,
                0.1
            )),
            "the validation ratio must be in [0.0, 1.0), got 1"
        );
    }

    #[test]
    fn rejects_out_of_range_test_ratio() {
        assert_eq!(
            message(try_build(
                10,
                None,
                0.01,
                SchedulerConfig::Constant,
                0.1,
                0.0
            )),
            "the test ratio must be in (0.0, 1.0), got 0"
        );
    }

    #[test]
    fn rejects_split_ratios_summing_to_one_or_more() {
        assert_eq!(
            message(try_build(
                10,
                None,
                0.01,
                SchedulerConfig::Constant,
                0.6,
                0.6
            )),
            "the sum of validation and test ratios must be less than 1.0, got val=0.6, test=0.6"
        );
    }

    #[test]
    fn rejects_negative_learning_rate() {
        assert_eq!(
            message(try_build(
                10,
                None,
                -1.0,
                SchedulerConfig::Constant,
                0.1,
                0.1
            )),
            "the learning rate must be a finite, non-negative value, got -1"
        );
    }

    #[test]
    fn rejects_cosine_max_not_greater_than_min() {
        let scheduler = SchedulerConfig::Cosine {
            lr_min: 0.02,
            steps: 5,
            warm_restarts: false,
            cycle_multiplier: 1,
        };
        assert_eq!(
            message(try_build(10, None, 0.01, scheduler, 0.1, 0.1)),
            "the maximum learning rate must be greater than the minimum, got min=0.02, max=0.01"
        );
    }

    #[test]
    fn rejects_cosine_zero_steps() {
        let scheduler = SchedulerConfig::Cosine {
            lr_min: 0.001,
            steps: 0,
            warm_restarts: false,
            cycle_multiplier: 1,
        };
        assert_eq!(
            message(try_build(10, None, 0.01, scheduler, 0.1, 0.1)),
            "the step size must be greater than zero"
        );
    }

    #[test]
    fn rejects_cosine_zero_cycle_multiplier() {
        let scheduler = SchedulerConfig::Cosine {
            lr_min: 0.001,
            steps: 5,
            warm_restarts: true,
            cycle_multiplier: 0,
        };
        assert_eq!(
            message(try_build(10, None, 0.01, scheduler, 0.1, 0.1)),
            "the cycle multiplier must be at least 1"
        );
    }

    #[test]
    fn rejects_cosine_invalid_min_learning_rate() {
        let scheduler = SchedulerConfig::Cosine {
            lr_min: -1.0,
            steps: 5,
            warm_restarts: false,
            cycle_multiplier: 1,
        };
        assert_eq!(
            message(try_build(10, None, 0.01, scheduler, 0.1, 0.1)),
            "the learning rate must be a finite, non-negative value, got -1"
        );
    }

    #[test]
    fn rejects_step_invalid_decay_factor() {
        let scheduler = SchedulerConfig::Step {
            decay_factor: 1.5,
            steps: 5,
        };
        assert_eq!(
            message(try_build(10, None, 0.01, scheduler, 0.1, 0.1)),
            "the decay factor must be in (0, 1), got 1.5"
        );
    }

    #[test]
    fn rejects_step_zero_steps() {
        let scheduler = SchedulerConfig::Step {
            decay_factor: 0.5,
            steps: 0,
        };
        assert_eq!(
            message(try_build(10, None, 0.01, scheduler, 0.1, 0.1)),
            "the step size must be greater than zero"
        );
    }

    #[test]
    fn clipping_error_is_wrapped_and_displayed() {
        let range_err = GradientClipping::value(1.0, 0.0).unwrap_err();
        assert_eq!(
            HyperParametersError::from(range_err).to_string(),
            "the gradient clipping range must satisfy min < max, got min=1, max=0"
        );
        let norm_err = GradientClipping::norm(-1.0).unwrap_err();
        assert_eq!(
            HyperParametersError::from(norm_err).to_string(),
            "the gradient clipping norm must be a positive value, got -1"
        );
    }

    #[test]
    fn early_stopping_error_is_wrapped_and_displayed() {
        let err = EarlyStoppingConfig::new(0, false).unwrap_err();
        assert_eq!(
            HyperParametersError::from(err).to_string(),
            "early stopping patience must be greater than zero"
        );
    }
}
