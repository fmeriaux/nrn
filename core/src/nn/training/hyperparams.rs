use super::callbacks::Callbacks;
use super::early_stopping::{EarlyStoppingConfig, EarlyStoppingConfigError};
use super::preprocessing::TrainingData;
use super::trainer::Trainer;
use crate::accuracies::{Accuracy, BINARY_ACCURACY, CATEGORICAL_ACCURACY};
use crate::data::Dataset;
use crate::data::scalers::{ScalerFeatureMismatch, ScalerKind, ScalerMethod};
use crate::gradients::{GradientClipping, GradientClippingError};
use crate::learning_rate::{LearningRate, LearningRateError};
use crate::loss_functions::{BinaryCrossEntropy, CategoricalCrossEntropy, LossFunction, Reduction};
use crate::model::{InputShapeMismatch, NeuralNetwork};
use crate::optimizers::{Adam, Optimizer, StochasticGradientDescent};
use crate::schedulers::{
    ConstantScheduler, CosineAnnealing, CosineAnnealingError, Scheduler, StepDecay, StepDecayError,
};
use crate::task::Task;
use crate::weight_decay::{WeightDecay, WeightDecayError};
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
    /// Instantiates the concrete optimizer for the shared learning rate `lr` and
    /// `weight_decay`. Adam applies decay as decoupled (AdamW); SGD as classic L2.
    fn instantiate(&self, lr: LearningRate, weight_decay: WeightDecay) -> Box<dyn Optimizer> {
        match self {
            OptimizerConfig::Sgd => Box::new(StochasticGradientDescent::new(lr, weight_decay)),
            OptimizerConfig::Adam => Box::new(Adam::with_defaults(lr, weight_decay)),
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
    /// Instantiates the concrete loss function for `task`: cross-entropy is the binary form
    /// for a single-logit task (binary or multi-label), the categorical form otherwise.
    fn instantiate(&self, task: &Task) -> Arc<dyn LossFunction> {
        match (self, task) {
            (LossConfig::CrossEntropy, Task::Binary | Task::MultiLabel { .. }) => {
                Arc::new(BinaryCrossEntropy::new(Reduction::Mean))
            }
            (LossConfig::CrossEntropy, Task::MultiClass { .. }) => {
                Arc::new(CategoricalCrossEntropy::new(Reduction::Mean))
            }
            (LossConfig::CrossEntropy, Task::Regression { .. }) => {
                panic!("Cross-entropy loss does not apply to a regression task.")
            }
        }
    }
}

/// Selects the accuracy metric for `task`: binary accuracy for a single-logit task
/// (binary or multi-label), categorical (argmax) accuracy for multi-class.
/// # Panics
/// When `task` is [`Regression`](Task::Regression), which has no accuracy metric.
fn accuracy_for(task: &Task) -> Arc<dyn Accuracy> {
    match task {
        Task::Binary | Task::MultiLabel { .. } => BINARY_ACCURACY.clone(),
        Task::MultiClass { .. } => CATEGORICAL_ACCURACY.clone(),
        Task::Regression { .. } => {
            panic!("Accuracy does not apply to a regression task.")
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
    /// Weight-decay coefficient passed to the optimizer.
    weight_decay: WeightDecay,
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
    /// Input scaling applied during [`build`](HyperParameters::build); `None`
    /// leaves the inputs untouched. The scaler is fitted on the train split only.
    scaler: Option<ScalerKind>,
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
    /// The weight decay was negative or non-finite.
    WeightDecay(WeightDecayError),
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
            HyperParametersError::WeightDecay(e) => write!(f, "{e}"),
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

impl From<WeightDecayError> for HyperParametersError {
    fn from(e: WeightDecayError) -> Self {
        HyperParametersError::WeightDecay(e)
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
        weight_decay: WeightDecay,
        optimizer: OptimizerConfig,
        scheduler: SchedulerConfig,
        clipping: GradientClipping,
        loss: LossConfig,
        early_stopping: Option<EarlyStoppingConfig>,
        val_ratio: f32,
        test_ratio: f32,
        seed: u64,
        scaler: Option<ScalerKind>,
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
            weight_decay,
            optimizer,
            scheduler,
            clipping,
            loss,
            early_stopping,
            val_ratio,
            test_ratio,
            seed,
            scaler,
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
        weight_decay: f32,
        optimizer: OptimizerConfig,
        scheduler: SchedulerConfig,
        clipping: GradientClipping,
        loss: LossConfig,
        early_stopping: Option<EarlyStoppingConfig>,
        val_ratio: f32,
        test_ratio: f32,
        seed: u64,
        scaler: Option<ScalerKind>,
    ) -> Result<Self, HyperParametersError> {
        HyperParameters::new(
            epochs,
            checkpoint_interval,
            batch_size,
            LearningRate::new(lr)?,
            WeightDecay::new(weight_decay)?,
            optimizer,
            scheduler,
            clipping,
            loss,
            early_stopping,
            val_ratio,
            test_ratio,
            seed,
            scaler,
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

    pub fn weight_decay(&self) -> WeightDecay {
        self.weight_decay
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

    pub fn scaler(&self) -> Option<ScalerKind> {
        self.scaler
    }

    /// Prepares `dataset` for training: splits it into train/validation/test using
    /// this spec's `val_ratio`/`test_ratio` (validated in [`new`](Self::new), so the
    /// split is always well-formed), then scales it. An explicit `scaler` is applied
    /// unchanged; otherwise this spec's [`scaler`](Self::scaler) kind is fitted on
    /// the train split.
    ///
    /// # Errors
    /// [`ScalerFeatureMismatch`] when an explicit `scaler` does not match the dataset's
    /// feature count (e.g. resuming with a dataset of a different width).
    pub fn prepare(
        &self,
        dataset: Dataset,
        scaler: Option<ScalerMethod>,
    ) -> Result<TrainingData, ScalerFeatureMismatch> {
        let split = dataset.split(self.val_ratio, self.test_ratio, self.seed);
        let scaler = scaler.or_else(|| self.scaler.map(|kind| split.train.fit_scaler(kind)));
        TrainingData::new(split, scaler)
    }

    /// Instantiates the runtime [`Trainer`] from this specification, binding it to
    /// the `model`, the `task`, the prepared [`TrainingData`], and `callbacks`. This
    /// is the one place declarative configs become concrete objects: the optimizer, scheduler,
    /// and loss are instantiated, and the `task` selects the loss and accuracy metric.
    ///
    /// The trainer starts from epoch 0; to resume a run, call
    /// [`Trainer::restore`](crate::training::Trainer::restore) on the result.
    ///
    /// # Errors
    /// [`InputShapeMismatch`] when `model`'s input size does not match the dataset's
    /// feature count. The two are supplied separately (e.g. resuming a run with a fresh
    /// dataset), so this is the one place their compatibility is checked; once past it,
    /// every forward pass over the split is guaranteed to fit.
    pub fn build(
        self,
        model: NeuralNetwork,
        task: Task,
        data: TrainingData,
        callbacks: Callbacks,
    ) -> Result<Trainer, InputShapeMismatch> {
        model.validate_inputs(data.split.train.inputs().view())?;

        let optimizer = self.optimizer.instantiate(self.lr, self.weight_decay);
        let scheduler = self
            .scheduler
            .instantiate(self.lr)
            .expect("schedule parameters were validated in HyperParameters::new");
        // Loss and accuracy are both derived from the task, not taken as config.
        let loss = self.loss.instantiate(&task);
        let accuracy = accuracy_for(&task);

        Ok(Trainer {
            model,
            callbacks,
            split: data.split,
            loss,
            accuracy,
            optimizer,
            scheduler,
            clipping: self.clipping,
            batch_size: self.batch_size,
            epochs: self.epochs,
            checkpoint_interval: self.checkpoint_interval,
            early_stopping: self.early_stopping,
            epoch_start: 0,
            seed: self.seed,
        })
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
            0.0,
            OptimizerConfig::Adam,
            scheduler,
            GradientClipping::None,
            LossConfig::CrossEntropy,
            None,
            val_ratio,
            test_ratio,
            0,
            None,
        )
    }

    fn message(result: Result<HyperParameters, HyperParametersError>) -> String {
        result.unwrap_err().to_string()
    }

    #[test]
    fn cross_entropy_resolves_to_binary_loss_for_a_binary_task() {
        // A binary task is a single logit: cross-entropy resolves to the binary form.
        // Mirrors accuracy_for's binary selection.
        let loss = LossConfig::CrossEntropy.instantiate(&Task::Binary);
        assert_eq!(loss.name(), "Binary-Cross-Entropy");
    }

    #[test]
    fn cross_entropy_resolves_to_categorical_loss_for_a_multi_class_task() {
        // A multi-class task is softmax logits: cross-entropy resolves to the categorical form.
        let loss = LossConfig::CrossEntropy.instantiate(&Task::MultiClass { n_classes: 3 });
        assert_eq!(loss.name(), "Categorical-Cross-Entropy");
    }

    #[test]
    fn accuracy_for_selects_binary_and_categorical_by_task() {
        // The two reachable classification task pick the matching metric; the selection
        // mirrors the loss resolution above.
        use ndarray::array;

        let binary = accuracy_for(&Task::Binary);
        assert_eq!(
            binary.compute(
                array![[0.5_f32, -0.5]].into_dyn().view(),
                array![[1.0_f32, 0.0]].into_dyn().view()
            ),
            100.0
        );

        let categorical = accuracy_for(&Task::MultiClass { n_classes: 3 });
        assert_eq!(
            categorical.compute(
                array![[2.0_f32], [1.0], [0.0]].into_dyn().view(),
                array![[1.0_f32], [0.0], [0.0]].into_dyn().view()
            ),
            100.0
        );
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
    fn rejects_negative_weight_decay() {
        let result = HyperParameters::from_values(
            10,
            1,
            None,
            0.01,
            -0.1,
            OptimizerConfig::Adam,
            SchedulerConfig::Constant,
            GradientClipping::None,
            LossConfig::CrossEntropy,
            None,
            0.1,
            0.1,
            0,
            None,
        );
        assert_eq!(
            message(result),
            "the weight decay must be a finite, non-negative value, got -0.1"
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

    /// A two-feature dataset whose values grow with the sample index, so MinMax
    /// scaling has a non-trivial effect.
    fn ramp_dataset() -> Dataset {
        use ndarray::{Array1, Array2};
        let features = Array2::from_shape_fn((10, 2), |(i, _)| i as f32);
        let labels = Array1::from_shape_fn(10, |i| (i % 2) as f32);
        Dataset::tabular(features, labels, None).unwrap()
    }

    fn spec_with_scaler(scaler: Option<ScalerKind>) -> HyperParameters {
        HyperParameters::from_values(
            5,
            1,
            None,
            0.01,
            0.0,
            OptimizerConfig::Adam,
            SchedulerConfig::Constant,
            GradientClipping::None,
            LossConfig::CrossEntropy,
            None,
            0.2,
            0.2,
            0,
            scaler,
        )
        .unwrap()
    }

    #[test]
    fn prepare_fits_the_configured_scaler_on_the_train_split() {
        let hp = spec_with_scaler(Some(ScalerKind::MinMax));
        let data = hp.prepare(ramp_dataset(), None).unwrap();

        // A scaler was fitted from the spec's kind, and applying it lands the train
        // inputs in [0, 1] — the signature of the MinMax kind that was configured.
        assert!(data.scaler().is_some());
        assert!(
            data.split
                .train
                .inputs()
                .iter()
                .all(|&v| (-1e-5..=1.0 + 1e-5).contains(&v))
        );
    }

    #[test]
    fn prepare_reuses_an_explicit_scaler_without_refitting() {
        use crate::data::scalers::MinMaxScaler;
        use ndarray::array;

        // A scaler fitted on a far wider range than the data (features on rows); reused
        // verbatim it keeps every input well below 0.5 (refitting on the train split
        // would not).
        let supplied = ScalerMethod::MinMax(
            MinMaxScaler::default().fit(array![[0.0, 100.0], [0.0, 100.0]].view()),
        );

        let hp = spec_with_scaler(None);
        let data = hp.prepare(ramp_dataset(), Some(supplied)).unwrap();

        assert!(data.split.train.inputs().iter().all(|&v| v < 0.5));
    }
}
