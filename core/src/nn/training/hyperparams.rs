use super::early_stopping::{EarlyStopping, EarlyStoppingConfig};
use crate::gradients::GradientClipping;
use crate::loss_functions::LossFunction;
use crate::model::NeuralNetwork;
use crate::optimizers::Optimizer;
use crate::schedulers::Scheduler;
use std::fmt;
use std::sync::Arc;

/// The full specification of a training run, embedded in
/// [`crate::training::TrainingLoop`] and passed by reference to
/// [`crate::training::TrainingCallback::on_train_start`].
pub struct HyperParams {
    epochs: usize,
    /// Interval (in epochs) between scheduled evaluations/checkpoints; `0` disables them.
    checkpoint_interval: usize,
    batch_size: Option<usize>,
    loss: Arc<dyn LossFunction>,
    optimizer: Box<dyn Optimizer>,
    scheduler: Box<dyn Scheduler>,
    clipping: GradientClipping,
    early_stopping: Option<EarlyStoppingConfig>,
    /// Fraction of the dataset held out for validation. Part of the run's
    /// identity (the resulting split), not consumed by [`crate::training::TrainingLoop`].
    val_ratio: f32,
    /// Fraction of the dataset held out for testing. Part of the run's
    /// identity (the resulting split), not consumed by [`crate::training::TrainingLoop`].
    test_ratio: f32,
}

/// Returned by [`HyperParams::new`] when a cross-field invariant is violated.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HyperParamsError {
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
}

impl fmt::Display for HyperParamsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HyperParamsError::ZeroEpochs => {
                write!(f, "the number of epochs must be greater than zero")
            }
            HyperParamsError::ZeroBatchSize => {
                write!(f, "the batch size must be greater than zero")
            }
            HyperParamsError::InvalidValRatio(ratio) => {
                write!(f, "the validation ratio must be in [0.0, 1.0), got {ratio}")
            }
            HyperParamsError::InvalidTestRatio(ratio) => {
                write!(f, "the test ratio must be in (0.0, 1.0), got {ratio}")
            }
            HyperParamsError::SplitRatiosTooLarge {
                val_ratio,
                test_ratio,
            } => write!(
                f,
                "the sum of validation and test ratios must be less than 1.0, got val={val_ratio}, test={test_ratio}"
            ),
        }
    }
}

impl std::error::Error for HyperParamsError {}

impl HyperParams {
    /// Creates a new hyperparameter spec, validating its cross-field invariants.
    /// # Errors
    /// Returns [`HyperParamsError`] if any invariant is violated.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        epochs: usize,
        checkpoint_interval: usize,
        batch_size: Option<usize>,
        loss: Arc<dyn LossFunction>,
        optimizer: Box<dyn Optimizer>,
        scheduler: Box<dyn Scheduler>,
        clipping: GradientClipping,
        early_stopping: Option<EarlyStoppingConfig>,
        val_ratio: f32,
        test_ratio: f32,
    ) -> Result<Self, HyperParamsError> {
        if epochs < 1 {
            return Err(HyperParamsError::ZeroEpochs);
        }
        if batch_size == Some(0) {
            return Err(HyperParamsError::ZeroBatchSize);
        }
        if !(0.0..1.0).contains(&val_ratio) {
            return Err(HyperParamsError::InvalidValRatio(val_ratio));
        }
        if test_ratio <= 0.0 || test_ratio >= 1.0 {
            return Err(HyperParamsError::InvalidTestRatio(test_ratio));
        }
        if val_ratio + test_ratio >= 1.0 {
            return Err(HyperParamsError::SplitRatiosTooLarge {
                val_ratio,
                test_ratio,
            });
        }

        Ok(HyperParams {
            epochs,
            checkpoint_interval,
            batch_size,
            loss,
            optimizer,
            scheduler,
            clipping,
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

    pub fn loss(&self) -> &Arc<dyn LossFunction> {
        &self.loss
    }

    pub fn clipping(&self) -> &GradientClipping {
        &self.clipping
    }

    pub fn val_ratio(&self) -> f32 {
        self.val_ratio
    }

    pub fn test_ratio(&self) -> f32 {
        self.test_ratio
    }

    pub fn optimizer(&self) -> &dyn Optimizer {
        self.optimizer.as_ref()
    }

    pub fn optimizer_mut(&mut self) -> &mut Box<dyn Optimizer> {
        &mut self.optimizer
    }

    pub fn scheduler(&self) -> &dyn Scheduler {
        self.scheduler.as_ref()
    }

    pub fn scheduler_mut(&mut self) -> &mut Box<dyn Scheduler> {
        &mut self.scheduler
    }

    /// Borrows the optimizer and scheduler mutably at once, for a single
    /// [`crate::model::NeuralNetwork::train`] step.
    pub fn optimizer_and_scheduler_mut(&mut self) -> (&mut dyn Optimizer, &mut dyn Scheduler) {
        (self.optimizer.as_mut(), self.scheduler.as_mut())
    }

    /// Builds the runtime [`EarlyStopping`] tracker from this spec's config,
    /// seeding its best model from `init_model`. Returns `None` if early
    /// stopping is not configured.
    pub fn build_early_stopping(&self, init_model: &NeuralNetwork) -> Option<EarlyStopping> {
        self.early_stopping
            .clone()
            .map(|config| EarlyStopping::new(config, init_model))
    }
}
