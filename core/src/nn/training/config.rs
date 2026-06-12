use super::early_stopping::EarlyStoppingConfig;
use crate::gradients::GradientClipping;
use crate::loss_functions::LossFunction;
use crate::optimizers::Optimizer;
use crate::schedulers::Scheduler;
use std::fmt;
use std::sync::Arc;

/// The full specification of a training run, embedded in
/// [`crate::training::TrainingLoop`] and passed by reference to
/// [`crate::training::TrainingCallback::on_train_start`].
pub struct HyperParams {
    pub epochs: usize,
    /// Interval (in epochs) between scheduled evaluations/checkpoints; `0` disables them.
    pub checkpoint_interval: usize,
    pub batch_size: Option<usize>,
    pub loss: Arc<dyn LossFunction>,
    pub optimizer: Box<dyn Optimizer>,
    pub scheduler: Box<dyn Scheduler>,
    pub clipping: GradientClipping,
    pub early_stopping: Option<EarlyStoppingConfig>,
    /// Fraction of the dataset held out for validation. Part of the run's
    /// identity (the resulting split), not consumed by [`crate::training::TrainingLoop`].
    pub val_ratio: f32,
    /// Fraction of the dataset held out for testing. Part of the run's
    /// identity (the resulting split), not consumed by [`crate::training::TrainingLoop`].
    pub test_ratio: f32,
}

/// Returned by [`HyperParams::validate`] when a cross-field invariant is violated.
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
    /// Validates the cross-field invariants of this spec.
    pub fn validate(&self) -> Result<(), HyperParamsError> {
        if self.epochs < 1 {
            return Err(HyperParamsError::ZeroEpochs);
        }
        if self.batch_size == Some(0) {
            return Err(HyperParamsError::ZeroBatchSize);
        }
        if !(0.0..1.0).contains(&self.val_ratio) {
            return Err(HyperParamsError::InvalidValRatio(self.val_ratio));
        }
        if self.test_ratio <= 0.0 || self.test_ratio >= 1.0 {
            return Err(HyperParamsError::InvalidTestRatio(self.test_ratio));
        }
        if self.val_ratio + self.test_ratio >= 1.0 {
            return Err(HyperParamsError::SplitRatiosTooLarge {
                val_ratio: self.val_ratio,
                test_ratio: self.test_ratio,
            });
        }
        Ok(())
    }
}
