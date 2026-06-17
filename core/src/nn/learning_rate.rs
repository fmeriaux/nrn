use std::fmt;

/// Represents the learning rate used in optimization algorithms.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LearningRate(f32);

/// Returned by [`LearningRate::new`] when the given value is negative or non-finite.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LearningRateError(pub f32);

impl fmt::Display for LearningRateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "the learning rate must be a finite, non-negative value, got {}",
            self.0
        )
    }
}

impl std::error::Error for LearningRateError {}

impl LearningRate {
    /// Creates a new `LearningRate` instance with the specified value.
    /// # Errors
    /// Returns [`LearningRateError`] when `value` is negative or non-finite.
    pub fn new(value: f32) -> Result<Self, LearningRateError> {
        if value.is_finite() && value >= 0.0 {
            Ok(LearningRate(value))
        } else {
            Err(LearningRateError(value))
        }
    }

    /// Returns the current learning rate value.
    pub fn value(&self) -> f32 {
        self.0
    }
}

impl TryFrom<f32> for LearningRate {
    type Error = LearningRateError;

    /// Validates a raw value into a [`LearningRate`], mirroring [`LearningRate::new`].
    fn try_from(value: f32) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}
