use std::fmt;

/// Represents the weight-decay coefficient applied by an optimizer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WeightDecay(f32);

/// Returned by [`WeightDecay::new`] when the given value is negative or non-finite.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WeightDecayError(pub f32);

impl fmt::Display for WeightDecayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "the weight decay must be a finite, non-negative value, got {}",
            self.0
        )
    }
}

impl std::error::Error for WeightDecayError {}

impl WeightDecay {
    /// No weight decay.
    pub const ZERO: Self = WeightDecay(0.0);

    /// Creates a new `WeightDecay` instance with the specified value.
    /// # Errors
    /// Returns [`WeightDecayError`] when `value` is negative or non-finite.
    pub fn new(value: f32) -> Result<Self, WeightDecayError> {
        if value.is_finite() && value >= 0.0 {
            Ok(WeightDecay(value))
        } else {
            Err(WeightDecayError(value))
        }
    }

    /// Returns the current weight-decay value.
    pub fn value(&self) -> f32 {
        self.0
    }

    /// Whether weight decay is active (a non-zero coefficient).
    pub fn is_active(&self) -> bool {
        self.0 > 0.0
    }
}

impl TryFrom<f32> for WeightDecay {
    type Error = WeightDecayError;

    /// Validates a raw value into a [`WeightDecay`], mirroring [`WeightDecay::new`].
    fn try_from(value: f32) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_is_inactive() {
        assert_eq!(WeightDecay::ZERO.value(), 0.0);
        assert!(!WeightDecay::ZERO.is_active());
    }

    #[test]
    fn positive_value_is_active() {
        let wd = WeightDecay::new(0.01).unwrap();
        assert_eq!(wd.value(), 0.01);
        assert!(wd.is_active());
    }

    #[test]
    fn rejects_negative_and_non_finite() {
        assert_eq!(
            WeightDecay::new(-0.1).unwrap_err().to_string(),
            "the weight decay must be a finite, non-negative value, got -0.1"
        );
        assert!(WeightDecay::new(f32::NAN).is_err());
        assert!(WeightDecay::new(f32::INFINITY).is_err());
    }
}
