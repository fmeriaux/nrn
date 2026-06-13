use crate::learning_rate::{LearningRate, LearningRateError};
use crate::schedulers::{Scheduler, SchedulerState};
use std::fmt;

#[derive(Debug)]
pub struct StepDecay {
    initial: LearningRate,
    steps: usize,
    current_step: usize,
    decay_factor: f32,
}

/// Returned by [`StepDecay::new`] / [`StepDecay::from_values`] when `steps` is
/// zero, `decay_factor` is not in (0, 1), or the initial learning rate is invalid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StepDecayError {
    /// `steps` was zero.
    ZeroSteps,
    /// `decay_factor` was not strictly between 0 and 1.
    InvalidDecayFactor(f32),
    /// The initial learning rate was invalid.
    LearningRate(LearningRateError),
}

impl fmt::Display for StepDecayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StepDecayError::ZeroSteps => write!(f, "the step size must be greater than zero"),
            StepDecayError::InvalidDecayFactor(decay_factor) => {
                write!(f, "the decay factor must be in (0, 1), got {decay_factor}")
            }
            StepDecayError::LearningRate(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for StepDecayError {}

impl From<LearningRateError> for StepDecayError {
    fn from(e: LearningRateError) -> Self {
        StepDecayError::LearningRate(e)
    }
}

impl StepDecay {
    /// Creates a new [`StepDecay`] scheduler.
    ///
    /// # Errors
    /// Returns [`StepDecayError`] if `steps` is zero or if `decay_factor` is not in (0, 1).
    pub fn new(
        initial: LearningRate,
        steps: usize,
        decay_factor: f32,
    ) -> Result<Self, StepDecayError> {
        if steps == 0 {
            return Err(StepDecayError::ZeroSteps);
        }
        if !(decay_factor > 0.0 && decay_factor < 1.0) {
            return Err(StepDecayError::InvalidDecayFactor(decay_factor));
        }

        Ok(Self {
            initial,
            steps,
            current_step: 0,
            decay_factor,
        })
    }

    /// Creates a new [`StepDecay`] scheduler from a raw initial learning rate value.
    /// # Errors
    /// Returns [`StepDecayError`] if `initial` is invalid, `steps` is zero, or
    /// `decay_factor` is not in (0, 1).
    pub fn from_values(
        initial: f32,
        steps: usize,
        decay_factor: f32,
    ) -> Result<Self, StepDecayError> {
        Self::new(LearningRate::new(initial)?, steps, decay_factor)
    }
}

impl Scheduler for StepDecay {
    fn name(&self) -> &'static str {
        "Step Decay"
    }

    /// Returns the learning rate for the current step and advances the counter.
    ///
    /// The decay is applied in discrete plateaus: `lr = initial * decay_factor^floor(t / steps)`.
    /// The learning rate stays constant for `steps` steps, then drops by `decay_factor`, and so on.
    /// For a continuous exponential decay, use `CosineAnnealing` instead.
    fn step(&mut self) -> LearningRate {
        let learning_rate = LearningRate::new(
            self.initial.value()
                * self
                    .decay_factor
                    .powi((self.current_step / self.steps) as i32),
        )
        .expect("a positive decay factor applied to a valid learning rate stays valid");
        self.current_step += 1;
        learning_rate
    }

    fn to_state(&self) -> Option<SchedulerState> {
        Some(SchedulerState {
            current_step: self.current_step,
        })
    }

    fn restore(&mut self, state: &SchedulerState) {
        self.current_step = state.current_step;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_is_step_decay() {
        let sched = StepDecay::new(LearningRate::new(0.1).unwrap(), 3, 0.5).unwrap();
        assert_eq!(sched.name(), "Step Decay");
    }

    #[test]
    fn lr_stays_constant_within_each_period() {
        // initial=0.1, steps=3, decay=0.5
        // Steps 0-2: floor(t/3)=0 → LR = 0.1 * 0.5^0 = 0.1
        // Steps 3-5: floor(t/3)=1 → LR = 0.1 * 0.5^1 = 0.05
        let mut sched = StepDecay::new(LearningRate::new(0.1).unwrap(), 3, 0.5).unwrap();
        for _ in 0..3 {
            assert!((sched.step().value() - 0.1).abs() < 1e-6);
        }
        for _ in 0..3 {
            assert!((sched.step().value() - 0.05).abs() < 1e-6);
        }
    }

    #[test]
    fn decay_never_reaches_zero() {
        let mut sched = StepDecay::new(LearningRate::new(0.1).unwrap(), 1, 0.5).unwrap();
        for _ in 0..100 {
            assert!(sched.step().value() > 0.0);
        }
    }

    #[test]
    fn rejects_zero_steps() {
        assert_eq!(
            StepDecay::new(LearningRate::new(0.1).unwrap(), 0, 0.5).unwrap_err(),
            StepDecayError::ZeroSteps
        );
    }

    #[test]
    fn rejects_decay_factor_out_of_range() {
        assert_eq!(
            StepDecay::new(LearningRate::new(0.1).unwrap(), 3, 1.0).unwrap_err(),
            StepDecayError::InvalidDecayFactor(1.0)
        );
        assert_eq!(
            StepDecay::new(LearningRate::new(0.1).unwrap(), 3, 0.0).unwrap_err(),
            StepDecayError::InvalidDecayFactor(0.0)
        );
    }
}
