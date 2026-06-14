use crate::learning_rate::{LearningRate, LearningRateError};
use crate::schedulers::{Scheduler, SchedulerState};
use core::f32::consts::PI;
use std::fmt;

/// A cosine annealing learning rate scheduler.
///
/// This scheduler implements a cosine annealing schedule that smoothly decreases
/// the learning rate from a maximum value to a minimum value over a fixed number of steps.
/// The learning rate follows a cosine curve, providing a smooth decay.
///
/// # Formula
///
/// The learning rate at step `t` is computed as:
/// ```text
/// lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * t / T))
/// ```
/// where `T` is the total number of steps.
#[derive(Debug)]
pub struct CosineAnnealing {
    /// Minimum learning rate (reached at the end of the schedule).
    min: LearningRate,
    /// Maximum learning rate (starting point of the schedule).
    max: LearningRate,
    /// Current step in the schedule.
    current_step: usize,
    /// Total number of steps in one cycle.
    steps: usize,
    /// Enable warm restarts
    restarts: bool,
    /// Multiplier for increasing the period after each restart
    steps_multiplier: usize,
}

/// Returned by [`CosineAnnealing::new`] / [`CosineAnnealing::from_values`] /
/// [`CosineAnnealing::with_restarts`] when the given parameters are invalid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CosineAnnealingError {
    /// `max` was not strictly greater than `min`.
    MaxNotGreaterThanMin { min: f32, max: f32 },
    /// `steps` was zero.
    ZeroSteps,
    /// `steps_multiplier` (for warm restarts) was less than 1.
    ZeroStepsMultiplier,
    /// One of the learning rate bounds was invalid.
    LearningRate(LearningRateError),
}

impl fmt::Display for CosineAnnealingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CosineAnnealingError::MaxNotGreaterThanMin { min, max } => write!(
                f,
                "the maximum learning rate must be greater than the minimum, got min={min}, max={max}"
            ),
            CosineAnnealingError::ZeroSteps => {
                write!(f, "the step size must be greater than zero")
            }
            CosineAnnealingError::ZeroStepsMultiplier => {
                write!(f, "the cycle multiplier must be at least 1")
            }
            CosineAnnealingError::LearningRate(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for CosineAnnealingError {}

impl From<LearningRateError> for CosineAnnealingError {
    fn from(e: LearningRateError) -> Self {
        CosineAnnealingError::LearningRate(e)
    }
}

impl CosineAnnealing {
    /// Creates a new [`CosineAnnealing`] scheduler.
    ///
    /// # Errors
    /// Returns [`CosineAnnealingError`] if `max` is not greater than `min` or if `steps` is zero.
    pub fn new(
        min: LearningRate,
        max: LearningRate,
        steps: usize,
    ) -> Result<Self, CosineAnnealingError> {
        if max.value() <= min.value() {
            return Err(CosineAnnealingError::MaxNotGreaterThanMin {
                min: min.value(),
                max: max.value(),
            });
        }
        if steps == 0 {
            return Err(CosineAnnealingError::ZeroSteps);
        }

        Ok(Self {
            min,
            max,
            current_step: 0,
            steps,
            restarts: false,
            steps_multiplier: 1,
        })
    }

    /// Creates a new [`CosineAnnealing`] scheduler from raw learning rate bounds.
    /// # Errors
    /// Returns [`CosineAnnealingError`] if `min` or `max` are invalid, `max` is not
    /// greater than `min`, or `steps` is zero.
    pub fn from_values(min: f32, max: f32, steps: usize) -> Result<Self, CosineAnnealingError> {
        Self::new(LearningRate::new(min)?, LearningRate::new(max)?, steps)
    }

    /// Enables or disables warm restarts and sets the steps multiplier.
    /// When restarts are enabled, the scheduler will reset the current step to zero
    /// and multiply the total steps by `steps_multiplier` after each cycle.
    /// # Arguments
    /// * `restarts` - Whether to enable warm restarts.
    /// * `steps_multiplier` - The factor by which to multiply the total steps after each restart. Must be at least 1.
    /// # Errors
    /// Returns [`CosineAnnealingError::ZeroStepsMultiplier`] if `steps_multiplier` is less than 1.
    pub fn with_restarts(
        mut self,
        restarts: bool,
        steps_multiplier: usize,
    ) -> Result<Self, CosineAnnealingError> {
        if steps_multiplier < 1 {
            return Err(CosineAnnealingError::ZeroStepsMultiplier);
        }
        self.restarts = restarts;
        self.steps_multiplier = steps_multiplier;
        Ok(self)
    }

    /// Computes the current learning rate based on the cosine annealing formula.
    pub fn current_value(&self) -> LearningRate {
        let step = self.current_step.min(self.steps) as f32;
        let cos = (PI * step / (self.steps as f32)).cos();
        let lr = self.min.value() + 0.5 * (self.max.value() - self.min.value()) * (1.0 + cos);
        LearningRate::new(lr).expect("a value between two valid learning rates is always valid")
    }
}

impl Scheduler for CosineAnnealing {
    fn name(&self) -> &'static str {
        "Cosine Annealing"
    }

    fn step(&mut self) -> LearningRate {
        if !self.restarts && self.current_step >= self.steps {
            return self.min;
        }

        let lr = self.current_value();

        self.current_step += 1;

        if self.restarts && self.current_step >= self.steps {
            self.current_step = 0;
            self.steps *= self.steps_multiplier;
        }

        lr
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
    fn name_is_cosine_annealing() {
        let s = CosineAnnealing::from_values(0.001, 0.1, 10).unwrap();
        assert_eq!(s.name(), "Cosine Annealing");
    }

    #[test]
    fn starts_at_max_and_ends_at_min() {
        let (min, max, steps) = (0.001, 0.1, 10);
        let mut s = CosineAnnealing::from_values(min, max, steps).unwrap();
        // First step (cos(0) = 1) yields the maximum.
        assert!((s.step().value() - max).abs() < 1e-6);
        // Exhaust the remaining steps of the cycle.
        for _ in 1..steps {
            s.step();
        }
        // Past the end of the cycle (no restarts) the minimum is returned.
        assert!((s.step().value() - min).abs() < 1e-6);
    }

    #[test]
    fn midpoint_is_average_of_min_and_max() {
        let (min, max, steps) = (0.0, 1.0, 4);
        let mut s = CosineAnnealing::from_values(min, max, steps).unwrap();
        s.step(); // step 0
        s.step(); // step 1
        // step 2 of 4: cos(π/2) = 0 → lr = (min + max) / 2.
        assert!((s.step().value() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn warm_restart_resets_to_max_and_grows_period() {
        let (min, max, steps) = (0.0, 1.0, 2);
        let mut s = CosineAnnealing::from_values(min, max, steps)
            .unwrap()
            .with_restarts(true, 2)
            .unwrap();
        let first = s.step().value(); // step 0 → max
        s.step(); // step 1 → triggers restart, period grows to 4
        let after_restart = s.step().value(); // step 0 of new cycle → max again
        assert!((first - max).abs() < 1e-6);
        assert!((after_restart - max).abs() < 1e-6);
    }

    #[test]
    fn rejects_max_not_greater_than_min() {
        assert_eq!(
            CosineAnnealing::new(
                LearningRate::new(0.1).unwrap(),
                LearningRate::new(0.1).unwrap(),
                10
            )
            .unwrap_err(),
            CosineAnnealingError::MaxNotGreaterThanMin { min: 0.1, max: 0.1 }
        );
    }

    #[test]
    fn rejects_zero_steps() {
        assert_eq!(
            CosineAnnealing::new(
                LearningRate::new(0.0).unwrap(),
                LearningRate::new(0.1).unwrap(),
                0
            )
            .unwrap_err(),
            CosineAnnealingError::ZeroSteps
        );
    }

    #[test]
    fn rejects_zero_steps_multiplier() {
        let s = CosineAnnealing::new(
            LearningRate::new(0.0).unwrap(),
            LearningRate::new(0.1).unwrap(),
            10,
        )
        .unwrap();
        assert_eq!(
            s.with_restarts(true, 0).unwrap_err(),
            CosineAnnealingError::ZeroStepsMultiplier
        );
    }
}
