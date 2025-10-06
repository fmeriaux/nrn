use crate::schedulers::Scheduler;
use crate::training::LearningRate;
use core::f32::consts::PI;

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
///
/// ```
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

impl CosineAnnealing {
    /// Creates a new [`CosineAnnealing`] scheduler.
    ///
    /// # Panics
    /// Will panic if `max` is not greater than `min` or if `steps` is zero.
    ///
    pub fn new(min: LearningRate, max: LearningRate, steps: usize) -> Self {
        assert!(
            max.value() > min.value(),
            "Maximum learning rate must be greater than minimum"
        );
        assert!(steps > 0, "Total steps must be greater than zero");

        Self {
            min,
            max,
            current_step: 0,
            steps,
            restarts: false,
            steps_multiplier: 1,
        }
    }

    /// Enables or disables warm restarts and sets the steps multiplier.
    /// When restarts are enabled, the scheduler will reset the current step to zero
    /// and multiply the total steps by `steps_multiplier` after each cycle.
    /// # Arguments
    /// * `restarts` - Whether to enable warm restarts.
    /// * `steps_multiplier` - The factor by which to multiply the total steps after each restart. Must be greater than 1.
    /// # Panics
    /// Will panic if `steps_multiplier` is less than 1.
    pub fn with_restarts(mut self, restarts: bool, steps_multiplier: usize) -> Self {
        assert!(steps_multiplier >= 1, "Steps multiplier must be at least 1");
        self.restarts = restarts;
        self.steps_multiplier = steps_multiplier;
        self
    }

    /// Computes the current learning rate based on the cosine annealing formula.
    pub fn current_value(&self) -> LearningRate {
        let step = self.current_step.min(self.steps) as f32;
        let cos = (PI * step / (self.steps as f32)).cos();
        let lr = self.min.value() + 0.5 * (self.max.value() - self.min.value()) * (1.0 + cos);
        LearningRate::new(lr)
    }
}

impl Scheduler for CosineAnnealing {
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
}
