use crate::schedulers::Scheduler;
use crate::training::LearningRate;
use std::f32::consts::PI;

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
    lr_min: f32,
    /// Maximum learning rate (starting point of the schedule).
    lr_max: f32,
    /// Current step in the schedule.
    current_step: usize,
    /// Total number of steps for the schedule.
    total_steps: usize,
}

impl CosineAnnealing {
    /// Creates a new [`CosineAnnealing`] scheduler.
    ///
    /// # Panics
    /// Will panic if `lr_min` is negative, `lr_max` is not greater than `lr_min`,
    ///
    pub fn new(lr_min: f32, lr_max: f32, total_steps: usize) -> Self {
        assert!(lr_min >= 0.0, "Minimum learning rate must be non-negative");
        assert!(
            lr_max > lr_min,
            "Maximum learning rate must be greater than minimum"
        );
        assert!(total_steps > 0, "Total steps must be greater than zero");

        Self {
            lr_min,
            lr_max,
            current_step: 0,
            total_steps,
        }
    }
}

impl Scheduler for CosineAnnealing {
    fn step(&mut self) -> LearningRate {
        if self.current_step >= self.total_steps {
            return LearningRate::new(self.lr_min);
        }

        let cos = (PI * (self.current_step as f32) / (self.total_steps as f32)).cos();
        let lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1.0 + cos);

        self.current_step += 1;
        LearningRate::new(lr)
    }
}
