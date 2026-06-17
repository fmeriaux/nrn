//! Learning rate scheduling for neural network training.
//!
//! This module provides the `Scheduler` trait and various implementations for dynamically adjusting the learning rate during training.
//! Learning rate schedules are essential for optimizing model convergence, helping to escape local minima and fine-tune model parameters.
//! Different scheduling strategies can significantly impact training speed and final model performance.

mod constant;
mod cosine_annealing;
mod step;

pub use constant::*;
pub use cosine_annealing::*;
pub use step::*;

use crate::learning_rate::LearningRate;

/// Scheduler-agnostic snapshot of internal state (the current step count),
/// for checkpointing and resuming. Serialization lives behind the `io`
/// feature; this type stays free of serde/safetensors.
pub struct SchedulerState {
    pub current_step: usize,
}

/// Trait for learning rate scheduling strategies.
///
/// Schedulers control how the learning rate evolves throughout the training process.
/// By adjusting the learning rate over time, schedulers can help improve convergence,
/// prevent overshooting, and enable fine-tuning in later epochs.
pub trait Scheduler {
    /// Returns a human-readable name for this scheduler.
    fn name(&self) -> &'static str;

    /// Adjusts the learning rate based on the current training state.
    ///
    /// This method is typically called at the end of each epoch or training iteration.
    /// Returns the new learning rate to be used for the next training step.
    ///
    /// # Returns
    ///
    /// The updated learning rate.
    fn step(&mut self) -> LearningRate;

    /// Returns a snapshot of this scheduler's internal state for checkpointing,
    /// or `None` for schedulers with no state to resume (e.g. constant).
    fn to_state(&self) -> Option<SchedulerState> {
        None
    }

    /// Restores internal state previously returned by [`to_state`](Scheduler::to_state).
    /// The default implementation ignores `state` (stateless schedulers).
    fn restore(&mut self, _state: &SchedulerState) {}
}
