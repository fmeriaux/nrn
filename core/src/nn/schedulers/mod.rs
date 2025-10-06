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

use crate::training::LearningRate;

/// Trait for learning rate scheduling strategies.
///
/// Schedulers control how the learning rate evolves throughout the training process.
/// By adjusting the learning rate over time, schedulers can help improve convergence,
/// prevent overshooting, and enable fine-tuning in later epochs.
pub trait Scheduler {
    /// Adjusts the learning rate based on the current training state.
    ///
    /// This method is typically called at the end of each epoch or training iteration.
    /// Returns the new learning rate to be used for the next training step.
    ///
    /// # Returns
    ///
    /// The updated learning rate.
    fn step(&mut self) -> LearningRate;
}
