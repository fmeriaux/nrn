use crate::nn::schedulers::Scheduler;
use crate::training::LearningRate;

/// A learning rate scheduler that always returns the same learning rate.
///
/// This scheduler does not modify the learning rate over time.
/// It is useful as a default or no-op scheduler when no dynamic scheduling is required.
///
/// # Example
///
/// ```
/// use your_crate::schedulers::{Scheduler, ConstantScheduler};
///
/// let mut scheduler = ConstantScheduler::new(0.001);
/// assert_eq!(scheduler.step(), 0.001);
/// assert_eq!(scheduler.step(), 0.001); // always the same
/// ```
pub struct ConstantScheduler {
    learning_rate: LearningRate,
}

impl ConstantScheduler {
    /// Creates a new `ConstantScheduler` with the given learning rate.
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - The fixed learning rate to use. Must be greater than zero.
    ///
    pub fn new(learning_rate: LearningRate) -> Self {
        Self { learning_rate }
    }
}

impl Scheduler for ConstantScheduler {
    fn step(&mut self) -> LearningRate {
        self.learning_rate
    }
}
