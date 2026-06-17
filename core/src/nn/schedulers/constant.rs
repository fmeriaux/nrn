use crate::learning_rate::LearningRate;
use crate::nn::schedulers::Scheduler;

/// A learning rate scheduler that always returns the same learning rate.
///
/// This scheduler does not modify the learning rate over time.
/// It is useful as a default or no-op scheduler when no dynamic scheduling is required.
///
/// # Example
///
/// ```
/// use nrn::schedulers::{Scheduler, ConstantScheduler};
/// use nrn::training::LearningRate;
///
/// let mut scheduler = ConstantScheduler::new(LearningRate::new(0.001).unwrap());
/// assert_eq!(scheduler.step().value(), 0.001);
/// assert_eq!(scheduler.step().value(), 0.001); // always the same learning rate
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
    fn name(&self) -> &'static str {
        "Constant"
    }

    fn step(&mut self) -> LearningRate {
        self.learning_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_is_constant() {
        let s = ConstantScheduler::new(0.003.try_into().unwrap());
        assert_eq!(s.name(), "Constant");
    }

    #[test]
    fn always_returns_the_same_rate() {
        let mut s = ConstantScheduler::new(0.003.try_into().unwrap());
        for _ in 0..5 {
            assert!((s.step().value() - 0.003).abs() < 1e-9);
        }
    }
}
