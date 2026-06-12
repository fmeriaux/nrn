/// Represents the learning rate used in optimization algorithms.
#[derive(Clone, Copy, Debug)]
pub struct LearningRate(f32);

impl LearningRate {
    /// Creates a new `LearningRate` instance with the specified value.
    /// # Panics
    /// - When the provided `value` is negative.
    /// # Arguments
    /// - `value`: The learning rate value to be used in optimization algorithms.
    pub fn new(value: f32) -> Self {
        assert!(value >= 0.0, "Learning rate must be non-negative.");
        LearningRate(value)
    }

    /// Returns the current learning rate value.
    pub fn value(&self) -> f32 {
        self.0
    }
}
