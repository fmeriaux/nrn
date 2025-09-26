mod binary_accuracy;
mod multi_class_accuracy;

pub use binary_accuracy::BINARY_ACCURACY;
pub use multi_class_accuracy::MULTI_CLASS_ACCURACY;

use ndarray::ArrayView2;

/// Computes the accuracy metric between predicted and expected outputs.
///
/// # Description
/// The accuracy metric measures the percentage of correctly predicted samples.
/// Implementors of this trait provide their own accuracy calculation logic, suitable
/// for different classification tasks (binary, multi-class, etc.).
///
/// # Parameters
/// - `predictions`: 2D array containing model predictions per sample.
/// - `targets`: 2D array containing ground truth labels per sample.
///
/// # Returns
/// A floating-point number representing accuracy as a percentage (0.0 to 100.0).
///
/// # Usage
/// This trait allows flexibility in defining accuracy computations tailored to
/// various problem settings while maintaining a unified interface.
pub trait Accuracy: Send + Sync {
    /// Computes the accuracy given the predicted and true values.
    ///
    /// # Arguments
    ///
    /// * `predictions` - A 2D array where each row represents a predicted output for a sample.
    /// * `targets` - A 2D array where each row represents the expected (true) output for a sample.
    ///
    /// # Returns
    /// A float representing the accuracy percentage (0.0 to 100.0).
    fn compute(&self, predictions: ArrayView2<f32>, targets: ArrayView2<f32>) -> f32;
}
