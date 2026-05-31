mod binary_accuracy;
mod multi_class_accuracy;

pub use binary_accuracy::BINARY_ACCURACY;
pub use multi_class_accuracy::MULTI_CLASS_ACCURACY;
use std::sync::Arc;

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

/// Returns an appropriate accuracy metric based on the number of classes.
/// # Arguments
/// * `n_classes` - The number of classes in the classification task.
/// # Panics
/// Will panic if `n_classes` is less than 2.
/// # Returns
/// An `Arc` to a struct implementing the `Accuracy` trait.
pub fn accuracy_for(n_classes: usize) -> Arc<dyn Accuracy> {
    assert!(
        n_classes > 1,
        "Number of classes must be greater than 1, got {}",
        n_classes
    );
    match n_classes {
        2 => BINARY_ACCURACY.clone(),
        _ => MULTI_CLASS_ACCURACY.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn two_classes_selects_binary_accuracy() {
        // One output row → binary metric. Perfect predictions score 100%.
        let accuracy = accuracy_for(2);
        let predictions = array![[0.9, 0.1]];
        let targets = array![[1.0, 0.0]];
        assert_eq!(accuracy.compute(predictions.view(), targets.view()), 100.0);
    }

    #[test]
    fn more_than_two_classes_selects_multi_class_accuracy() {
        // Three output rows → multi-class (argmax) metric. Perfect predictions score 100%.
        let accuracy = accuracy_for(3);
        let predictions = array![[0.8, 0.1], [0.1, 0.8], [0.1, 0.1]];
        let targets = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]];
        assert_eq!(accuracy.compute(predictions.view(), targets.view()), 100.0);
    }

    #[test]
    #[should_panic(expected = "Number of classes must be greater than 1")]
    fn fewer_than_two_classes_panics() {
        accuracy_for(1);
    }
}
