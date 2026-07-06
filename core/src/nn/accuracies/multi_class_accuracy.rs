//! Multi-class accuracy metric implementation.
//!
//! This module provides the `MultiClassAccuracy` struct, which implements the `Accuracy` trait for multi-class classification tasks.
//! Multi-class accuracy measures the percentage of correct predictions for problems with more than two possible classes.
//!
//! The metric expects predictions and ground truth labels as tensors of shape (n_classes, n_samples), where each column represents an example and each row a class.
//! For each sample, the predicted class is the index of the maximum value in the prediction vector (argmax), and the true class is the index of the maximum value in the label vector (one-hot encoding).
//! The accuracy is computed as the ratio of correct predictions to the total number of samples.

use crate::accuracies::Accuracy;
use ndarray::{ArrayView1, ArrayView2};
use once_cell::sync::Lazy;
use std::sync::Arc;

pub struct MultiClassAccuracy;

impl Accuracy for MultiClassAccuracy {
    /// Computes the multi-class accuracy for a batch of predictions and ground truth labels.
    ///
    /// # Arguments
    ///
    /// * `outputs` - A 2D tensor of shape (n_classes, n_samples), containing the network's output for each class and sample.
    /// * `targets` - A 2D tensor of shape (n_classes, n_samples), containing one-hot encoded ground truth labels for each sample.
    ///
    /// # Returns
    ///
    /// The accuracy as a percentage (between 0.0 and 100.0), representing the proportion of correct predictions.
    ///
    /// # Panics
    ///
    /// Panics if the shapes of `outputs` and `targets` do not match, or if there are fewer than 2 classes.
    fn compute(&self, outputs: ArrayView2<f32>, targets: ArrayView2<f32>) -> f32 {
        assert_eq!(
            outputs.shape(),
            targets.shape(),
            "Outputs and targets must have the same shape."
        );
        assert!(
            outputs.nrows() >= 2,
            "MultiClassAccuracy expects tensors with at least 2 classes (n_classes >= 2)."
        );
        // Index of the maximum value in a 1D array (argmax).
        let argmax = |col: ArrayView1<f32>| -> usize {
            col.iter()
                .enumerate()
                .max_by(|(_, left), (_, right)| left.total_cmp(right))
                .map(|(idx, _)| idx)
                .unwrap()
        };
        let correct = outputs
            .columns()
            .into_iter()
            .zip(targets.columns())
            .filter(|(prediction, target)| argmax(*prediction) == argmax(*target))
            .count();
        correct as f32 * 100.0 / (outputs.ncols() as f32)
    }
}

/// Shared static instance of `MultiClassAccuracy` for convenient reuse.
pub static MULTI_CLASS_ACCURACY: Lazy<Arc<MultiClassAccuracy>> =
    Lazy::new(|| Arc::new(MultiClassAccuracy));

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn does_not_panic_on_nan_predictions() {
        let outputs = array![[f32::NAN, 1.0], [0.5_f32, 0.8]];
        let targets = array![[0.0_f32, 1.0], [1.0, 0.0]];
        let _ = MULTI_CLASS_ACCURACY.compute(outputs.view(), targets.view());
    }

    #[test]
    fn all_correct_gives_100_percent() {
        let outputs = array![[0.9_f32, 0.1], [0.1, 0.9]];
        let targets = array![[1.0_f32, 0.0], [0.0, 1.0]];
        let acc = MULTI_CLASS_ACCURACY.compute(outputs.view(), targets.view());
        assert!((acc - 100.0).abs() < 1e-5, "Expected 100%, got {acc}");
    }
}
