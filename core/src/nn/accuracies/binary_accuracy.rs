//! Binary accuracy metric implementation.
//!
//! This module provides the `BinaryAccuracy` struct, which implements the `Accuracy` trait for binary classification tasks.
//! Binary accuracy measures the percentage of correct predictions for problems with two possible classes (0 or 1).
//!
//! The metric expects predictions and ground truth labels as tensors of shape (1, n_samples), where each value is a probability or binary label.
//! Predictions are thresholded at 0.5 to determine the predicted class. The accuracy is computed as the ratio of correct predictions to the total number of samples.

use crate::accuracies::Accuracy;
use ndarray::ArrayView2;
use std::sync::Arc;
use once_cell::sync::Lazy;

pub struct BinaryAccuracy;

impl Accuracy for BinaryAccuracy {
    /// Computes the binary accuracy for a batch of predictions and ground truth labels.
    ///
    /// # Arguments
    ///
    /// * `predictions` - A 2D tensor of shape (1, n_samples), containing predicted probabilities or binary values for each sample.
    /// * `targets` - A 2D tensor of shape (1, n_samples), containing ground truth binary labels (0 or 1) for each sample.
    ///
    /// # Returns
    ///
    /// The accuracy as a percentage (between 0.0 and 100.0), representing the proportion of correct predictions.
    ///
    /// # Panics
    ///
    /// Panics if the shapes of `predictions` and `targets` do not match, or if the number of rows is not 1.
    fn compute(&self, predictions: ArrayView2<f32>, targets: ArrayView2<f32>) -> f32 {
        assert_eq!(
            predictions.shape(),
            targets.shape(),
            "Predictions and targets must have the same shape."
        );
        assert_eq!(
            predictions.nrows(),
            1,
            "Binary accuracy expects tensors of shape (1, n_samples). For one-hot or multi-class, use another accuracy."
        );

        let n_samples = predictions.ncols();
        let mut correct = 0;
        for i in 0..n_samples {
            let pred = predictions[[0, i]];
            let exp = targets[[0, i]];
            let pred_label = if pred >= 0.5 { 1.0_f32 } else { 0.0_f32 };
            if pred_label == exp {
                correct += 1;
            }
        }
        (correct as f32) * 100.0 / (n_samples as f32)
    }
}

/// Shared static instance of `BinaryAccuracy` for convenient reuse.
pub static BINARY_ACCURACY: Lazy<Arc<BinaryAccuracy>> = Lazy::new(|| Arc::new(BinaryAccuracy));

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn threshold_at_05_predicts_positive() {
        // pred=0.5 → threshold to 1.0 (>= 0.5) → correct when target=1.0
        // pred=0.4999 → threshold to 0.0 → correct when target=0.0
        let predictions = array![[0.5_f32, 0.4999]];
        let targets = array![[1.0_f32, 0.0]];
        let acc = BINARY_ACCURACY.compute(predictions.view(), targets.view());
        assert!((acc - 100.0).abs() < 1e-5, "Expected 100%, got {acc}");
    }

    #[test]
    fn all_wrong_predictions_give_zero_accuracy() {
        let predictions = array![[0.9_f32, 0.9, 0.1]];
        let targets = array![[0.0_f32, 0.0, 1.0]];
        let acc = BINARY_ACCURACY.compute(predictions.view(), targets.view());
        assert!((acc - 0.0).abs() < 1e-5, "Expected 0%, got {acc}");
    }
}
