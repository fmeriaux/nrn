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
            let pred_label = if pred >= 0.5 { 1.0 } else { 0.0 };
            if (pred_label - exp).abs() < f32::EPSILON {
                correct += 1;
            }
        }
        (correct as f32) * 100.0 / (n_samples as f32)
    }
}

/// Shared static instance of `BinaryAccuracy` for convenient reuse.
pub static BINARY_ACCURACY: Lazy<Arc<BinaryAccuracy>> = Lazy::new(|| Arc::new(BinaryAccuracy));
