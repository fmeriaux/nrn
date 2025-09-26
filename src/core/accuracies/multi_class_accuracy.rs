//! Multi-class accuracy metric implementation.
//!
//! This module provides the `MultiClassAccuracy` struct, which implements the `Accuracy` trait for multi-class classification tasks.
//! Multi-class accuracy measures the percentage of correct predictions for problems with more than two possible classes.
//!
//! The metric expects predictions and ground truth labels as tensors of shape (n_classes, n_samples), where each column represents an example and each row a class.
//! For each sample, the predicted class is the index of the maximum value in the prediction vector (argmax), and the true class is the index of the maximum value in the label vector (one-hot encoding).
//! The accuracy is computed as the ratio of correct predictions to the total number of samples.

use crate::core::accuracies::Accuracy;
use ndarray::{ArrayView1, ArrayView2};
use std::sync::Arc;
use once_cell::sync::Lazy;

pub struct MultiClassAccuracy;

impl Accuracy for MultiClassAccuracy {
    /// Computes the multi-class accuracy for a batch of predictions and ground truth labels.
    ///
    /// # Arguments
    ///
    /// * `predictions` - A 2D tensor of shape (n_classes, n_samples), containing predicted scores or probabilities for each class and sample.
    /// * `targets` - A 2D tensor of shape (n_classes, n_samples), containing one-hot encoded ground truth labels for each sample.
    ///
    /// # Returns
    ///
    /// The accuracy as a percentage (between 0.0 and 100.0), representing the proportion of correct predictions.
    ///
    /// # Panics
    ///
    /// Panics if the shapes of `predictions` and `targets` do not match, or if there are fewer than 2 classes.
    fn compute(&self, predictions: ArrayView2<f32>, targets: ArrayView2<f32>) -> f32 {
        assert_eq!(
            predictions.shape(),
            targets.shape(),
            "Predictions and targets must have the same shape."
        );
        assert!(
            predictions.nrows() >= 2,
            "MultiClassAccuracy expects tensors with at least 2 classes (n_classes >= 2)."
        );
        let n_samples = predictions.ncols();
        let mut correct = 0;
        // Closure to find the index of the maximum value in a 1D array (argmax)
        let argmax = |col: ArrayView1<f32>| -> usize {
            col.iter()
                .enumerate()
                .max_by(|(_, left), (_, right)| left.partial_cmp(right).unwrap())
                .map(|(idx, _)| idx)
                .unwrap()
        };
        for i in 0..n_samples {
            let pred_col = predictions.column(i);
            let exp_col = targets.column(i);
            let pred_label = argmax(pred_col);
            let exp_label = argmax(exp_col);
            if pred_label == exp_label {
                correct += 1;
            }
        }
        (correct as f32) * 100.0 / (n_samples as f32)
    }
}

/// Shared static instance of `MultiClassAccuracy` for convenient reuse.
pub static MULTI_CLASS_ACCURACY: Lazy<Arc<MultiClassAccuracy>> = Lazy::new(|| Arc::new(MultiClassAccuracy));
