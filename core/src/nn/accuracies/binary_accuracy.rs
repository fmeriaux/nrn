//! Binary accuracy metric implementation.
//!
//! This module provides the `BinaryAccuracy` struct, which implements the `Accuracy` trait for binary classification tasks.
//! Binary accuracy measures the percentage of correct predictions for problems with two possible classes (0 or 1).
//!
//! The metric expects the network's output logits and ground truth labels as tensors of shape (1, n_samples).
//! A logit is thresholded at 0 (positive → class 1) to determine the predicted class. The accuracy is the ratio of correct predictions to the total number of samples.

use crate::accuracies::Accuracy;
use ndarray::{ArrayView2, Zip};
use once_cell::sync::Lazy;
use std::sync::Arc;

pub struct BinaryAccuracy;

impl Accuracy for BinaryAccuracy {
    /// Computes the binary accuracy for a batch of predictions and ground truth labels.
    ///
    /// # Arguments
    ///
    /// * `outputs` - A 2D tensor of shape (1, n_samples), containing the network's output for each sample.
    /// * `targets` - A 2D tensor of shape (1, n_samples), containing ground truth binary labels (0 or 1) for each sample.
    ///
    /// # Returns
    ///
    /// The accuracy as a percentage (between 0.0 and 100.0), representing the proportion of correct predictions.
    ///
    /// # Panics
    ///
    /// Panics if the shapes of `outputs` and `targets` do not match, or if the number of rows is not 1.
    fn compute(&self, outputs: ArrayView2<f32>, targets: ArrayView2<f32>) -> f32 {
        assert_eq!(
            outputs.shape(),
            targets.shape(),
            "Outputs and targets must have the same shape."
        );
        assert_eq!(
            outputs.nrows(),
            1,
            "Binary accuracy expects tensors of shape (1, n_samples). For one-hot or multi-class, use another accuracy."
        );

        let correct =
            Zip::from(&outputs)
                .and(&targets)
                .fold(0usize, |correct, &output, &target| {
                    let label = if output >= 0.0 { 1.0 } else { 0.0 };
                    correct + usize::from(label == target)
                });
        correct as f32 * 100.0 / (outputs.ncols() as f32)
    }
}

/// Shared static instance of `BinaryAccuracy` for convenient reuse.
pub static BINARY_ACCURACY: Lazy<Arc<BinaryAccuracy>> = Lazy::new(|| Arc::new(BinaryAccuracy));

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn threshold_at_zero_predicts_positive() {
        // output=0.0 → threshold to 1.0 (>= 0) → correct when target=1.0
        // output=-0.0001 → threshold to 0.0 → correct when target=0.0
        let outputs = array![[0.0_f32, -0.0001]];
        let targets = array![[1.0_f32, 0.0]];
        let acc = BINARY_ACCURACY.compute(outputs.view(), targets.view());
        assert!((acc - 100.0).abs() < 1e-5, "Expected 100%, got {acc}");
    }

    #[test]
    fn all_wrong_predictions_give_zero_accuracy() {
        let outputs = array![[3.0_f32, 3.0, -3.0]];
        let targets = array![[0.0_f32, 0.0, 1.0]];
        let acc = BINARY_ACCURACY.compute(outputs.view(), targets.view());
        assert!((acc - 0.0).abs() < 1e-5, "Expected 0%, got {acc}");
    }
}
