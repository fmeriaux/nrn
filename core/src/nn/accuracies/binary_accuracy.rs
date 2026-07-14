//! Binary accuracy metric, for a single-logit or multi-label output.
//!
//! Predictions and targets share a shape whose leading axis (axis 0) is one logit row per label
//! (a single row for binary, one row per label for multi-label); every trailing axis — the
//! samples, plus any spatial axes — is a position scored independently. Each logit is
//! thresholded at 0 (>= 0 → class 1) and matched against its `0.0`/`1.0` target. The metric is
//! the ratio of correctly classified positions to the total — a Hamming accuracy when there are
//! multiple labels.

use crate::accuracies::Accuracy;
use ndarray::{ArrayViewD, Zip};
use once_cell::sync::Lazy;
use std::sync::Arc;

pub struct BinaryAccuracy;

impl Accuracy for BinaryAccuracy {
    /// Computes the binary (or multi-label Hamming) accuracy over every position of a batch.
    ///
    /// # Arguments
    /// - `outputs`: the network's logit scores, one row per label; every trailing axis — the
    ///   samples, plus any spatial axes — is a position scored independently.
    /// - `targets`: the ground-truth labels (`0.0`/`1.0`), the same shape as `outputs`.
    ///
    /// # Returns
    /// The accuracy as a percentage (0.0 to 100.0): the fraction of positions whose thresholded
    /// logit matches the target.
    ///
    /// # Panics
    /// When `outputs` and `targets` do not have the same shape.
    fn compute(&self, outputs: ArrayViewD<f32>, targets: ArrayViewD<f32>) -> f32 {
        assert_eq!(
            outputs.shape(),
            targets.shape(),
            "Outputs and targets must have the same shape."
        );

        let correct =
            Zip::from(&outputs)
                .and(&targets)
                .fold(0usize, |correct, &output, &target| {
                    let label = if output >= 0.0 { 1.0 } else { 0.0 };
                    correct + usize::from(label == target)
                });

        correct as f32 * 100.0 / (outputs.len() as f32)
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
        let outputs = array![[0.0_f32, -0.0001]].into_dyn();
        let targets = array![[1.0_f32, 0.0]].into_dyn();
        let acc = BINARY_ACCURACY.compute(outputs.view(), targets.view());
        assert!((acc - 100.0).abs() < 1e-5, "Expected 100%, got {acc}");
    }

    #[test]
    fn all_wrong_predictions_give_zero_accuracy() {
        let outputs = array![[3.0_f32, 3.0, -3.0]].into_dyn();
        let targets = array![[0.0_f32, 0.0, 1.0]].into_dyn();
        let acc = BINARY_ACCURACY.compute(outputs.view(), targets.view());
        assert!((acc - 0.0).abs() < 1e-5, "Expected 0%, got {acc}");
    }

    #[test]
    fn scores_every_spatial_position() {
        // (class=1, height=2, samples=2): 4 positions, each thresholded at 0. Two match.
        let outputs = array![[[0.5_f32, -0.5], [-1.0, 2.0]]].into_dyn();
        let targets = array![[[1.0_f32, 0.0], [1.0, 0.0]]].into_dyn();
        let acc = BINARY_ACCURACY.compute(outputs.view(), targets.view());
        assert!((acc - 50.0).abs() < 1e-5, "Expected 50%, got {acc}");
    }

    #[test]
    fn scores_multi_label_rows_as_hamming_accuracy() {
        // (n_labels=3, samples=2): 6 label/sample positions, each thresholded independently.
        // Four match: (label0, s0), (label0, s1), (label1, s0), (label2, s1).
        let outputs = array![[0.1_f32, -0.1], [1.0, 1.0], [-1.0, -1.0]].into_dyn();
        let targets = array![[1.0_f32, 0.0], [1.0, 0.0], [1.0, 0.0]].into_dyn();
        let acc = BINARY_ACCURACY.compute(outputs.view(), targets.view());
        assert!(
            (acc - (4.0 / 6.0 * 100.0)).abs() < 1e-5,
            "Expected 66.67%, got {acc}"
        );
    }
}
