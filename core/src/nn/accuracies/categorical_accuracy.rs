//! Categorical accuracy metric, for a multi-class (softmax) output.
//!
//! Predictions and targets share a shape whose leading axis (axis 0) holds the per-class scores;
//! every trailing axis — the samples, plus any spatial axes — is a position scored independently.
//! Each position's class lane is a logit / one-hot vector whose argmax is the predicted class,
//! matched against the argmax of the target lane. The metric is the ratio of correctly classified
//! positions to the total.

use crate::accuracies::Accuracy;
use ndarray::{ArrayView1, ArrayViewD, Axis};
use once_cell::sync::Lazy;
use std::sync::Arc;

pub struct CategoricalAccuracy;

impl Accuracy for CategoricalAccuracy {
    /// Computes the categorical accuracy over every position of a batch.
    ///
    /// # Arguments
    /// - `outputs`: the network's scores with the class axis first (axis 0); every trailing axis —
    ///   the samples, plus any spatial axes — is a position scored independently.
    /// - `targets`: the ground-truth labels, the same shape as `outputs`.
    ///
    /// # Returns
    /// The accuracy as a percentage (0.0 to 100.0): the fraction of positions whose predicted
    /// class matches the target.
    ///
    /// # Panics
    /// When `outputs` and `targets` do not have the same shape, or when there are fewer than
    /// two classes on the leading axis.
    fn compute(&self, outputs: ArrayViewD<f32>, targets: ArrayViewD<f32>) -> f32 {
        assert_eq!(
            outputs.shape(),
            targets.shape(),
            "Outputs and targets must have the same shape."
        );
        assert!(
            outputs.shape()[0] >= 2,
            "Categorical accuracy expects at least two classes on the leading axis."
        );

        // Every axis but the class axis is a position: samples, plus any spatial axes.
        let positions = outputs.len() / outputs.shape()[0];

        // The class axis is a lane per position; the prediction is its argmax, matched against
        // the argmax of the target lane.
        let argmax = |lane: ArrayView1<f32>| -> usize {
            lane.iter()
                .enumerate()
                .max_by(|(_, left), (_, right)| left.total_cmp(right))
                .map(|(idx, _)| idx)
                .unwrap()
        };
        let correct = outputs
            .lanes(Axis(0))
            .into_iter()
            .zip(targets.lanes(Axis(0)))
            .filter(|(prediction, target)| argmax(*prediction) == argmax(*target))
            .count();

        correct as f32 * 100.0 / (positions as f32)
    }
}

/// Shared static instance of `CategoricalAccuracy` for convenient reuse.
pub static CATEGORICAL_ACCURACY: Lazy<Arc<CategoricalAccuracy>> =
    Lazy::new(|| Arc::new(CategoricalAccuracy));

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn all_correct_gives_100_percent() {
        let outputs = array![[0.9_f32, 0.1], [0.1, 0.9]].into_dyn();
        let targets = array![[1.0_f32, 0.0], [0.0, 1.0]].into_dyn();
        let acc = CATEGORICAL_ACCURACY.compute(outputs.view(), targets.view());
        assert!((acc - 100.0).abs() < 1e-5, "Expected 100%, got {acc}");
    }

    #[test]
    fn does_not_panic_on_nan_predictions() {
        let outputs = array![[f32::NAN, 1.0], [0.5_f32, 0.8]].into_dyn();
        let targets = array![[0.0_f32, 1.0], [1.0, 0.0]].into_dyn();
        let _ = CATEGORICAL_ACCURACY.compute(outputs.view(), targets.view());
    }

    #[test]
    fn scores_every_spatial_position_by_argmax_over_the_class_axis() {
        // (n_classes=2, height=2, samples=2): argmax over axis 0 at each of the 4 positions.
        // Three of the four positions match their one-hot target → 75%.
        let outputs = array![[[2.0_f32, 1.0], [2.0, 1.0]], [[1.0, 2.0], [1.0, 2.0]]].into_dyn();
        let targets = array![[[1.0_f32, 0.0], [0.0, 0.0]], [[0.0, 1.0], [1.0, 1.0]]].into_dyn();
        let acc = CATEGORICAL_ACCURACY.compute(outputs.view(), targets.view());
        assert!((acc - 75.0).abs() < 1e-5, "Expected 75%, got {acc}");
    }

    #[test]
    #[should_panic(expected = "at least two classes")]
    fn panics_on_a_single_class_row() {
        let outputs = array![[0.5_f32, -0.5]].into_dyn();
        let targets = array![[1.0_f32, 0.0]].into_dyn();
        let _ = CATEGORICAL_ACCURACY.compute(outputs.view(), targets.view());
    }
}
