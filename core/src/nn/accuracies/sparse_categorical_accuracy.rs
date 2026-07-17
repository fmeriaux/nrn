//! Sparse categorical accuracy metric, for a multi-class (softmax) output scored against
//! direct class ids rather than one-hot targets.
//!
//! Predictions carry the per-class scores on their leading axis (axis 0); targets carry a
//! single class-id row on that same axis. Every trailing axis — the samples, plus any spatial
//! axes — is a position scored independently: the predicted class is the argmax of the
//! prediction lane, matched against the target lane's id. The metric is the ratio of correctly
//! classified positions to the total.

use crate::accuracies::Accuracy;
use ndarray::{ArrayView1, ArrayViewD, Axis};
use once_cell::sync::Lazy;
use std::sync::Arc;

pub struct SparseCategoricalAccuracy;

impl Accuracy for SparseCategoricalAccuracy {
    /// Computes the sparse categorical accuracy over every position of a batch.
    ///
    /// # Arguments
    /// - `outputs`: the network's scores with the class axis first (axis 0); every trailing axis
    ///   — the samples, plus any spatial axes — is a position scored independently.
    /// - `targets`: a single class-id row on axis 0, matching `outputs` on every other axis.
    ///
    /// # Returns
    /// The accuracy as a percentage (0.0 to 100.0): the fraction of positions whose predicted
    /// class matches the target id.
    ///
    /// # Panics
    /// When `targets`' leading axis does not have length one, when `outputs` and `targets` do
    /// not agree on every other axis, or when there are fewer than two classes on `outputs`'
    /// leading axis.
    fn compute(&self, outputs: ArrayViewD<f32>, targets: ArrayViewD<f32>) -> f32 {
        assert_eq!(
            targets.shape()[0],
            1,
            "Sparse targets must carry a single class-id row."
        );
        assert_eq!(
            outputs.shape()[1..],
            targets.shape()[1..],
            "Outputs and targets must agree on every axis but the class axis."
        );
        assert!(
            outputs.shape()[0] >= 2,
            "Categorical accuracy expects at least two classes on the leading axis."
        );

        // Every axis but the class axis is a position: samples, plus any spatial axes.
        let positions = outputs.len() / outputs.shape()[0];

        // The class axis is a lane per position; the prediction is its argmax, matched against
        // the target id at that position.
        let argmax = |lane: ArrayView1<f32>| -> usize {
            lane.iter()
                .enumerate()
                .max_by(|(_, left), (_, right)| left.total_cmp(right))
                .map(|(idx, _)| idx)
                .unwrap()
        };
        let ids = targets.index_axis(Axis(0), 0);
        let correct = outputs
            .lanes(Axis(0))
            .into_iter()
            .zip(ids.iter())
            .filter(|(prediction, id)| argmax(*prediction) == **id as usize)
            .count();

        correct as f32 * 100.0 / (positions as f32)
    }
}

/// Shared static instance of `SparseCategoricalAccuracy` for convenient reuse.
pub static SPARSE_CATEGORICAL_ACCURACY: Lazy<Arc<SparseCategoricalAccuracy>> =
    Lazy::new(|| Arc::new(SparseCategoricalAccuracy));

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn all_correct_gives_100_percent() {
        let outputs = array![[0.9_f32, 0.1], [0.1, 0.9]].into_dyn();
        let targets = array![[0.0_f32, 1.0]].into_dyn();
        let acc = SPARSE_CATEGORICAL_ACCURACY.compute(outputs.view(), targets.view());
        assert!((acc - 100.0).abs() < 1e-5, "Expected 100%, got {acc}");
    }

    #[test]
    fn does_not_panic_on_nan_predictions() {
        let outputs = array![[f32::NAN, 1.0], [0.5_f32, 0.8]].into_dyn();
        let targets = array![[1.0_f32, 0.0]].into_dyn();
        let _ = SPARSE_CATEGORICAL_ACCURACY.compute(outputs.view(), targets.view());
    }

    #[test]
    fn scores_every_spatial_position_by_argmax_over_the_class_axis() {
        // (n_classes=2, height=2, samples=2): argmax over axis 0 at each of the 4 positions.
        // Three of the four positions match their target id → 75%.
        let outputs = array![[[2.0_f32, 1.0], [2.0, 1.0]], [[1.0, 2.0], [1.0, 2.0]]].into_dyn();
        let targets = array![[[0.0_f32, 1.0], [1.0, 1.0]]].into_dyn();
        let acc = SPARSE_CATEGORICAL_ACCURACY.compute(outputs.view(), targets.view());
        assert!((acc - 75.0).abs() < 1e-5, "Expected 75%, got {acc}");
    }

    #[test]
    #[should_panic(expected = "at least two classes")]
    fn panics_on_a_single_class_row() {
        let outputs = array![[0.5_f32, -0.5]].into_dyn();
        let targets = array![[0.0_f32, 0.0]].into_dyn();
        let _ = SPARSE_CATEGORICAL_ACCURACY.compute(outputs.view(), targets.view());
    }

    #[test]
    #[should_panic(expected = "single class-id row")]
    fn panics_on_a_one_hot_target_shape() {
        let outputs = array![[0.9_f32, 0.1], [0.1, 0.9]].into_dyn();
        let targets = array![[1.0_f32, 0.0], [0.0, 1.0]].into_dyn();
        let _ = SPARSE_CATEGORICAL_ACCURACY.compute(outputs.view(), targets.view());
    }
}
