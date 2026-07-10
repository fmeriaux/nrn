mod binary_accuracy;
mod categorical_accuracy;

pub use binary_accuracy::{BINARY_ACCURACY, BinaryAccuracy};
pub use categorical_accuracy::{CATEGORICAL_ACCURACY, CategoricalAccuracy};

use ndarray::ArrayViewD;
use std::sync::Arc;

/// Selects the accuracy metric for a classifier with `n_classes` classes: [`BinaryAccuracy`]
/// for a two-class (single-logit) output, [`CategoricalAccuracy`] otherwise.
pub fn accuracy_for(n_classes: usize) -> Arc<dyn Accuracy> {
    // Keyed on the class count, a stand-in for the output activation (a two-class output is a
    // single sigmoid logit); revisit to key on the output activation once it is explicit.
    if n_classes == 2 {
        BINARY_ACCURACY.clone()
    } else {
        CATEGORICAL_ACCURACY.clone()
    }
}

/// An accuracy metric between a network's outputs and the expected targets.
pub trait Accuracy: Send + Sync {
    /// Computes the accuracy of `outputs` against `targets`, as the percentage of correctly
    /// classified positions.
    ///
    /// # Arguments
    /// - `outputs`: the network's scores with the class axis first (axis 0); every trailing axis —
    ///   the samples, plus any spatial axes — is a position scored independently.
    /// - `targets`: the ground-truth labels, the same shape as `outputs`.
    ///
    /// # Returns
    /// The accuracy as a percentage (0.0 to 100.0).
    fn compute(&self, outputs: ArrayViewD<f32>, targets: ArrayViewD<f32>) -> f32;
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn two_classes_selects_the_binary_metric() {
        // A single logit row thresholded at 0: BinaryAccuracy scores it, CategoricalAccuracy panics.
        let outputs = array![[0.5_f32, -0.5]].into_dyn();
        let targets = array![[1.0_f32, 0.0]].into_dyn();
        let acc = accuracy_for(2).compute(outputs.view(), targets.view());
        assert!((acc - 100.0).abs() < 1e-5, "got {acc}");
    }

    #[test]
    fn three_or_more_classes_select_the_categorical_metric() {
        // A three-class lane scored by argmax; only the categorical metric accepts it.
        let outputs = array![[2.0_f32], [1.0], [0.0]].into_dyn();
        let targets = array![[1.0_f32], [0.0], [0.0]].into_dyn();
        let acc = accuracy_for(3).compute(outputs.view(), targets.view());
        assert!((acc - 100.0).abs() < 1e-5, "got {acc}");
    }
}
