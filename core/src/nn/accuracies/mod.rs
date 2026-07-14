mod binary_accuracy;
mod categorical_accuracy;

pub use binary_accuracy::{BINARY_ACCURACY, BinaryAccuracy};
pub use categorical_accuracy::{CATEGORICAL_ACCURACY, CategoricalAccuracy};

use ndarray::ArrayViewD;

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
