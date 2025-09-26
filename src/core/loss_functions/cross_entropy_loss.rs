//! Cross entropy loss module for binary and multi-class classification.
//!
//! This module defines the `LossFunction` trait and a cross-entropy
//! implementation usable for classification tasks.
//!
//! The loss computes the difference between predicted probabilities
//! and true one-hot labels, providing both the scalar loss and its gradient.

use crate::core::loss_functions::LossFunction;
use ndarray::{Array2, ArrayView2};
use once_cell::sync::Lazy;
use std::sync::Arc;

pub struct CrossEntropyLoss;

impl CrossEntropyLoss {
    /// Clips the predicted probabilities so that they lie within a safe numerical range
    /// to prevent invalid logarithms during loss computation.
    fn clip_probabilities(probabilities: &ArrayView2<f32>) -> Array2<f32> {
        let epsilon = 1e-15;
        probabilities.mapv(|p| p.clamp(epsilon, 1.0 - epsilon))
    }
}

impl LossFunction for CrossEntropyLoss {
    /// Computes the scalar loss value given predictions and true labels.
    ///
    /// # Arguments
    /// * `predictions` - 2D array where each row is the predicted probability distribution for a sample.
    /// * `targets` - 2D array where each row is the true one-hot encoded label for a sample.
    ///
    /// # Returns
    /// Average loss over the batch.
    fn compute(&self, predictions: ArrayView2<f32>, targets: ArrayView2<f32>) -> f32 {
        let clipped_predictions = Self::clip_probabilities(&predictions);
        let log_predictions = clipped_predictions.mapv(|p| p.ln());
        let n_samples = predictions.ncols() as f32;
        -(&targets * &log_predictions).sum() / n_samples
    }

    /// Computes the gradient of the loss with respect to the predictions.
    ///
    /// # Arguments
    /// * `predictions` - Same as in `compute`.
    /// * `targets` - Same as in `compute`.
    ///
    /// # Returns
    /// Gradient array (`Array2<f32>`) of the same shape as predictions, corresponding to
    /// the derivative of the loss with respect to each predicted probability.
    fn gradient(&self, predictions: ArrayView2<f32>, targets: ArrayView2<f32>) -> Array2<f32> {
        let clipped_predictions = Self::clip_probabilities(&predictions);
        clipped_predictions - &targets
    }
}

/// Static instance of the CrossEntropyLoss wrapped in an `Arc` for shared use.
pub static CROSS_ENTROPY_LOSS: Lazy<Arc<CrossEntropyLoss>> =
    Lazy::new(|| Arc::new(CrossEntropyLoss));
