//! Loss functions used to evaluate model performance and drive gradient descent.
//!
//! Every implementation defines the per-term error (`terms`), from which the scalar
//! evaluation (`compute`) is derived by [`Reduction`], plus the analytical derivative
//! with respect to the evaluated network state (`gradient`).

mod binary_cross_entropy;
mod categorical_cross_entropy;

pub use binary_cross_entropy::BinaryCrossEntropy;
pub use categorical_cross_entropy::CategoricalCrossEntropy;

use ndarray::{ArrayD, ArrayViewD};

/// How a loss reduces its per-term values into the single scalar reported for a batch.
///
/// The per-term granularity is the loss's own: element-wise for binary cross-entropy
/// (one term per logit), per-position for categorical cross-entropy (the class lane is
/// already summed). `Mean` therefore averages over the same unit — positions — for both.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Reduction {
    /// Sum of every term.
    Sum,
    /// Mean over every term.
    #[default]
    Mean,
}

impl Reduction {
    /// Collapses a loss's per-term array into the reported scalar.
    fn reduce(self, terms: ArrayD<f32>) -> f32 {
        match self {
            Reduction::Sum => terms.sum(),
            Reduction::Mean => terms.sum() / terms.len() as f32,
        }
    }

    /// Scales a raw sum-of-terms `gradient` to match this reduction over `n_terms` terms.
    fn scale_gradient(self, gradient: ArrayD<f32>, n_terms: usize) -> ArrayD<f32> {
        match self {
            Reduction::Sum => gradient,
            Reduction::Mean => gradient / n_terms as f32,
        }
    }
}

/// Defines the contract for a loss function.
///
/// Implementations of this trait are decoupled from specific model tasks. Standard
/// classification losses (like Cross-Entropy) expect these `inputs` to be raw,
/// un-activated scores (**logits**).
pub trait LossFunction: Send + Sync {
    /// Returns a human-readable name for this loss function.
    fn name(&self) -> &'static str;

    /// The [`Reduction`] this loss applies to turn its per-term values into a scalar.
    fn reduction(&self) -> Reduction;

    /// Computes the per-term loss at the function's native granularity — element-wise
    /// for binary cross-entropy, per-position for categorical — before reduction.
    ///
    /// # Arguments
    ///
    /// * `inputs` - An N-Dimensional array view of the network's final layer outputs.
    ///   Laid out samples-last, with the features/classes on the leading axis.
    /// * `targets` - An N-Dimensional array view of the expected true values, matching
    ///   the exact shape of `inputs`.
    fn terms(&self, inputs: ArrayViewD<f32>, targets: ArrayViewD<f32>) -> ArrayD<f32>;

    /// Computes the scalar loss over the given batch: [`terms`](Self::terms) collapsed by
    /// the loss's [`reduction`](Self::reduction).
    ///
    /// # Arguments
    ///
    /// * `inputs` - An N-Dimensional array view of the network's final layer outputs.
    ///   Laid out samples-last, with the features/classes on the leading axis.
    /// * `targets` - An N-Dimensional array view of the expected true values, matching
    ///   the exact shape of `inputs`.
    fn compute(&self, inputs: ArrayViewD<f32>, targets: ArrayViewD<f32>) -> f32 {
        self.reduction().reduce(self.terms(inputs, targets))
    }

    /// Computes `∂L/∂inputs` — the gradient of the reduced loss
    /// ([`compute`](Self::compute)) with respect to the incoming network outputs,
    /// carrying the same [`reduction`](Self::reduction) as the reported value.
    ///
    /// This array serves as the entry point for the backward pass, which propagates
    /// these derivatives back through the network layers.
    ///
    /// # Arguments
    ///
    /// * `inputs` - An N-Dimensional array view of the network's final layer outputs.
    ///   Laid out samples-last, with the features/classes on the leading axis.
    /// * `targets` - An N-Dimensional array view of the expected true values, matching
    ///   the exact shape of `inputs`.
    ///
    /// # Returns
    /// An owned N-Dimensional array (`ArrayD<f32>`) containing the gradients,
    /// matching the exact shape of `inputs`.
    fn gradient(&self, inputs: ArrayViewD<f32>, targets: ArrayViewD<f32>) -> ArrayD<f32>;
}
