//! Layer abstraction for a neural network.
//!
//! A [`Layer`] is one stage of the network: it maps a batch of inputs to a batch of
//! outputs in the forward pass and, in the backward pass, turns the gradient of the loss
//! with respect to its output into the gradients of its parameters and the gradient with
//! respect to its input. Layers are stateless with respect to the forward activations —
//! the network threads those between layers — so a layer holds only its parameters.

mod dense;
mod flatten;

pub use dense::Dense;
pub use flatten::Flatten;

use crate::gradients::LayerGradients;
use dyn_clone::DynClone;
use ndarray::{ArrayD, ArrayView2, ArrayViewD, ArrayViewMutD};
use std::any::Any;
use std::fmt::Debug;

/// One trainable parameter of a layer, exposed for an optimizer to update in place.
pub struct Parameter<'a> {
    /// A mutable view of the parameter's values.
    pub value: ArrayViewMutD<'a, f32>,
    /// Whether weight decay applies to this parameter: weights decay, biases do not.
    pub decays: bool,
}

/// The result of a layer's backward pass.
pub struct BackwardPass {
    /// The gradients of the layer's parameters, in parameter order.
    pub gradients: LayerGradients,
    /// The gradient of the loss with respect to the layer's input, or `None` when the
    /// caller did not request it.
    pub input_gradient: Option<ArrayD<f32>>,
}

/// One stage of a neural network, mapping a batch of inputs to a batch of outputs.
///
/// Arrays are `(features, samples)`: columns are samples, rows are features.
pub trait Layer: DynClone + Debug {
    /// Computes the forward pass of this layer given the inputs.
    /// # Arguments
    /// - `input`: An array `(input_size, samples)` representing the inputs to this layer.
    /// # Returns
    /// - An array `(output_size, samples)` representing the outputs of this layer.
    fn forward(&self, input: ArrayViewD<f32>) -> ArrayD<f32>;

    /// Computes the backward pass of this layer for one batch, propagating the gradient
    /// back one stage.
    /// # Arguments
    /// - `da`: An array, the gradient of the loss with respect to this layer's output.
    /// - `input`: The batch that was fed to this layer in the forward pass.
    /// - `output`: This layer's output from the forward pass.
    /// - `compute_input_gradient`: Whether to also compute the gradient with respect to `input`.
    /// # Returns
    /// - A [`BackwardPass`] carrying the parameter gradients and, when
    ///   `compute_input_gradient` is set, the gradient of the loss with respect to `input` —
    ///   the `da` the upstream layer receives.
    fn backward(
        &self,
        da: ArrayViewD<f32>,
        input: ArrayViewD<f32>,
        output: ArrayViewD<f32>,
        compute_input_gradient: bool,
    ) -> BackwardPass;

    /// The layer's trainable parameters, in a stable order matching the gradient order
    /// returned by [`backward`](Layer::backward).
    fn parameters_mut(&mut self) -> Vec<Parameter<'_>>;

    /// The number of input features this layer expects.
    fn input_size(&self) -> usize;

    /// The number of output features this layer produces.
    fn output_size(&self) -> usize;

    /// Whether every parameter value is finite (no NaN or Inf).
    fn is_finite(&self) -> bool;

    /// A short identifier for the layer type, such as `"dense"`.
    fn kind(&self) -> &'static str;

    /// The layer's parameters as named tensors.
    fn named_tensors(&self) -> Vec<(String, ArrayD<f32>)>;

    /// The name of the activation this layer applies, or `None` for a layer without one.
    fn activation_name(&self) -> Option<&str>;

    /// The layer's weight matrix `(output_size, input_size)`, or `None` for a layer
    /// that is not an affine map and carries no weights.
    fn weight_matrix(&self) -> Option<ArrayView2<'_, f32>>;

    /// The layer as `&dyn Any`, for downcasting to a concrete layer type.
    fn as_any(&self) -> &dyn Any;

    /// The layer as `&mut dyn Any`, for downcasting to a concrete layer type.
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

dyn_clone::clone_trait_object!(Layer);
