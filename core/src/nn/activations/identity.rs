//! Identity activation function implementation.
//!
//! This module provides the `Identity` struct, the identity activation `f(x) = x`. It is the
//! output activation of a network trained on **logits**: softmax/sigmoid is not applied in the
//! forward pass but folded into the cross-entropy loss (training) and reapplied as a
//! post-processing step at inference. An identity output keeps the backward pass uniform — its
//! Jacobian is the identity, so its VJP passes the upstream gradient straight through — which
//! is exactly what lets the fused loss gradient `p − y` reach the affine map unchanged.

use crate::activations::{Activation, ActivationProvider};
use crate::initializations::{Initialization, XAVIER_UNIFORM};
use ndarray::{ArrayD, ArrayViewD};
use once_cell::sync::Lazy;
use std::sync::Arc;

#[derive(Debug)]
pub struct Identity;

impl Activation for Identity {
    /// Returns the canonical name of the activation function.
    fn name(&self) -> &'static str {
        "identity"
    }

    /// Returns the input unchanged: the identity `f(x) = x`.
    fn apply(&self, input: ArrayViewD<f32>) -> ArrayD<f32> {
        input.to_owned()
    }

    /// Computes ∂L/∂z = upstream: the identity Jacobian passes the upstream gradient through.
    fn vjp(&self, upstream: ArrayViewD<f32>, _activations: ArrayViewD<f32>) -> ArrayD<f32> {
        upstream.to_owned()
    }

    /// Provides the recommended initialization for an identity (linear) layer.
    ///
    /// Xavier initialization keeps the pre-activation variance stable for an identity map.
    fn initialization(&self) -> Arc<dyn Initialization> {
        XAVIER_UNIFORM.clone()
    }
}

/// Static instance of the Identity activation wrapped in an `Arc` for shared use.
pub static IDENTITY: Lazy<Arc<Identity>> = Lazy::new(|| Arc::new(Identity));
inventory::submit!(ActivationProvider(|| IDENTITY.clone()));

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn name_is_identity() {
        assert_eq!(Identity.name(), "identity");
    }

    #[test]
    fn apply_returns_input_unchanged() {
        let input = array![[1.0, -2.0], [3.0, 4.5]].into_dyn();
        assert_eq!(Identity.apply(input.view()), input);
    }

    #[test]
    fn vjp_passes_upstream_through_unchanged() {
        let upstream = array![[0.1, -0.2], [0.3, 0.4]].into_dyn();
        let activations = array![[5.0, 6.0], [7.0, 8.0]].into_dyn();
        assert_eq!(Identity.vjp(upstream.view(), activations.view()), upstream);
    }

    #[test]
    fn is_registered_by_name() {
        let identity = ActivationProvider::get_by_name("identity").expect("identity is registered");
        assert_eq!(identity.name(), "identity");
    }
}
