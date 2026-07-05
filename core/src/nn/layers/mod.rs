//! Layer abstraction for a neural network.
//!
//! A [`Layer`] is one stage of the network: it maps a batch of inputs to a batch of
//! outputs in the forward pass and, in the backward pass, turns the gradient of the loss
//! with respect to its output into the gradients of its parameters and the gradient with
//! respect to its input. Layers are stateless with respect to the forward activations —
//! the network threads those between layers — so a layer holds only its parameters.

mod conv2d;
mod dense;
mod flatten;

pub use conv2d::Conv2d;
pub use dense::Dense;
pub use flatten::Flatten;

use crate::activations::{Activation, ActivationProvider};
use crate::gradients::LayerGradients;
use dyn_clone::DynClone;
use ndarray::{Array, ArrayD, ArrayView2, ArrayViewD, ArrayViewMutD, Dimension};
use std::any::Any;
use std::collections::HashMap;
use std::fmt::{self, Debug};
use std::sync::Arc;

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

    /// The per-sample input shape this layer expects, the sample axis excluded — the
    /// single source of truth for the layer's input geometry (e.g. `[features]` for a
    /// dense layer, `[channels, height, width]` for a convolution).
    fn input_shape(&self) -> Vec<usize>;

    /// The per-sample output shape this layer produces, the sample axis excluded.
    fn output_shape(&self) -> Vec<usize>;

    /// The number of input features this layer expects: the product of [`input_shape`](Layer::input_shape).
    fn input_size(&self) -> usize {
        self.input_shape().iter().product()
    }

    /// The number of output features this layer produces: the product of [`output_shape`](Layer::output_shape).
    fn output_size(&self) -> usize {
        self.output_shape().iter().product()
    }

    /// Whether every parameter value is finite (no NaN or Inf).
    fn is_finite(&self) -> bool;

    /// The concrete kind of this layer.
    fn kind(&self) -> LayerKind;

    /// The layer's non-tensor hyperparameters, as name/value pairs: the configuration that
    /// describes this layer beyond its [`named_tensors`](Layer::named_tensors), such as a
    /// convolution's stride and padding or its activation's name. Empty for a layer with no
    /// such configuration.
    fn config(&self) -> Vec<(String, String)> {
        vec![]
    }

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

/// The concrete kind of a [`Layer`]. It names the layer type and rebuilds a layer of that
/// kind from its [`config`](Layer::config) and [`named_tensors`](Layer::named_tensors) via
/// [`instantiate`](LayerKind::instantiate).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind {
    /// A fully connected [`Dense`] layer.
    Dense,
    /// A 2D convolution [`Conv2d`] layer.
    Conv2d,
    /// A reshaping [`Flatten`] layer.
    Flatten,
}

impl LayerKind {
    /// The stable string tag naming this kind.
    pub fn as_str(&self) -> &'static str {
        match self {
            LayerKind::Dense => "dense",
            LayerKind::Conv2d => "conv2d",
            LayerKind::Flatten => "flatten",
        }
    }

    /// Builds a layer of this kind from its configuration and tensors.
    /// # Arguments
    /// - `config`: The layer's non-tensor hyperparameters, as produced by [`Layer::config`].
    /// - `tensors`: The layer's named tensors, as produced by [`Layer::named_tensors`].
    pub fn instantiate(
        &self,
        config: &HashMap<String, String>,
        tensors: HashMap<String, ArrayD<f32>>,
    ) -> Result<Box<dyn Layer>, LayerConfigError> {
        Ok(match self {
            LayerKind::Dense => Box::new(Dense::from_config(config, tensors)?),
            LayerKind::Conv2d => Box::new(Conv2d::from_config(config, tensors)?),
            LayerKind::Flatten => Box::new(Flatten::from_config(config, tensors)?),
        })
    }
}

impl TryFrom<&str> for LayerKind {
    type Error = LayerConfigError;

    fn try_from(tag: &str) -> Result<Self, LayerConfigError> {
        match tag {
            "dense" => Ok(LayerKind::Dense),
            "conv2d" => Ok(LayerKind::Conv2d),
            "flatten" => Ok(LayerKind::Flatten),
            other => Err(LayerConfigError::UnknownKind(other.to_string())),
        }
    }
}

/// Error returned when a layer cannot be built from its configuration and tensors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayerConfigError {
    /// A required configuration key was absent.
    MissingConfig(String),
    /// A configuration value could not be parsed.
    InvalidConfig {
        /// The offending key.
        key: String,
        /// Why the value was rejected.
        reason: String,
    },
    /// A required tensor was absent.
    MissingTensor(String),
    /// A tensor did not have the rank the layer requires.
    WrongTensorRank {
        /// The tensor's name.
        name: String,
        /// The rank the layer requires.
        expected: usize,
        /// The rank the tensor actually had.
        got: usize,
    },
    /// The named activation is not registered.
    UnknownActivation(String),
    /// The kind tag does not name a known layer.
    UnknownKind(String),
}

impl fmt::Display for LayerConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LayerConfigError::MissingConfig(key) => write!(f, "missing config key `{key}`"),
            LayerConfigError::InvalidConfig { key, reason } => {
                write!(f, "config key `{key}` is invalid: {reason}")
            }
            LayerConfigError::MissingTensor(name) => write!(f, "missing tensor `{name}`"),
            LayerConfigError::WrongTensorRank {
                name,
                expected,
                got,
            } => write!(f, "tensor `{name}` has rank {got}, expected {expected}"),
            LayerConfigError::UnknownActivation(name) => {
                write!(f, "unknown activation function: {name}")
            }
            LayerConfigError::UnknownKind(tag) => write!(f, "unknown layer kind: {tag}"),
        }
    }
}

impl std::error::Error for LayerConfigError {}

/// Reads a required string entry from a layer's configuration.
fn config_str<'a>(
    config: &'a HashMap<String, String>,
    key: &str,
) -> Result<&'a str, LayerConfigError> {
    config
        .get(key)
        .map(String::as_str)
        .ok_or_else(|| LayerConfigError::MissingConfig(key.to_string()))
}

/// Reads a required unsigned integer entry from a layer's configuration.
fn config_usize(config: &HashMap<String, String>, key: &str) -> Result<usize, LayerConfigError> {
    config_str(config, key)?
        .parse()
        .map_err(
            |e: std::num::ParseIntError| LayerConfigError::InvalidConfig {
                key: key.to_string(),
                reason: e.to_string(),
            },
        )
}

/// Reads a comma-separated dimension list (e.g. `"1,28,28"`) from a layer's configuration.
fn config_dims(
    config: &HashMap<String, String>,
    key: &str,
) -> Result<Vec<usize>, LayerConfigError> {
    config_str(config, key)?
        .split(',')
        .map(|part| {
            part.trim()
                .parse::<usize>()
                .map_err(|e| LayerConfigError::InvalidConfig {
                    key: key.to_string(),
                    reason: e.to_string(),
                })
        })
        .collect()
}

/// Resolves the activation named under `"activation"` in a layer's configuration.
fn config_activation(
    config: &HashMap<String, String>,
) -> Result<Arc<dyn Activation>, LayerConfigError> {
    let name = config_str(config, "activation")?;
    ActivationProvider::get_by_name(name)
        .ok_or_else(|| LayerConfigError::UnknownActivation(name.to_string()))
}

/// Removes a required tensor and casts it to the rank `D` the layer expects.
fn take_tensor<D: Dimension>(
    tensors: &mut HashMap<String, ArrayD<f32>>,
    name: &str,
) -> Result<Array<f32, D>, LayerConfigError> {
    let tensor = tensors
        .remove(name)
        .ok_or_else(|| LayerConfigError::MissingTensor(name.to_string()))?;
    let got = tensor.ndim();
    tensor
        .into_dimensionality::<D>()
        .map_err(|_| LayerConfigError::WrongTensorRank {
            name: name.to_string(),
            expected: D::NDIM.unwrap_or(got),
            got,
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kind_tags_round_trip_and_reject_unknown() {
        for kind in [LayerKind::Dense, LayerKind::Conv2d, LayerKind::Flatten] {
            assert_eq!(LayerKind::try_from(kind.as_str()), Ok(kind));
        }
        assert_eq!(
            LayerKind::try_from("mystery"),
            Err(LayerConfigError::UnknownKind("mystery".to_string()))
        );
    }

    #[test]
    fn config_usize_reports_the_offending_key_on_a_non_integer_value() {
        let config = HashMap::from([("stride".to_string(), "wide".to_string())]);
        let error = config_usize(&config, "stride").unwrap_err();

        assert!(
            matches!(&error, LayerConfigError::InvalidConfig { key, .. } if key == "stride"),
            "unexpected error: {error:?}"
        );
    }

    #[test]
    fn config_dims_rejects_a_non_integer_dimension() {
        let config = HashMap::from([("shape".to_string(), "2,x,4".to_string())]);
        let error = config_dims(&config, "shape").unwrap_err();

        assert!(
            matches!(&error, LayerConfigError::InvalidConfig { key, .. } if key == "shape"),
            "unexpected error: {error:?}"
        );
    }

    #[test]
    fn errors_display_their_key_and_tag() {
        let invalid = LayerConfigError::InvalidConfig {
            key: "stride".to_string(),
            reason: "not a number".to_string(),
        };
        assert_eq!(
            invalid.to_string(),
            "config key `stride` is invalid: not a number"
        );
        assert_eq!(
            LayerConfigError::UnknownKind("mystery".to_string()).to_string(),
            "unknown layer kind: mystery"
        );
    }
}
