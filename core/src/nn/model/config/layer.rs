//! [`LayerConfig`]: a typed, weight-free description of one layer.

use crate::activations::Activation;
use crate::layers::{Conv2d, Dense, Flatten, Layer};
use crate::tensors::{TensorError, Tensors};
use ndarray::{Ix2, Ix4};
use ndarray_rand::rand::RngCore;
use std::fmt;
use std::sync::Arc;

/// A typed, weight-free description of one layer: its kind and the hyperparameters needed to
/// build it, the concrete input shape excepted (that is threaded from the network's input).
///
/// It spans every [`Layer`] kind and is the declarative counterpart to a layer's weights:
/// combined with weights — freshly [initialized](LayerConfig::initialization) or supplied as
/// [named tensors](LayerConfig::from_tensors) — a config yields a live layer.
#[derive(Clone, Debug)]
pub enum LayerConfig {
    /// A fully connected [`Dense`] layer.
    Dense {
        /// The number of neurons, i.e. output features.
        neurons: usize,
        /// The activation applied to the layer's output.
        activation: Arc<dyn Activation>,
    },
    /// A 2D convolution [`Conv2d`] layer.
    Conv2d {
        /// The number of filters, i.e. output channels.
        out_channels: usize,
        /// The kernel's spatial size `(height, width)`.
        kernel: (usize, usize),
        /// The stride applied on both spatial axes.
        stride: usize,
        /// The zero-padding added on both spatial axes.
        padding: usize,
        /// The activation applied to the layer's output.
        activation: Arc<dyn Activation>,
    },
    /// A reshaping [`Flatten`] layer.
    Flatten,
}

impl LayerConfig {
    /// Builds a fresh layer at `input_shape`, drawing its weights from `rng`.
    pub fn initialization(
        &self,
        input_shape: &[usize],
        rng: &mut dyn RngCore,
    ) -> Result<Box<dyn Layer>, LayerConfigError> {
        Ok(match self {
            LayerConfig::Dense {
                neurons,
                activation,
            } => {
                if *neurons == 0 {
                    return Err(LayerConfigError::EmptyLayer);
                }
                let inputs = flat_input(input_shape)?;
                Box::new(Dense::initialization(
                    inputs,
                    *neurons,
                    activation.clone(),
                    rng,
                ))
            }
            LayerConfig::Conv2d {
                out_channels,
                kernel,
                stride,
                padding,
                activation,
            } => {
                if *out_channels == 0 {
                    return Err(LayerConfigError::EmptyLayer);
                }
                let input = conv_input(input_shape, *kernel, *padding)?;
                Box::new(Conv2d::initialization(
                    input,
                    *out_channels,
                    *kernel,
                    *stride,
                    *padding,
                    activation.clone(),
                    rng,
                ))
            }
            LayerConfig::Flatten => Box::new(Flatten::new(input_shape.to_vec())),
        })
    }

    /// Builds a layer at `input_shape`, taking its weights from the given named `tensors`.
    pub fn from_tensors(
        &self,
        input_shape: &[usize],
        mut tensors: Tensors,
    ) -> Result<Box<dyn Layer>, LayerConfigError> {
        Ok(match self {
            LayerConfig::Dense {
                neurons,
                activation,
            } => {
                if *neurons == 0 {
                    return Err(LayerConfigError::EmptyLayer);
                }
                flat_input(input_shape)?;
                let weights = tensors
                    .take_weight::<Ix2>()
                    .map_err(LayerConfigError::Tensor)?;
                let biases = tensors.take_bias().map_err(LayerConfigError::Tensor)?;
                Box::new(Dense::new(weights, biases, activation.clone()))
            }
            LayerConfig::Conv2d {
                out_channels,
                kernel,
                stride,
                padding,
                activation,
            } => {
                if *out_channels == 0 {
                    return Err(LayerConfigError::EmptyLayer);
                }
                let input = conv_input(input_shape, *kernel, *padding)?;
                let kernels = tensors
                    .take_weight::<Ix4>()
                    .map_err(LayerConfigError::Tensor)?;
                let biases = tensors.take_bias().map_err(LayerConfigError::Tensor)?;
                Box::new(Conv2d::new(
                    kernels,
                    biases,
                    input,
                    *stride,
                    *padding,
                    activation.clone(),
                ))
            }
            LayerConfig::Flatten => Box::new(Flatten::new(input_shape.to_vec())),
        })
    }
}

/// Validates that a dense layer's input is flat (rank-1), returning its feature count.
fn flat_input(input_shape: &[usize]) -> Result<usize, LayerConfigError> {
    match *input_shape {
        [inputs] => Ok(inputs),
        _ => Err(LayerConfigError::UnexpectedInputRank {
            expected: 1,
            got: input_shape.to_vec(),
        }),
    }
}

/// Validates that a convolution's input is rank-3 and admits the kernel, returning its
/// `(channels, height, width)`.
fn conv_input(
    input_shape: &[usize],
    kernel: (usize, usize),
    padding: usize,
) -> Result<(usize, usize, usize), LayerConfigError> {
    let &[channels, height, width] = input_shape else {
        return Err(LayerConfigError::UnexpectedInputRank {
            expected: 3,
            got: input_shape.to_vec(),
        });
    };
    let (kh, kw) = kernel;
    if height + 2 * padding < kh || width + 2 * padding < kw {
        return Err(LayerConfigError::WindowDoesNotFit {
            input: (height, width),
            window: kernel,
            padding,
        });
    }
    Ok((channels, height, width))
}

/// Error returned when a [`LayerConfig`] cannot be resolved into a layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayerConfigError {
    /// The layer was given an input of the wrong rank — a dense head needs a flat (rank-1)
    /// input, a convolution a rank-3 `(channels, height, width)` one; insert a [`Flatten`]
    /// where the rank must drop.
    UnexpectedInputRank {
        /// The input rank the layer requires.
        expected: usize,
        /// The per-sample input shape the layer received.
        got: Vec<usize>,
    },
    /// The sliding window (a convolution kernel, a pooling extent), with padding, does not fit
    /// the input's spatial extent — no valid output position exists.
    WindowDoesNotFit {
        /// The input's spatial extent `(height, width)`.
        input: (usize, usize),
        /// The window's spatial size `(height, width)`.
        window: (usize, usize),
        /// The zero-padding added on both spatial axes.
        padding: usize,
    },
    /// A tensor required to build a layer's weights was absent or had the wrong rank.
    Tensor(TensorError),
    /// A layer was configured with zero output units (a Dense with no neurons, a Conv2d with
    /// no filters) — every layer must produce at least one output.
    EmptyLayer,
    /// A network was assembled with no layers at all.
    NoLayers,
}

impl fmt::Display for LayerConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LayerConfigError::UnexpectedInputRank { expected, got } => write!(
                f,
                "layer expects a rank-{expected} input, got shape {got:?}"
            ),
            LayerConfigError::WindowDoesNotFit {
                input,
                window,
                padding,
            } => write!(
                f,
                "window {window:?} with padding {padding} does not fit input {input:?}"
            ),
            LayerConfigError::Tensor(error) => write!(f, "{error}"),
            LayerConfigError::EmptyLayer => {
                write!(f, "a layer must have at least one output unit")
            }
            LayerConfigError::NoLayers => {
                write!(f, "a network must have at least one layer")
            }
        }
    }
}

impl std::error::Error for LayerConfigError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::RELU;

    use ndarray_rand::rand::SeedableRng;
    use ndarray_rand::rand::rngs::StdRng;

    #[test]
    fn dense_spec_initializes_a_layer_at_the_flat_input() {
        let spec = LayerConfig::Dense {
            neurons: 4,
            activation: RELU.clone(),
        };
        let mut rng = StdRng::seed_from_u64(0);
        let layer = spec.initialization(&[3], &mut rng).unwrap();

        assert_eq!(layer.input_shape(), vec![3]);
        assert_eq!(layer.output_shape(), vec![4]);
        assert_eq!(layer.activation_name(), Some("relu"));
    }

    #[test]
    fn conv_spec_initializes_a_layer_at_the_spatial_input() {
        let spec = LayerConfig::Conv2d {
            out_channels: 2,
            kernel: (3, 3),
            stride: 1,
            padding: 0,
            activation: RELU.clone(),
        };
        let mut rng = StdRng::seed_from_u64(0);
        let layer = spec.initialization(&[1, 4, 4], &mut rng).unwrap();

        assert_eq!(layer.input_shape(), vec![1, 4, 4]);
        // conv_output_dim(4, 3, 1, 0) = 2 on each spatial axis.
        assert_eq!(layer.output_shape(), vec![2, 2, 2]);
    }

    #[test]
    fn dense_spec_rejects_a_non_flat_input() {
        let spec = LayerConfig::Dense {
            neurons: 4,
            activation: RELU.clone(),
        };
        let mut rng = StdRng::seed_from_u64(0);
        let err = spec.initialization(&[1, 4, 4], &mut rng).unwrap_err();

        assert_eq!(
            err,
            LayerConfigError::UnexpectedInputRank {
                expected: 1,
                got: vec![1, 4, 4],
            }
        );
    }

    #[test]
    fn conv_spec_rejects_a_non_spatial_input() {
        let spec = LayerConfig::Conv2d {
            out_channels: 2,
            kernel: (3, 3),
            stride: 1,
            padding: 0,
            activation: RELU.clone(),
        };
        let mut rng = StdRng::seed_from_u64(0);
        let err = spec.initialization(&[16], &mut rng).unwrap_err();

        assert_eq!(
            err,
            LayerConfigError::UnexpectedInputRank {
                expected: 3,
                got: vec![16],
            }
        );
    }

    #[test]
    fn conv_spec_rejects_a_window_larger_than_the_input() {
        // A 3×3 kernel with no padding cannot fit a 2×2 spatial input.
        let spec = LayerConfig::Conv2d {
            out_channels: 2,
            kernel: (3, 3),
            stride: 1,
            padding: 0,
            activation: RELU.clone(),
        };
        let mut rng = StdRng::seed_from_u64(0);
        let err = spec.initialization(&[1, 2, 2], &mut rng).unwrap_err();

        assert_eq!(
            err,
            LayerConfigError::WindowDoesNotFit {
                input: (2, 2),
                window: (3, 3),
                padding: 0,
            }
        );
    }

    #[test]
    fn empty_dense_layer_is_rejected() {
        let spec = LayerConfig::Dense {
            neurons: 0,
            activation: RELU.clone(),
        };
        let mut rng = StdRng::seed_from_u64(0);
        let err = spec.initialization(&[3], &mut rng).unwrap_err();

        assert_eq!(err, LayerConfigError::EmptyLayer);
        assert_eq!(
            err.to_string(),
            "a layer must have at least one output unit"
        );
    }

    #[test]
    fn empty_conv_layer_is_rejected() {
        let spec = LayerConfig::Conv2d {
            out_channels: 0,
            kernel: (3, 3),
            stride: 1,
            padding: 0,
            activation: RELU.clone(),
        };
        let mut rng = StdRng::seed_from_u64(0);
        let err = spec.initialization(&[1, 4, 4], &mut rng).unwrap_err();

        assert_eq!(err, LayerConfigError::EmptyLayer);
    }
}
