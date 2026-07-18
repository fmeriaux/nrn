//! [`NetworkConfig`]: the per-sample input shape plus an ordered stack of [`LayerConfig`]s.

use super::layer::LayerConfig;
use crate::activations::Activation;
use std::sync::Arc;

/// A network's architecture with no weights: the per-sample input shape and the ordered stack
/// of layer configs — the declarative counterpart to a [`NeuralNetwork`](crate::model::NeuralNetwork).
#[derive(Debug, Clone)]
pub struct NetworkConfig {
    /// The per-sample input shape, sample axis excluded.
    pub input_shape: Vec<usize>,
    /// The layers in order, from input to output.
    pub layers: Vec<LayerConfig>,
}

impl NetworkConfig {
    /// Starts a [`NetworkConfigBuilder`] at the given per-sample input shape.
    pub fn builder(input_shape: Vec<usize>) -> NetworkConfigBuilder {
        NetworkConfigBuilder::new(input_shape)
    }
}

/// Fluent builder for a [`NetworkConfig`]: start from the per-sample input shape, append layers
/// in order, then finish with [`build`](Self::build).
pub struct NetworkConfigBuilder {
    input_shape: Vec<usize>,
    layers: Vec<LayerConfig>,
}

impl NetworkConfigBuilder {
    /// Starts a builder at the given per-sample input shape, with no layers yet.
    pub fn new(input_shape: Vec<usize>) -> Self {
        Self {
            input_shape,
            layers: Vec::new(),
        }
    }

    /// Appends a [`Conv2d`](LayerConfig::Conv2d) layer.
    pub fn conv2d<A: Activation + 'static>(
        mut self,
        out_channels: usize,
        kernel: (usize, usize),
        stride: usize,
        padding: usize,
        activation: &Arc<A>,
    ) -> Self {
        self.layers.push(LayerConfig::Conv2d {
            out_channels,
            kernel,
            stride,
            padding,
            activation: activation.clone(),
        });
        self
    }

    /// Appends a [`Flatten`](LayerConfig::Flatten) layer.
    pub fn flatten(mut self) -> Self {
        self.layers.push(LayerConfig::Flatten);
        self
    }

    /// Appends a [`Dense`](LayerConfig::Dense) layer.
    pub fn dense<A: Activation + 'static>(mut self, neurons: usize, activation: &Arc<A>) -> Self {
        self.layers.push(LayerConfig::Dense {
            neurons,
            activation: activation.clone(),
        });
        self
    }

    /// Finishes the builder: the last added layer is the output.
    pub fn build(self) -> NetworkConfig {
        NetworkConfig {
            input_shape: self.input_shape,
            layers: self.layers,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::{IDENTITY, RELU};

    #[test]
    fn builder_assembles_layers_in_order() {
        let config = NetworkConfig::builder(vec![1, 4, 4])
            .conv2d(2, (3, 3), 1, 0, &RELU)
            .flatten()
            .dense(4, &RELU)
            .dense(1, &IDENTITY)
            .build();

        assert_eq!(config.input_shape, vec![1, 4, 4]);
        assert_eq!(config.layers.len(), 4);
        assert!(matches!(config.layers[0], LayerConfig::Conv2d { .. }));
        assert!(matches!(config.layers[1], LayerConfig::Flatten));
        assert!(matches!(
            config.layers[2],
            LayerConfig::Dense { neurons: 4, .. }
        ));
        match &config.layers[3] {
            LayerConfig::Dense {
                neurons: 1,
                activation,
            } => assert_eq!(activation.name(), "identity"),
            other => panic!("expected the head to be a Dense layer, got {other:?}"),
        }
    }
}
