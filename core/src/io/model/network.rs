//! Persistence for a [`NeuralNetwork`]: its weights as a `.safetensors` file and its
//! architecture as a serializable [`NetworkConfigRecord`] beside them.
//! [`save_weights`](NeuralNetwork::save_weights) writes tensors only; the architecture lives in
//! the [`NetworkConfigRecord`], from which [`load_weights`](NeuralNetwork::load_weights) rebuilds it.

use crate::activations::{Activation, ActivationProvider};
use crate::io::json;
use crate::io::model::tensors;
use crate::model::{LayerConfig, NetworkConfig, NeuralNetwork};
use crate::tensors::Tensors;
use safetensors::SafeTensors;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::ErrorKind::InvalidData;
use std::io::{Error, Result};
use std::path::{Path, PathBuf};
use std::sync::Arc;

impl NeuralNetwork {
    /// Writes this network's weights to `path.safetensors`, tensors only. Each tensor is named
    /// `layers.{i}.{tensor}`; the architecture lives in a [`NetworkConfigRecord`] beside them.
    /// # Arguments
    /// - `path`: The path (without extension) to write the weights to.
    pub fn save_weights<P: AsRef<Path>>(&self, path: P) -> Result<PathBuf> {
        let entries = self
            .layers()
            .iter()
            .enumerate()
            .flat_map(|(i, layer)| {
                layer.tensors().into_iter().map(move |(name, tensor)| {
                    (format!("layers.{i}.{name}"), tensors::tensor(&tensor))
                })
            })
            .collect();
        tensors::save(path, entries, HashMap::new())
    }

    /// Rebuilds a network from a weights file written by
    /// [`save_weights`](NeuralNetwork::save_weights) and the [`NetworkConfigRecord`] describing
    /// its architecture.
    /// # Arguments
    /// - `path`: The path (without extension) to read the weights from.
    /// - `record`: The architecture the weights belong to.
    pub fn load_weights<P: AsRef<Path>>(path: P, record: &NetworkConfigRecord) -> Result<Self> {
        let bytes = tensors::load(path)?;
        let st =
            SafeTensors::deserialize(&bytes).map_err(|e| Error::new(InvalidData, e.to_string()))?;
        let config = record.to_config()?;

        let layers = config
            .layers
            .into_iter()
            .enumerate()
            .map(|(i, layer_config)| {
                Ok((
                    layer_config,
                    read_layer_tensors(&st, &format!("layers.{i}."))?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;

        NeuralNetwork::from_config_and_weights(config.input_shape, layers)
            .map_err(|e| Error::new(InvalidData, e.to_string()))
    }
}

/// Serializable blueprint of a network's architecture: its per-sample input shape and the stack
/// of weight-free layer configs — the mirror of a core [`NetworkConfig`].
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct NetworkConfigRecord {
    /// The per-sample input shape, sample axis excluded.
    pub input_shape: Vec<usize>,
    /// The layers in order, from input to output.
    pub layers: Vec<LayerConfigRecord>,
}

/// Serializable mirror of a [`LayerConfig`], one JSON object tagged by kind.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LayerConfigRecord {
    /// Mirror of [`LayerConfig::Dense`].
    Dense {
        /// The number of neurons, i.e. output features.
        neurons: usize,
        /// The name of the activation applied to the layer's output.
        activation: String,
    },
    /// Mirror of [`LayerConfig::Conv2d`].
    Conv2d {
        /// The number of filters, i.e. output channels.
        out_channels: usize,
        /// The kernel's spatial size `[height, width]`.
        kernel: [usize; 2],
        /// The stride applied on both spatial axes.
        stride: usize,
        /// The zero-padding added on both spatial axes.
        padding: usize,
        /// The name of the activation applied to the layer's output.
        activation: String,
    },
    /// Mirror of [`LayerConfig::Flatten`].
    Flatten,
}

impl NetworkConfigRecord {
    /// Saves this record to `path.json`.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<PathBuf> {
        json::save(self, path)
    }

    /// Loads a record from `path.json`.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        json::load(path)
    }

    /// Resolves this record into a core [`NetworkConfig`], looking each layer's activation up
    /// in the registry.
    pub fn to_config(&self) -> Result<NetworkConfig> {
        Ok(NetworkConfig {
            input_shape: self.input_shape.clone(),
            layers: self
                .layers
                .iter()
                .map(LayerConfigRecord::to_config)
                .collect::<Result<_>>()?,
        })
    }
}

impl From<&NetworkConfig> for NetworkConfigRecord {
    fn from(config: &NetworkConfig) -> Self {
        NetworkConfigRecord {
            input_shape: config.input_shape.clone(),
            layers: config.layers.iter().map(LayerConfigRecord::from).collect(),
        }
    }
}

impl From<&NeuralNetwork> for NetworkConfigRecord {
    fn from(network: &NeuralNetwork) -> Self {
        NetworkConfigRecord::from(&network.config())
    }
}

impl From<&LayerConfig> for LayerConfigRecord {
    fn from(config: &LayerConfig) -> Self {
        match config {
            LayerConfig::Dense {
                neurons,
                activation,
            } => LayerConfigRecord::Dense {
                neurons: *neurons,
                activation: activation.name().to_string(),
            },
            LayerConfig::Conv2d {
                out_channels,
                kernel,
                stride,
                padding,
                activation,
            } => LayerConfigRecord::Conv2d {
                out_channels: *out_channels,
                kernel: [kernel.0, kernel.1],
                stride: *stride,
                padding: *padding,
                activation: activation.name().to_string(),
            },
            LayerConfig::Flatten => LayerConfigRecord::Flatten,
        }
    }
}

impl LayerConfigRecord {
    /// Resolves this record into a core [`LayerConfig`], looking its activation up in the registry.
    fn to_config(&self) -> Result<LayerConfig> {
        Ok(match self {
            LayerConfigRecord::Dense {
                neurons,
                activation,
            } => LayerConfig::Dense {
                neurons: *neurons,
                activation: resolve_activation(activation)?,
            },
            LayerConfigRecord::Conv2d {
                out_channels,
                kernel,
                stride,
                padding,
                activation,
            } => LayerConfig::Conv2d {
                out_channels: *out_channels,
                kernel: (kernel[0], kernel[1]),
                stride: *stride,
                padding: *padding,
                activation: resolve_activation(activation)?,
            },
            LayerConfigRecord::Flatten => LayerConfig::Flatten,
        })
    }
}

/// Resolves an activation name against the registry.
fn resolve_activation(name: &str) -> Result<Arc<dyn Activation>> {
    ActivationProvider::get_by_name(name)
        .ok_or_else(|| Error::new(InvalidData, format!("unknown activation function: {name}")))
}

/// Reads one layer's tensors from the buffer, stripping the `{prefix}` from each name.
fn read_layer_tensors(st: &SafeTensors, prefix: &str) -> Result<Tensors> {
    let mut tensors = HashMap::new();
    for name in st.names() {
        if let Some(key) = name.strip_prefix(prefix) {
            let view = st
                .tensor(name)
                .map_err(|e| Error::new(InvalidData, e.to_string()))?;
            tensors.insert(key.to_string(), tensors::read_arrayd(&view)?);
        }
    }
    Ok(tensors.into())
}

#[cfg(test)]
mod tests {
    use crate::activations::{IDENTITY, RELU};
    use crate::model::{NetworkConfig, NeuralNetwork};
    use crate::testing::sample_batch;
    use std::path::{Path, PathBuf};

    fn temp_path(tag: &str) -> PathBuf {
        PathBuf::from(format!("target/nrn_test_{tag}_{}", std::process::id()))
    }

    fn cleanup(path: &Path) {
        let _ = std::fs::remove_file(path.with_extension("safetensors"));
    }

    #[test]
    fn save_weights_rejects_path_traversal() {
        // Writing through a path that escapes the working directory is refused by
        // the path guard in `tensors::save`; nothing is written.
        let config = NetworkConfig::builder(vec![2])
            .dense(2, &RELU)
            .dense(1, &IDENTITY)
            .build();
        let model = NeuralNetwork::from_config(config, 0).unwrap();

        assert!(model.save_weights("../../nrn_traversal_model").is_err());
    }

    #[test]
    fn layer_config_record_round_trips_every_kind() {
        use super::{LayerConfigRecord, resolve_activation};
        use crate::model::LayerConfig;

        // LayerConfig carries an Arc<dyn Activation> and has no PartialEq, so the round-trip is
        // checked at the record level: config -> record -> config -> record must be stable.
        let configs = [
            LayerConfig::Dense {
                neurons: 4,
                activation: resolve_activation("relu").unwrap(),
            },
            LayerConfig::Conv2d {
                out_channels: 2,
                kernel: (3, 3),
                stride: 1,
                padding: 0,
                activation: resolve_activation("relu").unwrap(),
            },
            LayerConfig::Flatten,
        ];

        for config in &configs {
            let record = LayerConfigRecord::from(config);
            let rebuilt = record.to_config().unwrap();
            assert_eq!(LayerConfigRecord::from(&rebuilt), record);
        }
    }

    #[test]
    fn layer_config_record_serializes_tagged_by_kind() {
        use super::LayerConfigRecord;

        let record = LayerConfigRecord::Conv2d {
            out_channels: 2,
            kernel: [3, 3],
            stride: 1,
            padding: 0,
            activation: "relu".to_string(),
        };
        let json = serde_json::to_string(&record).unwrap();
        assert_eq!(
            json,
            r#"{"kind":"conv2d","out_channels":2,"kernel":[3,3],"stride":1,"padding":0,"activation":"relu"}"#
        );
    }

    #[test]
    fn network_config_record_captures_architecture_of_a_network() {
        use super::{LayerConfigRecord, NetworkConfigRecord};

        let config = NetworkConfig::builder(vec![3])
            .dense(4, &RELU)
            .dense(3, &IDENTITY)
            .build();
        let model = NeuralNetwork::from_config(config, 0).unwrap();

        let record = NetworkConfigRecord::from(&model);

        assert_eq!(record.input_shape, vec![3]);
        assert_eq!(record.layers.len(), 2);
        assert_eq!(
            record.layers[0],
            LayerConfigRecord::Dense {
                neurons: 4,
                activation: "relu".to_string(),
            }
        );
    }

    #[test]
    fn to_config_rejects_unknown_activation() {
        use super::LayerConfigRecord;
        use std::io::ErrorKind::InvalidData;

        let record = LayerConfigRecord::Dense {
            neurons: 1,
            activation: "not_an_activation".to_string(),
        };
        let err = record.to_config().unwrap_err();
        assert_eq!(err.kind(), InvalidData);
        assert!(
            err.to_string().contains("unknown activation function"),
            "got: {err}"
        );
    }

    #[test]
    fn weights_save_load_roundtrip_predictions_are_identical() {
        use super::NetworkConfigRecord;

        let model = NeuralNetwork::hidden_dense_regression();
        let record = NetworkConfigRecord::from(&model);

        let inputs = sample_batch();
        let predictions_before = model.output(inputs.view());

        let path = temp_path("weights_model");
        model.save_weights(&path).unwrap();

        // The weights file carries no architecture metadata: it is tensors only.
        let bytes = crate::io::model::tensors::load(&path).unwrap();
        let metadata = crate::io::model::tensors::read_metadata(&bytes).unwrap();
        assert!(metadata.is_empty(), "weights file must hold no metadata");

        let loaded = NeuralNetwork::load_weights(&path, &record).unwrap();
        let predictions_after = loaded.output(inputs.view());

        cleanup(&path);

        assert_eq!(predictions_before, predictions_after);
    }

    #[test]
    fn cnn_weights_save_load_roundtrip_predictions_are_identical() {
        use super::NetworkConfigRecord;
        use crate::activations::SIGMOID;
        use ndarray::{Array, IxDyn};

        // Conv2d → Flatten → Dense (sigmoid head), exercising every layer kind through the
        // weights-only path.
        let config = NetworkConfig::builder(vec![1, 4, 4])
            .conv2d(2, (3, 3), 1, 0, &RELU)
            .flatten()
            .dense(1, &SIGMOID)
            .build();
        let model = NeuralNetwork::from_config(config, 3).unwrap();
        let record = NetworkConfigRecord::from(&model);

        let inputs = Array::from_shape_fn(IxDyn(&[1, 4, 4, 5]), |d| {
            (d[1] * 4 + d[2]) as f32 * 0.1 + d[3] as f32
        });
        let predictions_before = model.output(inputs.view());

        let path = temp_path("weights_cnn_model");
        model.save_weights(&path).unwrap();
        let loaded = NeuralNetwork::load_weights(&path, &record).unwrap();
        let predictions_after = loaded.output(inputs.view());

        cleanup(&path);

        assert_eq!(predictions_before, predictions_after);
    }
}
