//! Persistence for a [`NeuralNetwork`]: its weights as a `.safetensors` file and its
//! architecture as a serializable [`NetworkConfig`] beside them.
//! [`save_weights`](NeuralNetwork::save_weights) writes tensors only; the architecture lives in
//! the `NetworkConfig`, from which [`load_weights`](NeuralNetwork::load_weights) rebuilds it.

use crate::activations::{Activation, ActivationProvider};
use crate::io::json;
use crate::io::model::tensors;
use crate::model::{LayerSpec, NeuralNetwork};
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
    /// `layers.{i}.{tensor}`; the architecture lives in a [`NetworkConfig`] beside them.
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
    /// [`save_weights`](NeuralNetwork::save_weights) and the [`NetworkConfig`] describing its
    /// architecture.
    /// # Arguments
    /// - `path`: The path (without extension) to read the weights from.
    /// - `config`: The architecture the weights belong to.
    pub fn load_weights<P: AsRef<Path>>(path: P, config: &NetworkConfig) -> Result<Self> {
        let bytes = tensors::load(path)?;
        let st =
            SafeTensors::deserialize(&bytes).map_err(|e| Error::new(InvalidData, e.to_string()))?;

        let layers = config
            .specs()?
            .into_iter()
            .enumerate()
            .map(|(i, spec)| Ok((spec, read_layer_tensors(&st, &format!("layers.{i}."))?)))
            .collect::<Result<Vec<_>>>()?;

        NeuralNetwork::from_specs_and_weights(config.input_shape.clone(), layers)
            .map_err(|e| Error::new(InvalidData, e.to_string()))
    }
}

/// Serializable blueprint of a network's architecture: its per-sample input shape and the stack
/// of weight-free layer specs — the typed counterpart to the tensors in the weights file.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct NetworkConfig {
    /// The per-sample input shape, sample axis excluded.
    pub input_shape: Vec<usize>,
    /// The layers in order, from input to output.
    pub layers: Vec<LayerSpecRecord>,
}

/// Serializable mirror of a [`LayerSpec`], one JSON object tagged by kind.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LayerSpecRecord {
    /// Mirror of [`LayerSpec::Dense`].
    Dense {
        /// The number of neurons, i.e. output features.
        neurons: usize,
        /// The name of the activation applied to the layer's output.
        activation: String,
    },
    /// Mirror of [`LayerSpec::Conv2d`].
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
    /// Mirror of [`LayerSpec::Flatten`].
    Flatten,
}

impl NetworkConfig {
    /// Saves this config to `path.json`.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<PathBuf> {
        json::save(self, path)
    }

    /// Loads a config from `path.json`.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        json::load(path)
    }

    /// The core [`LayerSpec`]s this config describes, in order, resolving each activation name
    /// against the registry.
    fn specs(&self) -> Result<Vec<LayerSpec>> {
        self.layers.iter().map(LayerSpecRecord::to_spec).collect()
    }
}

impl From<&NeuralNetwork> for NetworkConfig {
    fn from(network: &NeuralNetwork) -> Self {
        NetworkConfig {
            input_shape: network.input_shape(),
            layers: network.specs().iter().map(LayerSpecRecord::from).collect(),
        }
    }
}

impl From<&LayerSpec> for LayerSpecRecord {
    fn from(spec: &LayerSpec) -> Self {
        match spec {
            LayerSpec::Dense {
                neurons,
                activation,
            } => LayerSpecRecord::Dense {
                neurons: *neurons,
                activation: activation.name().to_string(),
            },
            LayerSpec::Conv2d {
                out_channels,
                kernel,
                stride,
                padding,
                activation,
            } => LayerSpecRecord::Conv2d {
                out_channels: *out_channels,
                kernel: [kernel.0, kernel.1],
                stride: *stride,
                padding: *padding,
                activation: activation.name().to_string(),
            },
            LayerSpec::Flatten => LayerSpecRecord::Flatten,
        }
    }
}

impl LayerSpecRecord {
    /// Resolves this record into a core [`LayerSpec`], looking its activation up in the registry.
    fn to_spec(&self) -> Result<LayerSpec> {
        Ok(match self {
            LayerSpecRecord::Dense {
                neurons,
                activation,
            } => LayerSpec::Dense {
                neurons: *neurons,
                activation: resolve_activation(activation)?,
            },
            LayerSpecRecord::Conv2d {
                out_channels,
                kernel,
                stride,
                padding,
                activation,
            } => LayerSpec::Conv2d {
                out_channels: *out_channels,
                kernel: (kernel[0], kernel[1]),
                stride: *stride,
                padding: *padding,
                activation: resolve_activation(activation)?,
            },
            LayerSpecRecord::Flatten => LayerSpec::Flatten,
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
    use crate::activations::RELU;
    use crate::model::{NeuralNetwork, NeuronLayerSpec};
    use ndarray::Array2;
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
        let specs = NeuronLayerSpec::network_for(vec![2], &*RELU, 2);
        let model = NeuralNetwork::initialization(2, &specs, 0);

        assert!(model.save_weights("../../nrn_traversal_model").is_err());
    }

    #[test]
    fn layer_spec_record_round_trips_every_kind() {
        use super::{LayerSpecRecord, resolve_activation};
        use crate::model::LayerSpec;

        // LayerSpec carries an Arc<dyn Activation> and has no PartialEq, so the round-trip is
        // checked at the record level: spec -> record -> spec -> record must be stable.
        let specs = [
            LayerSpec::Dense {
                neurons: 4,
                activation: resolve_activation("relu").unwrap(),
            },
            LayerSpec::Conv2d {
                out_channels: 2,
                kernel: (3, 3),
                stride: 1,
                padding: 0,
                activation: resolve_activation("relu").unwrap(),
            },
            LayerSpec::Flatten,
        ];

        for spec in &specs {
            let record = LayerSpecRecord::from(spec);
            let rebuilt = record.to_spec().unwrap();
            assert_eq!(LayerSpecRecord::from(&rebuilt), record);
        }
    }

    #[test]
    fn layer_spec_record_serializes_tagged_by_kind() {
        use super::LayerSpecRecord;

        let record = LayerSpecRecord::Conv2d {
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
    fn network_config_captures_architecture_of_a_network() {
        use super::{LayerSpecRecord, NetworkConfig};

        let specs = NeuronLayerSpec::network_for(vec![4], &*RELU, 3);
        let model = NeuralNetwork::initialization(3, &specs, 0);

        let config = NetworkConfig::from(&model);

        assert_eq!(config.input_shape, vec![3]);
        assert_eq!(config.layers.len(), 2);
        assert_eq!(
            config.layers[0],
            LayerSpecRecord::Dense {
                neurons: 4,
                activation: "relu".to_string(),
            }
        );
    }

    #[test]
    fn to_spec_rejects_unknown_activation() {
        use super::LayerSpecRecord;
        use std::io::ErrorKind::InvalidData;

        let record = LayerSpecRecord::Dense {
            neurons: 1,
            activation: "not_an_activation".to_string(),
        };
        let err = record.to_spec().unwrap_err();
        assert_eq!(err.kind(), InvalidData);
        assert!(
            err.to_string().contains("unknown activation function"),
            "got: {err}"
        );
    }

    #[test]
    fn weights_save_load_roundtrip_predictions_are_identical() {
        use super::NetworkConfig;

        let specs = NeuronLayerSpec::network_for(vec![4], &*RELU, 2);
        let model = NeuralNetwork::initialization(3, &specs, 0);
        let config = NetworkConfig::from(&model);

        let inputs = Array2::from_shape_fn((3, 5), |(i, j)| (i * 5 + j) as f32 * 0.1);
        let predictions_before = model.output(inputs.view());

        let path = temp_path("weights_model");
        model.save_weights(&path).unwrap();

        // The weights file carries no architecture metadata: it is tensors only.
        let bytes = crate::io::model::tensors::load(&path).unwrap();
        let metadata = crate::io::model::tensors::read_metadata(&bytes).unwrap();
        assert!(metadata.is_empty(), "weights file must hold no metadata");

        let loaded = NeuralNetwork::load_weights(&path, &config).unwrap();
        let predictions_after = loaded.output(inputs.view());

        cleanup(&path);

        assert_eq!(predictions_before, predictions_after);
    }

    #[test]
    fn cnn_weights_save_load_roundtrip_predictions_are_identical() {
        use super::NetworkConfig;
        use crate::activations::SIGMOID;
        use crate::layers::{Conv2d, Dense, Flatten, Layer};
        use ndarray::{Array, IxDyn};
        use ndarray_rand::rand::SeedableRng;
        use ndarray_rand::rand::rngs::StdRng;

        // Conv2d → Flatten → Dense, exercising every layer kind through the weights-only path.
        let mut rng = StdRng::seed_from_u64(3);
        let conv = Conv2d::initialization((1, 4, 4), 2, (3, 3), 1, 0, RELU.clone(), &mut rng);
        let flatten = Flatten::new(conv.output_shape());
        let dense = Dense::initialization(
            flatten.output_size(),
            &NeuronLayerSpec {
                neurons: 1,
                activation: SIGMOID.clone(),
            },
            &mut rng,
        );
        let model = NeuralNetwork::single(conv)
            .with_layer(flatten)
            .with_layer(dense);
        let config = NetworkConfig::from(&model);

        let inputs = Array::from_shape_fn(IxDyn(&[1, 4, 4, 5]), |d| {
            (d[1] * 4 + d[2]) as f32 * 0.1 + d[3] as f32
        });
        let predictions_before = model.output(inputs.view());

        let path = temp_path("weights_cnn_model");
        model.save_weights(&path).unwrap();
        let loaded = NeuralNetwork::load_weights(&path, &config).unwrap();
        let predictions_after = loaded.output(inputs.view());

        cleanup(&path);

        assert_eq!(predictions_before, predictions_after);
    }
}
