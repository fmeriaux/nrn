//! Persistence for a [`NeuralNetwork`]: its weights as a `.safetensors` file and its
//! architecture as a serializable [`NetworkConfig`].
//! [`save_weights`](NeuralNetwork::save_weights) writes tensors only, the architecture living in
//! a `NetworkConfig` beside them; the older self-describing
//! [`save`](NeuralNetwork::save)/[`load`](NeuralNetwork::load) pair embeds it in the file's
//! metadata instead.

use crate::activations::{Activation, ActivationProvider};
use crate::io::json;
use crate::io::tensors::{self, F32Tensor};
use crate::layers::{Layer, LayerConfigError, LayerKind};
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
    /// Appends this network's tensors and metadata to the provided collections.
    fn collect_tensors(
        &self,
        entries: &mut Vec<(String, F32Tensor)>,
        metadata: &mut HashMap<String, String>,
    ) {
        metadata.insert("n_layers".to_string(), self.layers().len().to_string());

        for (i, layer) in self.layers().iter().enumerate() {
            metadata.insert(format!("layer{i}.kind"), layer.kind().as_str().to_string());
            for (key, value) in layer.config() {
                metadata.insert(format!("layer{i}.{key}"), value);
            }
            for (name, tensor) in layer.named_tensors() {
                entries.push((format!("layer{i}.{name}"), tensors::tensor(&tensor)));
            }
        }
    }

    /// Rebuilds a network from a deserialized safetensors buffer and its metadata map.
    fn from_tensors(st: &SafeTensors, metadata: &HashMap<String, String>) -> Result<Self> {
        let n_layers: usize = tensors::meta(metadata, "n_layers")?
            .parse()
            .map_err(|e| Error::new(InvalidData, format!("invalid layer count: {e}")))?;

        if n_layers == 0 {
            return Err(Error::new(InvalidData, "model has no layers"));
        }

        let mut layers: Vec<Box<dyn Layer>> = Vec::with_capacity(n_layers);

        for i in 0..n_layers {
            let prefix = format!("layer{i}.");
            let kind = LayerKind::try_from(tensors::meta(metadata, &format!("{prefix}kind"))?)
                .map_err(|e| Error::new(InvalidData, e.to_string()))?;
            let config = layer_config(metadata, &prefix);
            let tensors = read_layer_tensors(st, &prefix)?;
            let layer = kind
                .instantiate(&config, tensors)
                .map_err(|e| Error::new(InvalidData, e.to_string()))?;
            layers.push(layer);
        }

        Ok(NeuralNetwork::new(layers))
    }

    /// Saves the neural network to a `.safetensors` file.
    /// # Arguments
    /// - `path`: The path to the file where the network will be saved.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<PathBuf> {
        let mut entries = Vec::new();
        let mut metadata = HashMap::new();
        self.collect_tensors(&mut entries, &mut metadata);
        tensors::save(path, entries, metadata)
    }

    /// Loads a neural network from a `.safetensors` file.
    /// # Arguments
    /// - `path`: The path to the file to load the network from.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let bytes = tensors::load(path)?;
        let st =
            SafeTensors::deserialize(&bytes).map_err(|e| Error::new(InvalidData, e.to_string()))?;
        let metadata = tensors::read_metadata(&bytes)?;
        NeuralNetwork::from_tensors(&st, &metadata)
    }

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
                layer
                    .named_tensors()
                    .into_iter()
                    .map(move |(name, tensor)| {
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
        self.layers
            .iter()
            .map(|record| {
                record
                    .to_spec()
                    .map_err(|e| Error::new(InvalidData, e.to_string()))
            })
            .collect()
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
    fn to_spec(&self) -> std::result::Result<LayerSpec, LayerConfigError> {
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

/// Resolves an activation name against the registry, mirroring the layer config path.
fn resolve_activation(name: &str) -> std::result::Result<Arc<dyn Activation>, LayerConfigError> {
    ActivationProvider::get_by_name(name)
        .ok_or_else(|| LayerConfigError::UnknownActivation(name.to_string()))
}

/// Collects one layer's configuration from the metadata map, stripping the `layer{i}.`
/// prefix from each key.
fn layer_config(metadata: &HashMap<String, String>, prefix: &str) -> HashMap<String, String> {
    metadata
        .iter()
        .filter_map(|(key, value)| {
            key.strip_prefix(prefix)
                .map(|k| (k.to_string(), value.clone()))
        })
        .collect()
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
    fn save_load_roundtrip_predictions_are_identical() {
        let specs = NeuronLayerSpec::network_for(vec![4], &*RELU, 2);
        let model = NeuralNetwork::initialization(3, &specs, 0);

        let inputs = Array2::from_shape_fn((3, 5), |(i, j)| (i * 5 + j) as f32 * 0.1);
        let predictions_before = model.output(inputs.view());

        let path = temp_path("model");
        model.save(&path).unwrap();

        let loaded = NeuralNetwork::load(&path).unwrap();
        let predictions_after = loaded.output(inputs.view());

        cleanup(&path);

        assert_eq!(predictions_before, predictions_after);
    }

    #[test]
    fn cnn_save_load_roundtrip_predictions_are_identical() {
        use crate::activations::{RELU, SIGMOID};
        use crate::layers::{Conv2d, Dense, Flatten, Layer};
        use ndarray::{Array, IxDyn};
        use ndarray_rand::rand::SeedableRng;
        use ndarray_rand::rand::rngs::StdRng;

        // A full CNN: Conv2d → Flatten → Dense, exercising every layer kind through io.
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

        let inputs = Array::from_shape_fn(IxDyn(&[1, 4, 4, 5]), |d| {
            (d[1] * 4 + d[2]) as f32 * 0.1 + d[3] as f32
        });
        let predictions_before = model.output(inputs.view());

        let path = temp_path("cnn_model");
        model.save(&path).unwrap();

        let loaded = NeuralNetwork::load(&path).unwrap();
        let predictions_after = loaded.output(inputs.view());

        cleanup(&path);

        assert_eq!(predictions_before, predictions_after);
    }

    #[test]
    fn load_rejects_unknown_activation() {
        // Hand-write a model file whose activation name is not registered.
        use crate::io::tensors;
        use ndarray::{Array1, array};
        use std::collections::HashMap;

        let path = temp_path("unknown_activation");
        let entries = vec![
            (
                "layer0.weight".to_string(),
                tensors::tensor(&array![[1.0_f32]]),
            ),
            (
                "layer0.bias".to_string(),
                tensors::tensor(&Array1::<f32>::zeros(1)),
            ),
        ];
        let mut metadata = HashMap::new();
        metadata.insert("n_layers".to_string(), "1".to_string());
        metadata.insert("layer0.kind".to_string(), "dense".to_string());
        metadata.insert(
            "layer0.activation".to_string(),
            "not_an_activation".to_string(),
        );
        tensors::save(&path, entries, metadata).unwrap();

        let result = NeuralNetwork::load(&path);
        cleanup(&path);

        let message = result.err().map(|e| e.to_string()).unwrap_or_default();
        assert!(message.contains("unknown activation"), "got: {message}");
    }

    #[test]
    fn load_rejects_zero_layers() {
        use crate::io::tensors;
        use ndarray::array;
        use std::collections::HashMap;

        let path = temp_path("zero_layers");
        let mut metadata = HashMap::new();
        metadata.insert("n_layers".to_string(), "0".to_string());
        let entries = vec![("placeholder".to_string(), tensors::tensor(&array![0.0_f32]))];
        tensors::save(&path, entries, metadata).unwrap();

        let result = NeuralNetwork::load(&path);
        cleanup(&path);

        let message = result.err().map(|e| e.to_string()).unwrap_or_default();
        assert!(message.contains("no layers"), "got: {message}");
    }

    #[test]
    fn load_rejects_non_numeric_layer_count() {
        use crate::io::tensors;
        use ndarray::array;
        use std::collections::HashMap;

        let path = temp_path("bad_n_layers");
        let mut metadata = HashMap::new();
        metadata.insert("n_layers".to_string(), "abc".to_string());
        let entries = vec![("placeholder".to_string(), tensors::tensor(&array![0.0_f32]))];
        tensors::save(&path, entries, metadata).unwrap();

        let result = NeuralNetwork::load(&path);
        cleanup(&path);

        let message = result.err().map(|e| e.to_string()).unwrap_or_default();
        assert!(message.contains("invalid layer count"), "got: {message}");
    }

    #[test]
    fn load_rejects_corrupt_file() {
        let path = temp_path("corrupt").with_extension("safetensors");
        path.parent().map(std::fs::create_dir_all);
        std::fs::write(&path, b"this is not a safetensors buffer").unwrap();

        let result = NeuralNetwork::load(path.with_extension(""));
        let _ = std::fs::remove_file(&path);

        assert!(result.is_err());
    }

    #[test]
    fn load_rejects_truncated_or_malformed_files() {
        use crate::io::tensors::{self, F32Tensor};
        use ndarray::{Array1, array};
        use std::collections::HashMap;

        // A well-formed layer 0 (rank-2 weights, rank-1 biases). The activation
        // value is irrelevant here: every case below fails *before* the activation
        // is looked up, so any string serves.
        let weights = || {
            (
                "layer0.weight".to_string(),
                tensors::tensor(&array![[1.0_f32, 2.0]]),
            )
        };
        let biases = || {
            (
                "layer0.bias".to_string(),
                tensors::tensor(&Array1::<f32>::zeros(1)),
            )
        };
        let activation = || ("layer0.activation".to_string(), "relu".to_string());
        let kind = || ("layer0.kind".to_string(), "dense".to_string());
        let one_layer = || {
            HashMap::from([
                ("n_layers".to_string(), "1".to_string()),
                kind(),
                activation(),
            ])
        };

        // Each case is well-formed at the safetensors level but violates the model
        // schema, so `load` must reject it — exercising the read guards that
        // `model::load` delegates to `tensors` (missing key / missing tensor /
        // wrong rank).
        type Case = (
            &'static str,
            Vec<(String, F32Tensor)>,
            HashMap<String, String>,
        );
        let cases: Vec<Case> = vec![
            (
                "n_layers metadata absent",
                vec![weights(), biases()],
                HashMap::from([activation()]),
            ),
            ("weights tensor absent", vec![biases()], one_layer()),
            ("biases tensor absent", vec![weights()], one_layer()),
            (
                "activation metadata absent",
                vec![weights(), biases()],
                HashMap::from([("n_layers".to_string(), "1".to_string()), kind()]),
            ),
            (
                "weights stored with the wrong rank",
                vec![
                    (
                        "layer0.weight".to_string(),
                        tensors::tensor(&array![1.0_f32, 2.0]),
                    ),
                    biases(),
                ],
                one_layer(),
            ),
        ];

        for (i, (label, entries, metadata)) in cases.into_iter().enumerate() {
            let path = temp_path(&format!("malformed_{i}"));
            tensors::save(&path, entries, metadata).unwrap();
            let result = NeuralNetwork::load(&path);
            cleanup(&path);
            assert!(result.is_err(), "{label} should be rejected");
        }
    }

    #[test]
    fn load_rejects_non_f32_weights() {
        use safetensors::{Dtype, View, serialize};
        use std::borrow::Cow;
        use std::collections::HashMap;

        // A weights tensor whose dtype is not f32 must be refused by the read
        // guard `model::load` delegates to `tensors::read_f32`.
        struct U8Tensor;
        impl View for U8Tensor {
            fn dtype(&self) -> Dtype {
                Dtype::U8
            }
            fn shape(&self) -> &[usize] {
                &[1, 1]
            }
            fn data(&self) -> Cow<'_, [u8]> {
                Cow::Owned(vec![0])
            }
            fn data_len(&self) -> usize {
                1
            }
        }

        let metadata = HashMap::from([
            ("n_layers".to_string(), "1".to_string()),
            ("layer0.kind".to_string(), "dense".to_string()),
            ("layer0.activation".to_string(), "relu".to_string()),
        ]);
        let bytes = serialize(
            vec![("layer0.weight".to_string(), U8Tensor)],
            Some(metadata),
        )
        .unwrap();

        let path = temp_path("non_f32_weights").with_extension("safetensors");
        path.parent().map(std::fs::create_dir_all);
        std::fs::write(&path, bytes).unwrap();
        let result = NeuralNetwork::load(path.with_extension(""));
        let _ = std::fs::remove_file(&path);

        assert!(result.is_err());
    }

    #[test]
    fn save_rejects_path_traversal() {
        // Saving through a path that escapes the working directory is refused by
        // the path guard in `tensors::save`; nothing is written.
        let specs = NeuronLayerSpec::network_for(vec![2], &*RELU, 2);
        let model = NeuralNetwork::initialization(2, &specs, 0);

        assert!(model.save("../../nrn_traversal_model").is_err());
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
        use crate::layers::LayerConfigError;

        let record = LayerSpecRecord::Dense {
            neurons: 1,
            activation: "not_an_activation".to_string(),
        };
        let err = record.to_spec().unwrap_err();
        assert_eq!(
            err,
            LayerConfigError::UnknownActivation("not_an_activation".to_string())
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
        let bytes = crate::io::tensors::load(&path).unwrap();
        let metadata = crate::io::tensors::read_metadata(&bytes).unwrap();
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
