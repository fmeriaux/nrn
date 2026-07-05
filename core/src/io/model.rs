use crate::io::tensors::{self, F32Tensor};
use crate::layers::{Layer, LayerKind};
use crate::model::NeuralNetwork;
use ndarray::ArrayD;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::io::ErrorKind::InvalidData;
use std::io::{Error, Result};
use std::path::{Path, PathBuf};

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

/// Reads one layer's tensors from the buffer, stripping the `layer{i}.` prefix from each name.
fn read_layer_tensors(st: &SafeTensors, prefix: &str) -> Result<HashMap<String, ArrayD<f32>>> {
    let mut tensors = HashMap::new();
    for name in st.names() {
        if let Some(key) = name.strip_prefix(prefix) {
            let view = st
                .tensor(name)
                .map_err(|e| Error::new(InvalidData, e.to_string()))?;
            tensors.insert(key.to_string(), tensors::read_arrayd(&view)?);
        }
    }
    Ok(tensors)
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
        let predictions_before = model.predict(inputs.view());

        let path = temp_path("model");
        model.save(&path).unwrap();

        let loaded = NeuralNetwork::load(&path).unwrap();
        let predictions_after = loaded.predict(inputs.view());

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
        let predictions_before = model.predict(inputs.view());

        let path = temp_path("cnn_model");
        model.save(&path).unwrap();

        let loaded = NeuralNetwork::load(&path).unwrap();
        let predictions_after = loaded.predict(inputs.view());

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
                "layer0.weights".to_string(),
                tensors::tensor(&array![[1.0_f32]]),
            ),
            (
                "layer0.biases".to_string(),
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
                "layer0.weights".to_string(),
                tensors::tensor(&array![[1.0_f32, 2.0]]),
            )
        };
        let biases = || {
            (
                "layer0.biases".to_string(),
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
                        "layer0.weights".to_string(),
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
            vec![("layer0.weights".to_string(), U8Tensor)],
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
}
