use crate::activations::ActivationProvider;
use crate::io::tensors::{self, F32Tensor};
use crate::model::{NeuralNetwork, NeuronLayer};
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
        metadata.insert("n_layers".to_string(), self.layers.len().to_string());

        for (i, layer) in self.layers.iter().enumerate() {
            metadata.insert(
                format!("layer{i}.activation"),
                layer.activation.name().to_string(),
            );
            entries.push((format!("layer{i}.weights"), tensors::tensor(&layer.weights)));
            entries.push((format!("layer{i}.biases"), tensors::tensor(&layer.biases)));
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

        let mut layers = Vec::with_capacity(n_layers);

        for i in 0..n_layers {
            let weights = tensors::read_array2(&format!("layer{i}.weights"), st)?;
            let biases = tensors::read_array1(&format!("layer{i}.biases"), st)?;

            let activation_name = tensors::meta(metadata, &format!("layer{i}.activation"))?;
            let activation = ActivationProvider::get_by_name(activation_name).ok_or_else(|| {
                Error::new(
                    InvalidData,
                    format!("Unknown activation function: {activation_name}"),
                )
            })?;

            layers.push(NeuronLayer {
                weights,
                biases,
                activation,
            });
        }

        Ok(NeuralNetwork { layers })
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
        let model = NeuralNetwork::initialization(3, &specs);

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
        metadata.insert(
            "layer0.activation".to_string(),
            "not_an_activation".to_string(),
        );
        tensors::save(&path, entries, metadata).unwrap();

        let result = NeuralNetwork::load(&path);
        cleanup(&path);

        let message = result.err().map(|e| e.to_string()).unwrap_or_default();
        assert!(message.contains("Unknown activation"), "got: {message}");
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
}
