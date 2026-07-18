//! Directory serializer for a [`Predictor`]: the network weights as `model.safetensors`, the
//! model blueprint as `config.json`, and, when present, the scaler as a `preprocessor.json` sidecar.

use crate::io::model::config::{CONFIG_STEM, ModelConfigRecord, PREPROCESSOR_STEM};
use crate::io::model::network::NetworkConfigRecord;
use crate::io::model::scalers::ScalerRecord;
use crate::io::path::PathExt;
use crate::model::{NeuralNetwork, Predictor};
use std::io::Result;
use std::path::{Path, PathBuf};

/// File stem of the network weights inside a predictor directory.
const MODEL_STEM: &str = "model";

impl Predictor {
    /// Saves the predictor as a directory: `dir/model.safetensors` for the network weights,
    /// `dir/config.json` for its architecture and task, and, when a scaler is present,
    /// `dir/preprocessor.json` beside them. Returns the directory path.
    pub fn save<P: AsRef<Path>>(&self, dir: P) -> Result<PathBuf> {
        let dir = dir.as_ref();
        self.network.save_weights(dir.join(MODEL_STEM))?;
        ModelConfigRecord::from_parts(NetworkConfigRecord::from(&self.network), &self.config)
            .save(dir.join(CONFIG_STEM))?;
        if let Some(scaler) = &self.scaler {
            ScalerRecord::from(scaler.clone()).save(dir.join(PREPROCESSOR_STEM))?;
        }
        Ok(dir.to_path_buf())
    }

    /// Loads a predictor from a directory written by [`Predictor::save`]. The scaler is read only
    /// when a `preprocessor.json` sidecar exists.
    pub fn load<P: AsRef<Path>>(dir: P) -> Result<Self> {
        let dir = dir.as_ref();
        let record = ModelConfigRecord::load(dir.join(CONFIG_STEM))?;
        let network = NeuralNetwork::load_weights(dir.join(MODEL_STEM), &record.network)?;
        let config = record.to_model_config()?;

        let scaler = dir
            .join(PREPROCESSOR_STEM)
            .optional_sidecar("json")
            .map(ScalerRecord::load)
            .transpose()?
            .map(Into::into);

        Ok(Predictor {
            network,
            config,
            scaler,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::MODEL_STEM;
    use crate::activations::{IDENTITY, RELU};
    use crate::data::scalers::{MinMaxScaler, ScalerMethod};
    use crate::io::model::config::{CONFIG_STEM, ModelConfigRecord};
    use crate::model::{Labels, ModelConfig, NetworkConfig, NeuralNetwork, Predictor};
    use crate::task::Task;
    use ndarray::{Array2, array};

    fn temp_dir(tag: &str) -> std::path::PathBuf {
        std::path::PathBuf::from(format!("target/nrn_predictor_{tag}_{}", std::process::id()))
    }

    fn model_and_inputs() -> (NeuralNetwork, Array2<f32>) {
        let config = NetworkConfig::builder(vec![3])
            .dense(4, &RELU)
            .dense(1, &IDENTITY)
            .build();
        let model = NeuralNetwork::from_config(config, 0).unwrap();
        let inputs = Array2::from_shape_fn((3, 5), |(i, j)| (i * 5 + j) as f32 * 0.1);
        (model, inputs)
    }

    #[test]
    fn save_load_roundtrip_outputs_are_identical() {
        let (network, inputs) = model_and_inputs();
        let predictor = Predictor::new(network, ModelConfig::unlabeled(Task::Binary), None);
        let outputs_before = predictor.output(inputs.view()).unwrap();

        let dir = temp_dir("plain");
        predictor.save(&dir).unwrap();
        let loaded = Predictor::load(&dir).unwrap();
        let outputs_after = loaded.output(inputs.view()).unwrap();

        std::fs::remove_dir_all(&dir).ok();

        assert!(loaded.scaler.is_none());
        assert_eq!(outputs_before, outputs_after);
    }

    #[test]
    fn save_load_roundtrip_preserves_the_scaler() {
        let (network, inputs) = model_and_inputs();
        let scaler = ScalerMethod::MinMax(MinMaxScaler {
            min: array![0.0, 0.0, 0.0],
            max: array![1.0, 2.0, 0.5],
            range: (0.0, 1.0),
        });
        let predictor = Predictor::new(network, ModelConfig::unlabeled(Task::Binary), Some(scaler));
        let outputs_before = predictor.output(inputs.view()).unwrap();

        let dir = temp_dir("scaled");
        predictor.save(&dir).unwrap();
        let loaded = Predictor::load(&dir).unwrap();
        let outputs_after = loaded.output(inputs.view()).unwrap();

        std::fs::remove_dir_all(&dir).ok();

        assert!(loaded.scaler.is_some());
        assert_eq!(outputs_before, outputs_after);
    }

    #[test]
    fn save_load_roundtrip_preserves_labels() {
        let (network, _) = model_and_inputs();
        let config = ModelConfig::new(
            Task::Binary,
            Some(Labels::new(vec!["cat".to_string(), "dog".to_string()])),
        )
        .unwrap();
        let predictor = Predictor::new(network, config, None);

        let dir = temp_dir("labels");
        predictor.save(&dir).unwrap();
        let loaded = Predictor::load(&dir).unwrap();

        std::fs::remove_dir_all(&dir).ok();

        assert_eq!(
            loaded.config.labels().map(Labels::names),
            Some(["cat".to_string(), "dog".to_string()].as_slice())
        );
    }

    #[test]
    fn config_carries_the_network_architecture() {
        use crate::io::model::network::LayerConfigRecord;

        let (network, _) = model_and_inputs();
        let predictor = Predictor::new(network, ModelConfig::unlabeled(Task::Binary), None);

        let dir = temp_dir("config");
        predictor.save(&dir).unwrap();
        let config = ModelConfigRecord::load(dir.join(CONFIG_STEM)).unwrap();

        std::fs::remove_dir_all(&dir).ok();

        assert_eq!(config.network.input_shape, vec![3]);
        assert_eq!(
            config.network.layers[0],
            LayerConfigRecord::Dense {
                neurons: 4,
                activation: "relu".to_string(),
            }
        );
    }

    #[test]
    fn model_file_holds_weights_only() {
        let (network, _) = model_and_inputs();
        let predictor = Predictor::new(network, ModelConfig::unlabeled(Task::Binary), None);

        let dir = temp_dir("weights_only");
        predictor.save(&dir).unwrap();

        let bytes = crate::io::model::tensors::load(dir.join(MODEL_STEM)).unwrap();
        let metadata = crate::io::model::tensors::read_metadata(&bytes).unwrap();

        std::fs::remove_dir_all(&dir).ok();

        assert!(metadata.is_empty(), "model file must carry no metadata");
    }
}
