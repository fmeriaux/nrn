//! Directory serializer for a [`Predictor`]: the network weights as `model.safetensors`, the
//! model blueprint as `config.json`, and, when present, the scaler as a `preprocessor.json` sidecar.

use crate::io::json;
use crate::io::scalers::ScalerRecord;
use crate::io::task::TaskRecord;
use crate::model::{NeuralNetwork, Predictor};
use serde::{Deserialize, Serialize};
use std::io::Result;
use std::path::{Path, PathBuf};

/// File stem of the network weights inside a predictor directory.
const MODEL_STEM: &str = "model";
/// File stem of the model blueprint inside a predictor directory.
const CONFIG_STEM: &str = "config";
/// File stem of the preprocessor sidecar inside a predictor directory.
const PREPROCESSOR_STEM: &str = "preprocessor";

/// Serializable blueprint of a predictor's model: the task its logits are read for.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct PredictorConfig {
    /// The learning task the network was trained for.
    pub task: TaskRecord,
}

impl PredictorConfig {
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<PathBuf> {
        json::save(self, path)
    }

    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        json::load(path)
    }
}

impl Predictor {
    /// Saves the predictor as a directory: `dir/model.safetensors` for the network,
    /// `dir/config.json` for the task, and, when a scaler is present, `dir/preprocessor.json`
    /// beside them. Returns the directory path.
    pub fn save<P: AsRef<Path>>(&self, dir: P) -> Result<PathBuf> {
        let dir = dir.as_ref();
        self.network.save(dir.join(MODEL_STEM))?;
        PredictorConfig {
            task: self.task.into(),
        }
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
        let network = NeuralNetwork::load(dir.join(MODEL_STEM))?;
        let task = PredictorConfig::load(dir.join(CONFIG_STEM))?.task.into();

        let scaler = if dir.join(PREPROCESSOR_STEM).with_extension("json").exists() {
            Some(ScalerRecord::load(dir.join(PREPROCESSOR_STEM))?.into())
        } else {
            None
        };

        Ok(Predictor {
            network,
            task,
            scaler,
        })
    }
}
