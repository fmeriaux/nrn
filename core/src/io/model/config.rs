//! The `config.json` blueprint: network architecture plus the task its logits are read for and
//! the labels naming its classes or multi-label positions, when known.

use crate::io::json;
use crate::io::model::network::NetworkConfigRecord;
use crate::io::model::task::TaskRecord;
use crate::model::{Labels, ModelConfig};
use crate::task::Task;
use serde::{Deserialize, Serialize};
use std::io::ErrorKind::InvalidData;
use std::io::{Error, Result};
use std::path::{Path, PathBuf};

/// File stem of the blueprint inside a predictor or training run directory.
pub(crate) const CONFIG_STEM: &str = "config";
/// File stem of the preprocessor sidecar inside a predictor or training run directory.
pub(crate) const PREPROCESSOR_STEM: &str = "preprocessor";

/// Serializable blueprint of a model: the network architecture its weights belong to, the task
/// its logits are read for, and the labels naming its classes or multi-label positions, when known.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct ModelConfigRecord {
    /// The network architecture the weights belong to.
    pub network: NetworkConfigRecord,
    /// The learning task the network was trained for.
    pub task: TaskRecord,
    /// The name vocabulary for the task's classes or multi-label positions, when known.
    pub labels: Option<Vec<String>>,
}

impl ModelConfigRecord {
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<PathBuf> {
        json::save(self, path)
    }

    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        json::load(path)
    }

    /// Builds a record from a network architecture and the core [`ModelConfig`] it pairs with.
    pub fn from_parts(network: NetworkConfigRecord, config: &ModelConfig) -> Self {
        ModelConfigRecord {
            network,
            task: config.task().clone().into(),
            labels: config.labels().map(|labels| labels.names().to_vec()),
        }
    }

    /// Resolves this record's task and labels into a core [`ModelConfig`].
    ///
    /// # Errors
    /// `InvalidData` when the labels' count does not match the task's declared class or label count.
    pub fn to_model_config(&self) -> Result<ModelConfig> {
        let task = Task::from(self.task.clone());
        let labels = self.labels.clone().map(Labels::new);
        ModelConfig::new(task, labels).map_err(|e| Error::new(InvalidData, e.to_string()))
    }
}
