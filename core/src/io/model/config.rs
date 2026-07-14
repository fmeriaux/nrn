//! The `config.json` blueprint: network architecture plus the task its logits are read for.

use crate::io::json;
use crate::io::model::network::NetworkConfigRecord;
use crate::io::model::task::TaskRecord;
use serde::{Deserialize, Serialize};
use std::io::Result;
use std::path::{Path, PathBuf};

/// File stem of the blueprint inside a predictor or training run directory.
pub(crate) const CONFIG_STEM: &str = "config";
/// File stem of the preprocessor sidecar inside a predictor or training run directory.
pub(crate) const PREPROCESSOR_STEM: &str = "preprocessor";

/// Serializable blueprint of a model: the network architecture its weights belong to
/// and the task its logits are read for.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct ModelConfigRecord {
    /// The network architecture the weights belong to.
    pub network: NetworkConfigRecord,
    /// The learning task the network was trained for.
    pub task: TaskRecord,
}

impl ModelConfigRecord {
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<PathBuf> {
        json::save(self, path)
    }

    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        json::load(path)
    }
}
