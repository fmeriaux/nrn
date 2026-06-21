//! Directory serializer for a [`Predictor`]: the network weights as
//! `model.safetensors` and, when present, the scaler as a `scaler.json` sidecar.

use crate::io::scalers::ScalerRecord;
use crate::model::{NeuralNetwork, Predictor};
use std::io::Result;
use std::path::{Path, PathBuf};

/// File stem of the network weights inside a predictor directory.
const MODEL_STEM: &str = "model";
/// File stem of the scaler sidecar inside a predictor directory.
const SCALER_STEM: &str = "scaler";

impl Predictor {
    /// Saves the predictor as a directory: `dir/model.safetensors` for the
    /// network and, when a scaler is present, `dir/scaler.json` beside it.
    /// Returns the directory path.
    pub fn save<P: AsRef<Path>>(&self, dir: P) -> Result<PathBuf> {
        let dir = dir.as_ref();
        self.network.save(dir.join(MODEL_STEM))?;
        if let Some(scaler) = &self.scaler {
            ScalerRecord::from(scaler.clone()).save(dir.join(SCALER_STEM))?;
        }
        Ok(dir.to_path_buf())
    }

    /// Loads a predictor from a directory written by [`Predictor::save`]. The
    /// scaler is read only when a `scaler.json` sidecar exists.
    pub fn load<P: AsRef<Path>>(dir: P) -> Result<Self> {
        let dir = dir.as_ref();
        let network = NeuralNetwork::load(dir.join(MODEL_STEM))?;

        let scaler = if dir.join(SCALER_STEM).with_extension("json").exists() {
            Some(ScalerRecord::load(dir.join(SCALER_STEM))?.into())
        } else {
            None
        };

        Ok(Predictor { network, scaler })
    }
}
