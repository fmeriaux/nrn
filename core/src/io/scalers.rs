use crate::io::json;
use crate::data::scalers::{MinMaxScaler, ScalerMethod, ZScoreScaler};
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::io::Result;
use std::path::Path;

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type", content = "params")]
pub enum ScalerRecord {
    MinMax(MinMaxRecord),
    ZScore(ZScoreRecord),
}

#[derive(Serialize, Deserialize, Debug)]
pub struct MinMaxRecord {
    pub min: Array1<f32>,
    pub max: Array1<f32>,
    pub range: (f32, f32),
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ZScoreRecord {
    pub mean: Array1<f32>,
    pub std_dev: Array1<f32>,
}

impl From<ScalerMethod> for ScalerRecord {
    fn from(scaler: ScalerMethod) -> Self {
        match scaler {
            ScalerMethod::MinMax(s) => ScalerRecord::MinMax(MinMaxRecord {
                min: s.min,
                max: s.max,
                range: s.range,
            }),
            ScalerMethod::ZScore(s) => ScalerRecord::ZScore(ZScoreRecord {
                mean: s.mean,
                std_dev: s.std_dev,
            }),
        }
    }
}

impl From<ScalerRecord> for ScalerMethod {
    fn from(record: ScalerRecord) -> Self {
        match record {
            ScalerRecord::MinMax(s) => ScalerMethod::MinMax(MinMaxScaler {
                min: s.min,
                max: s.max,
                range: s.range,
            }),
            ScalerRecord::ZScore(s) => ScalerMethod::ZScore(ZScoreScaler {
                mean: s.mean,
                std_dev: s.std_dev,
            }),
        }
    }
}

impl ScalerRecord {
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        json::save(self, path)
    }

    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        json::load(path)
    }
}
