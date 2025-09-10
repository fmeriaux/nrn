use Scaler::*;
use ScalingMethod::*;
use ndarray::{Array, Dimension};
use std::fmt;
use std::fs::File;
use std::io::Write;
use serde::{Deserialize, Serialize};
use crate::synth::SplitDataset;

/// Represents the names of the scaling methods available for the dataset features.
#[derive(Clone, Debug)]
pub enum ScalingMethod {
    MinMax,
    ZScore,
}

/// Represents a scaling method that can be applied to datasets.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Scaler {
    /// Min-Max scaling, which scales the data to a range of [0, 1].
    #[serde(rename = "min_max")]
    MinMaxScaler { min: f32, max: f32 },
    /// Z-Score scaling, which standardizes the data to have a mean of 0 and a standard deviation of 1.
    #[serde(rename = "z_score")]
    ZScoreScaler { mean: f32, std_dev: f32 },
}

impl fmt::Display for ScalingMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MinMax => write!(f, "min-max"),
            ZScore => write!(f, "z-score"),
        }
    }
}

impl Scaler {
    /// Applies the scaling method to the provided data.
    pub fn apply<D: Dimension>(&self, data: &Array<f32, D>) -> Array<f32, D> {
        if data.is_empty() {
            return data.to_owned();
        }

        match self {
            MinMaxScaler { min, max } => data.mapv(|x| (x - min) / (max - min + f32::EPSILON)),
            ZScoreScaler { mean, std_dev } => data.mapv(|x| (x - mean) / (std_dev + f32::EPSILON)),
        }
    }

    /// Computes the Min-Max scaling parameters (min and max) from the provided data.
    pub fn min_max<D: Dimension>(data: &Array<f32, D>) -> Scaler {
        let min = data.fold(f32::INFINITY, |a, &b| a.min(b));
        let max = data.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        MinMaxScaler { min, max }
    }

    /// Computes the Z-Score scaling parameters (mean and standard deviation) from the provided data.
    pub fn z_score<D: Dimension>(data: &Array<f32, D>) -> Scaler {
        let mean = data.mean().unwrap_or(0.0);
        let std_dev = data.std(0.0);
        ZScoreScaler { mean, std_dev }
    }

    /// Creates a `ScalingMethod` from the provided name and data.
    pub fn fit_from_name<D: Dimension>(name: &ScalingMethod, data: &Array<f32, D>) -> Scaler {
        match name {
            MinMax => Self::min_max(&data),
            ZScore => Self::z_score(&data),
        }
    }

    /// Converts the scaler to a `ScalingMethod`.
    pub fn to_method(&self) -> ScalingMethod {
        match self {
            MinMaxScaler { .. } => MinMax,
            ZScoreScaler { .. } => ZScore,
        }
    }

    /// Saves the scaler parameters to a JSON file.
    /// # Arguments
    /// - `name`: The name of the file to save the scaler parameters to (without extension).
    pub fn save(&self, name: &str) -> std::io::Result<()> {
        let filename = format!("{}.json", name);
        let serialized = serde_json::to_string(self)?;
        let mut file = File::create(filename)?;
        file.write_all(serialized.as_bytes())?;
        Ok(())
    }

    /// Loads the scaler parameters from a JSON file.
    /// # Arguments
    /// - `name`: The name of the file to load the scaler parameters from
    pub fn load(name: &str) -> std::io::Result<Self> {
        let filename = format!("{}.json", name.trim_end_matches(".json"));
        let file = File::open(filename)?;
        let scaler: Scaler = serde_json::from_reader(file)?;
        Ok(scaler)
    }
}

impl SplitDataset {
    /// Applies the specified scaling method to the training and test datasets.
    pub fn scale(&mut self, scaler: &Scaler) {
        self.train.features = scaler.apply(&self.train.features);
        self.test.features = scaler.apply(&self.test.features);
    }
}
