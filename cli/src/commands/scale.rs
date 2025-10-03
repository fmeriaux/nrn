use crate::actions;
use crate::display::completed;
use clap::{Args, ValueEnum};
use console::style;
use ndarray::ArrayView2;
use nrn::data::scalers::{MinMaxScaler, Scaler, ScalerMethod, ZScoreScaler};
use std::path::Path;

#[derive(Args, Debug)]
pub struct ScaleArgs {
    /// Name of the dataset to scale
    dataset: String,

    /// Specify the scaling method to apply to the dataset
    scaling: ScalingOption,

    /// Indicates whether to visualize the scaled dataset
    #[arg(long, default_value_t = false)]
    plot: bool,
}

#[derive(ValueEnum, Clone, Debug)]
pub enum ScalingOption {
    MinMax,
    ZScore,
}

impl ScalingOption {
    pub fn fit(&self, data: ArrayView2<f32>) -> ScalerMethod {
        match self {
            ScalingOption::MinMax => ScalerMethod::MinMax(MinMaxScaler::default().fit(data)),
            ScalingOption::ZScore => ScalerMethod::ZScore(ZScoreScaler::default().fit(data)),
        }
    }
}

impl ScaleArgs {
    pub fn run(self) -> Result<(), Box<dyn std::error::Error>> {
        let mut split_dataset = actions::load_dataset(&self.dataset)?;

        let scaler = self.scaling.fit(split_dataset.train.features.view());
        split_dataset.scale_inplace(&scaler);

        completed(&format!(
            "Scaled with {}",
            style(&scaler.name().to_uppercase()).bold().blue()
        ));

        // Extract the filename without extension
        let path = Path::new(&self.dataset);
        let filename = path
            .file_stem()
            .ok_or_else(|| panic!("Failed to get file stem from path: {}", path.display()))?;

        let dataset_path = path.with_file_name(format!("scaled-{}", filename.to_string_lossy()));
        let scaler_path = path.with_file_name(format!("scaler-{}", filename.to_string_lossy()));

        actions::save_dataset(split_dataset, "SCALED DATASET", self.plot, &dataset_path)?;
        actions::save_scaler(scaler, scaler_path)?;

        Ok(())
    }
}
