use crate::actions::{get_file_stem, load_dataset, save_dataset, save_scaler};
use crate::display::completed;
use clap::{Args, ValueEnum};
use console::style;
use ndarray::ArrayView2;
use nrn::data::scalers::{MinMaxScaler, Scaler, ScalerMethod, ZScoreScaler};
use std::path::Path;

#[derive(Args, Debug)]
pub struct ScaleArgs {
    /// The name of the dataset to scale. If a split dataset is provided, it will be unsplit before scaling.
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
        let mut dataset = load_dataset(&self.dataset)?;

        let scaler = self.scaling.fit(dataset.features.view());
        dataset.scale_inplace(&scaler);

        completed(&format!(
            "Scaled with {}",
            style(&scaler.name().to_uppercase()).bold().blue()
        ));

        // Extract the filename without extension
        let path = Path::new(&self.dataset);
        let dataset_name = get_file_stem(&path);

        save_dataset(
            dataset,
            "SCALED DATASET",
            self.plot,
            path.with_file_name(format!("scaled-{}", dataset_name)),
        )?;

        save_scaler(
            scaler,
            path.with_file_name(format!("scaler-{}", dataset_name)),
        )?;

        Ok(())
    }
}
