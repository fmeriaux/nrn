use crate::{actions, display_success};
use clap::{Args, ValueEnum};
use colored::Colorize;
use ndarray::ArrayView2;
use nrn::data::scalers::{MinMaxScaler, Scaler, ScalerMethod, ZScoreScaler};

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

        display_success!(
            "{} with {}",
            "Dataset scaled".bright_green(),
            scaler.name().yellow()
        );

        actions::save_dataset(&split_dataset, &format!("scaled-{}", self.dataset))?;
        actions::save_scaler(
            scaler,
            &format!("scaler-{}", self.dataset.trim_end_matches(".h5")),
        )?;

        if self.plot {
            actions::plot_dataset(
                &format!("scaled-{}", self.dataset),
                &split_dataset.unsplit(),
                800,
                600,
            )?;
        }

        Ok(())
    }
}
