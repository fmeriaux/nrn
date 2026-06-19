use crate::actions::plot_dataset;
use crate::display::{Artifacts, completed, loaded, saved, show};
use crate::path::PathExt;
use clap::{Args, ValueEnum};
use ndarray::ArrayView2;
use nrn::data::Dataset;
use nrn::data::scalers::{MinMaxScaler, ScalerMethod, ZScoreScaler};
use nrn::io::scalers::ScalerRecord;
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
        let mut dataset = Dataset::load(&self.dataset)?;
        loaded(&dataset);

        let scaler = self.scaling.fit(dataset.features().view());
        dataset.scale_inplace(&scaler);

        completed!("Scaling completed");
        show(&scaler);

        let path = Path::new(&self.dataset);
        let scaled_path = path.sibling("scaled");

        let record: ScalerRecord = scaler.into();

        let mut artifacts = Artifacts::from([
            ("Scaled Dataset", dataset.save(&scaled_path)?),
            ("Scaler", record.save(path.sibling("scaler"))?),
        ]);

        if self.plot
            && let Some(plot) = plot_dataset(&dataset, &scaled_path)?
        {
            artifacts.add("Visualization", plot);
        }

        saved(&artifacts);

        Ok(())
    }
}
