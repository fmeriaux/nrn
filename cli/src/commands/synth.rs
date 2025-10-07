use crate::actions::save_dataset;
use crate::display::generated;
use clap::{Args, ValueEnum};
use nrn::data::synth::{DatasetGenerator, RingDataset, UniformDataset};
use std::error::Error;
use std::fmt;
use std::sync::Arc;

#[derive(Args, Debug)]
pub struct SynthArgs {
    /// Seed for random number generation reproducibility
    #[arg(short, long)]
    seed: u64,

    /// Type of distribution to use for generating data
    #[arg(short, long)]
    distribution: DistributionOption,

    /// Number of samples to generate
    #[arg(short = 'n', long, default_value_t = 100)]
    samples: usize,

    /// Number of features in the generated data
    #[arg(short, long, default_value_t = 2)]
    features: usize,

    /// Number of clusters to generate in the dataset
    #[arg(short, long, default_value_t = 2)]
    clusters: usize,

    /// Minimum value for each feature in the dataset
    #[arg(long, default_value_t = 0.0)]
    min: f32,

    /// Maximum value for each feature in the dataset
    #[arg(long, default_value_t = 10.0)]
    max: f32,

    /// Indicates whether to visualize the generated dataset (requires exactly two features)
    #[arg(long, default_value_t = false)]
    plot: bool,
}

#[derive(ValueEnum, Clone, Debug)]
enum DistributionOption {
    Uniform,
    Ring,
}

impl fmt::Display for DistributionOption {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DistributionOption::Uniform => write!(f, "uniform"),
            DistributionOption::Ring => write!(f, "ring"),
        }
    }
}

impl SynthArgs {
    /// Validate the command line arguments
    fn validate(&self) -> Result<(), String> {
        if self.features < 1 {
            return Err("Le nombre de features doit être au moins 1.".to_string());
        }
        if self.clusters < 1 {
            return Err("Le nombre de clusters doit être au moins 1.".to_string());
        }
        if self.samples < self.clusters {
            return Err(
                "Le nombre d'échantillons doit être au moins égal au nombre de clusters."
                    .to_string(),
            );
        }
        if self.min >= self.max {
            return Err(
                "La valeur minimale doit être inférieure à la valeur maximale.".to_string(),
            );
        }
        Ok(())
    }

    pub fn run(&self) -> Result<(), Box<dyn Error>> {
        self.validate()?;

        // 🗂️ GENERATE THE DATASET
        let generator: Arc<dyn DatasetGenerator> = match self.distribution {
            DistributionOption::Uniform => Arc::new(UniformDataset {
                n_samples: self.samples,
                n_features: self.features,
                n_clusters: self.clusters,
                feature_min: self.min,
                feature_max: self.max,
            }),
            DistributionOption::Ring => Arc::new(RingDataset {
                n_samples: self.samples,
                n_features: self.features,
                n_clusters: self.clusters,
                feature_min: self.min,
                feature_max: self.max,
            }),
        };

        let dataset = generator.generate(self.seed);

        generated(&dataset);

        let filename = format!(
            "{}-c{}-f{}-n{}-seed{}",
            self.distribution.to_string(),
            self.clusters,
            dataset.n_features(),
            dataset.n_samples(),
            self.seed
        );

        save_dataset(dataset, "DATASET", self.plot, &filename)?;

        Ok(())
    }
}
