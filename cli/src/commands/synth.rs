use crate::actions;
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
    #[arg(short, long, default_value_t = 2, value_parser=1..)]
    features: usize,

    /// Number of clusters to generate in the dataset
    #[arg(short, long, default_value_t = 2, value_parser=1..)]
    clusters: usize,

    /// Specify the training ratio for the dataset split
    #[arg(long, default_value_t = 0.8, value_parser = 0..=1)]
    train_ratio: f32,

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
    pub fn run(&self) -> Result<(), Box<dyn Error>> {
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

        let split_dataset = dataset.split(self.train_ratio);

        generated(&split_dataset);

        let filename = format!(
            "{}-c{}-f{}-n{}-seed{}",
            self.distribution.to_string(),
            self.clusters,
            dataset.n_features(),
            dataset.n_samples(),
            self.seed
        );

        actions::save_dataset(split_dataset, "DATASET", self.plot, &filename)?;

        Ok(())
    }
}
