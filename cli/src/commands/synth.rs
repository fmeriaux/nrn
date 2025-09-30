use std::error::Error;
use clap::{Args, ValueEnum};
use std::fmt;
use std::sync::Arc;
use colored::Colorize;
use nrn::data::synth::{DatasetGenerator, RingDataset, UniformDataset};
use crate::{actions, display_success};

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

    /// Specify the training ratio for the dataset split
    #[arg(long, default_value_t = 0.8)]
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

        display_success!(
                "{} ({} features, {} samples -> {} training, {} test)",
                "Dataset generated".bright_green(),
                dataset.n_features().to_string().yellow(),
                dataset.n_samples().to_string().yellow(),
                split_dataset.train.n_samples().to_string().yellow(),
                split_dataset.test.n_samples().to_string().yellow()
            );

        let filename = format!(
            "{}-c{}-f{}-n{}-seed{}",
            self.distribution.to_string(),
            self.clusters,
            dataset.n_features(),
            dataset.n_samples(),
            self.seed
        );

        actions::save_dataset(&split_dataset, &filename)?;

        if self.plot {
            actions::plot_dataset(&filename, &dataset, 800, 600)?;
        }

        Ok(())
    }
}