use crate::actions::save_dataset;
use crate::console::{generated, warning};
use clap::{Args, ValueEnum};
use nrn::data::synth::{Distribution, SynthDataset, SynthParams, SynthParamsError};
use std::error::Error;

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

    /// Ring overlap fraction between consecutive rings (negative = a gap)
    #[arg(long, default_value_t = -0.2)]
    overlap: f32,

    /// Number of turns each spiral arm makes
    #[arg(long, default_value_t = 1.5)]
    turns: f32,

    /// Spiral jitter, as a fraction of the arm's max radius
    #[arg(long, default_value_t = 0.03)]
    noise: f32,

    /// Indicates whether to visualize the generated dataset (requires exactly two features)
    #[arg(long, default_value_t = false)]
    plot: bool,

    /// Name to save the dataset under (defaults to the dataset's identifier)
    #[arg(short, long)]
    output: Option<String>,
}

#[derive(ValueEnum, Clone, Copy, Debug)]
enum DistributionOption {
    Uniform,
    Ring,
    Spiral,
}

impl From<&SynthArgs> for Distribution {
    /// Selects the core distribution, supplying each variant's shape knobs from
    /// the relevant flags.
    fn from(args: &SynthArgs) -> Self {
        match args.distribution {
            DistributionOption::Uniform => Distribution::Uniform,
            DistributionOption::Ring => Distribution::Ring {
                overlap: args.overlap,
            },
            DistributionOption::Spiral => Distribution::Spiral {
                turns: args.turns,
                noise: args.noise,
            },
        }
    }
}

impl TryFrom<&SynthArgs> for SynthParams {
    type Error = SynthParamsError;

    fn try_from(args: &SynthArgs) -> Result<Self, Self::Error> {
        SynthParams::new(
            args.samples,
            args.features,
            args.clusters,
            args.min,
            args.max,
        )
    }
}

impl SynthArgs {
    pub fn run(&self) -> Result<(), Box<dyn Error>> {
        let generator = SynthDataset::new(SynthParams::try_from(self)?, Distribution::from(self))?;

        let dataset = generator.generate(self.seed);

        if dataset.n_samples() != self.samples {
            warning(&format!(
                "Requested {} samples but generated {} ({} dropped due to uneven cluster division)",
                self.samples,
                dataset.n_samples(),
                self.samples - dataset.n_samples()
            ));
        }

        generated(&dataset);

        let filename = self.output.clone().unwrap_or_else(|| dataset.id());

        save_dataset(dataset, "DATASET", self.plot, &filename)?;

        Ok(())
    }
}
