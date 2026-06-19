use crate::actions::plot_dataset;
use crate::display::{Artifacts, generated, saved, warning};
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
            warning!(
                "Requested {} samples but generated {} ({} dropped due to uneven cluster division)",
                self.samples,
                dataset.n_samples(),
                self.samples - dataset.n_samples()
            );
        }

        generated(&dataset);

        let filename = self.output.clone().unwrap_or_else(|| dataset.id());

        let mut artifacts = Artifacts::single("Dataset", dataset.save(&filename)?);

        if self.plot
            && let Some(plot) = plot_dataset(&dataset, &filename)?
        {
            artifacts.add("Visualization", plot);
        }

        saved(&artifacts);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    /// Minimal parser wrapper so the flattened [`SynthArgs`] can be exercised
    /// through clap (defaults, value enums, and the conversion impls) in one path.
    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        args: SynthArgs,
    }

    fn parse(extra: &[&str]) -> SynthArgs {
        let mut argv = vec!["synth", "--seed", "7", "--distribution"];
        argv.extend_from_slice(extra);
        Cli::parse_from(argv).args
    }

    #[test]
    fn uniform_maps_to_uniform_distribution() {
        let args = parse(&["uniform"]);
        assert_eq!(Distribution::from(&args), Distribution::Uniform);
    }

    #[test]
    fn ring_carries_overlap_flag() {
        let args = parse(&["ring", "--overlap", "0.25"]);
        assert_eq!(
            Distribution::from(&args),
            Distribution::Ring { overlap: 0.25 }
        );
    }

    #[test]
    fn ring_overlap_defaults_when_omitted() {
        let args = parse(&["ring"]);
        assert_eq!(
            Distribution::from(&args),
            Distribution::Ring { overlap: -0.2 }
        );
    }

    #[test]
    fn spiral_carries_turns_and_noise() {
        let args = parse(&["spiral", "--turns", "2.0", "--noise", "0.1"]);
        assert_eq!(
            Distribution::from(&args),
            Distribution::Spiral {
                turns: 2.0,
                noise: 0.1
            }
        );
    }

    #[test]
    fn synth_params_built_from_sizing_flags() {
        let args = parse(&[
            "uniform", "-n", "200", "-f", "3", "-c", "4", "--min=-1", "--max", "5",
        ]);
        let params = SynthParams::try_from(&args).expect("valid params");
        assert_eq!(params, SynthParams::new(200, 3, 4, -1.0, 5.0).unwrap());
    }

    #[test]
    fn synth_params_surface_validation_errors() {
        // Fewer samples than clusters is rejected by the core validator.
        let args = parse(&["uniform", "-n", "2", "-c", "5"]);
        assert_eq!(
            SynthParams::try_from(&args),
            Err(SynthParamsError::NotEnoughSamples {
                samples: 2,
                clusters: 5,
            })
        );
    }
}
