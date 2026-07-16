use crate::display::{Artifacts, generated, preview, saved, warning};
use clap::{Args, ValueEnum};
use ndarray_rand::rand::random;
use nrn::data::synth::{Distribution, SynthDataset, SynthParams, SynthParamsError};
use std::error::Error;

#[derive(Args, Debug)]
pub struct SynthArgs {
    /// Seed for reproducibility (random when omitted)
    #[arg(short, long)]
    seed: Option<u64>,

    /// Type of distribution to use for generating data
    #[arg(short, long)]
    distribution: DistributionOption,

    /// Number of samples to generate
    #[arg(short = 'n', long, default_value_t = 1000)]
    samples: usize,

    /// Number of features in the generated data
    #[arg(short, long, default_value_t = 2)]
    features: usize,

    /// Number of clusters to generate in the dataset
    #[arg(short, long, default_value_t = 2)]
    clusters: usize,

    /// Minimum value for each feature in the dataset
    #[arg(long, default_value_t = -100.0)]
    min: f32,

    /// Maximum value for each feature in the dataset
    #[arg(long, default_value_t = 100.0)]
    max: f32,

    /// Ring overlap fraction between consecutive rings, negative for a gap [ring only]
    #[arg(long, default_value_t = -0.2)]
    overlap: f32,

    /// Number of turns each spiral arm makes [spiral only]
    #[arg(long, default_value_t = 1.5)]
    turns: f32,

    /// Spiral jitter, as a fraction of the arm's max radius [spiral only]
    #[arg(long, default_value_t = 0.03)]
    noise: f32,

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
    /// Selects the distribution, reading each variant's shape knobs from the
    /// flags scoped to it. Knobs for other distributions are ignored (see help).
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
        let distribution = Distribution::from(self);
        let generator = SynthDataset::new(SynthParams::try_from(self)?, distribution)?;

        let seed = self.seed.unwrap_or_else(random);
        let dataset = generator.generate(seed);

        if dataset.n_samples() != self.samples {
            warning!(
                "Requested {} samples but generated {} ({} dropped due to uneven cluster division)",
                self.samples,
                dataset.n_samples(),
                self.samples - dataset.n_samples()
            );
        }

        generated(&dataset);

        // A two-feature dataset gets an inline scatter preview in the terminal.
        if dataset.n_features() == 2 {
            preview(&dataset.figure()?);
        }

        let filename = self.output.clone().unwrap_or_else(|| {
            format!(
                "{distribution}-seed{seed}-c{}-f{}-n{}",
                self.clusters,
                dataset.n_features(),
                dataset.n_samples()
            )
        });
        let artifacts = Artifacts::single("Dataset", dataset.save(&filename)?);

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

    fn distribution(extra: &[&str]) -> Distribution {
        Distribution::from(&parse(extra))
    }

    #[test]
    fn uniform_maps_to_uniform_distribution() {
        assert_eq!(distribution(&["uniform"]), Distribution::Uniform);
    }

    #[test]
    fn ring_carries_overlap_flag() {
        assert_eq!(
            distribution(&["ring", "--overlap", "0.25"]),
            Distribution::Ring { overlap: 0.25 }
        );
    }

    #[test]
    fn ring_overlap_defaults_when_omitted() {
        assert_eq!(
            distribution(&["ring"]),
            Distribution::Ring { overlap: -0.2 }
        );
    }

    #[test]
    fn spiral_carries_turns_and_noise() {
        assert_eq!(
            distribution(&["spiral", "--turns", "2.0", "--noise", "0.1"]),
            Distribution::Spiral {
                turns: 2.0,
                noise: 0.1
            }
        );
    }

    #[test]
    fn spiral_knobs_default_when_omitted() {
        assert_eq!(
            distribution(&["spiral"]),
            Distribution::Spiral {
                turns: 1.5,
                noise: 0.03
            }
        );
    }

    #[test]
    fn seed_is_optional() {
        let args = Cli::parse_from(["synth", "--distribution", "uniform"]).args;
        assert_eq!(args.seed, None);
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
