use super::{Format, ImageSize};
use crate::actions::load_or_read_instance;
use crate::display::{Artifacts, loaded, saved};
use clap::Args;
use nrn::model::Predictor;
use nrn::plot::DiagramOptions;
use std::error::Error;
use std::path::Path;

#[derive(Args, Debug)]
pub struct ActivationsArgs {
    /// Model directory to visualize (network plus its optional scaler)
    model: String,

    /// Instance file to run through the network; when omitted, the features are read from stdin
    #[arg(short, long)]
    instance: Option<String>,

    /// Maximum neurons drawn per layer; larger layers are sampled evenly
    #[arg(long, default_value_t = 24, value_parser = clap::value_parser!(u16).range(1..=256))]
    max_units: u16,

    /// Drop connections whose normalized magnitude is below this (image only)
    #[arg(long, default_value_t = 0.0)]
    min_edge: f32,

    /// Output format
    #[arg(long, value_enum, default_value_t = Format::default())]
    format: Format,

    #[command(flatten)]
    size: ImageSize,
}

impl ActivationsArgs {
    pub fn run(self) -> Result<(), Box<dyn Error>> {
        if !(0.0..=1.0).contains(&self.min_edge) {
            return Err("--min-edge must be between 0.0 and 1.0".into());
        }

        let predictor = Predictor::load(&self.model)?;
        loaded(&predictor);

        let instance = load_or_read_instance(self.instance, predictor.network.input_size())?;

        let options = DiagramOptions {
            max_units: self.max_units as usize,
            min_edge_magnitude: self.min_edge,
        };
        let diagram = predictor.activation_diagram(instance.view(), &options)?;

        match self.format {
            Format::Console => println!("{}", diagram.to_console()),
            Format::Image => {
                let base = Path::new(&self.model).join("activations");
                let path = diagram.to_image(&self.size.config())?.save(base)?;
                saved(&Artifacts::single("Activation Diagram", path));
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        args: ActivationsArgs,
    }

    fn parse(extra: &[&str]) -> ActivationsArgs {
        let mut argv = vec!["activations", "model.dir"];
        argv.extend_from_slice(extra);
        Cli::parse_from(argv).args
    }

    #[test]
    fn format_defaults_to_console_and_max_units_to_twenty_four() {
        let args = parse(&[]);
        assert_eq!(args.format, Format::Console);
        assert_eq!(args.max_units, 24);
    }

    #[test]
    fn flags_override_the_defaults() {
        let args = parse(&["--format", "image", "--max-units", "8", "--min-edge", "0.2"]);
        assert_eq!(args.format, Format::Image);
        assert_eq!(args.max_units, 8);
        assert_eq!(args.min_edge, 0.2);
    }

    #[test]
    fn instance_is_optional() {
        assert!(parse(&[]).instance.is_none());
    }
}
