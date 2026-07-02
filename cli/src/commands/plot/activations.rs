use super::{DiagramArgs, Format, ImageSize};
use crate::actions::load_or_read_instance;
use crate::display::{Artifacts, loaded, saved};
use crate::path::PathExt;
use clap::Args;
use nrn::model::Predictor;
use std::error::Error;
use std::path::{Path, PathBuf};

#[derive(Args, Debug)]
pub struct ActivationsArgs {
    /// Model directory to visualize (network plus its optional scaler)
    model: String,

    /// Instance file to run through the network; when omitted, the features are read from stdin
    #[arg(short, long)]
    instance: Option<String>,

    /// Output path for the image, without extension; defaults to `activations-{model}` in the current directory (image only)
    #[arg(short, long)]
    output: Option<String>,

    /// Output format
    #[arg(long, value_enum, default_value_t = Format::default())]
    format: Format,

    #[command(flatten)]
    diagram: DiagramArgs,

    #[command(flatten)]
    size: ImageSize,
}

impl ActivationsArgs {
    pub fn run(self) -> Result<(), Box<dyn Error>> {
        let options = self.diagram.options()?;

        let predictor = Predictor::load(&self.model)?;
        loaded(&predictor);

        let image_path = self.image_path();
        let instance = load_or_read_instance(self.instance, predictor.network.input_size())?;

        let diagram = predictor.activation_diagram(instance.view(), &options)?;

        match self.format {
            Format::Console => println!("{}", diagram.to_console()),
            Format::Image => {
                let path = diagram.to_image(&self.size.config())?.save(image_path)?;
                saved(&Artifacts::single("Activation Diagram", path));
            }
        }

        Ok(())
    }

    /// The output path for the rendered diagram: `--output` when given, otherwise
    /// `activations-{model}` in the current directory (the extension is set by the
    /// writer).
    fn image_path(&self) -> PathBuf {
        self.output.clone().map(PathBuf::from).unwrap_or_else(|| {
            PathBuf::from(format!(
                "activations-{}",
                Path::new(&self.model).file_stem_string()
            ))
        })
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
        let options = args.diagram.options().unwrap();
        assert_eq!(options.max_units, 24);
        // A small positive floor prunes dead connections out of the box.
        assert_eq!(options.min_edge_magnitude, 0.01);
    }

    #[test]
    fn flags_override_the_defaults() {
        let args = parse(&["--format", "image", "--max-units", "8", "--min-edge", "0.2"]);
        assert_eq!(args.format, Format::Image);
        let options = args.diagram.options().unwrap();
        assert_eq!(options.max_units, 8);
        assert_eq!(options.min_edge_magnitude, 0.2);
    }

    #[test]
    fn a_min_edge_outside_the_unit_range_is_rejected() {
        assert!(parse(&["--min-edge", "1.5"]).diagram.options().is_err());
    }

    #[test]
    fn instance_is_optional() {
        assert!(parse(&[]).instance.is_none());
    }

    #[test]
    fn image_defaults_to_the_current_directory_named_after_the_model() {
        // The default model argument is `model.dir`, whose stem is `model`.
        assert_eq!(parse(&[]).image_path(), Path::new("activations-model"));
    }

    #[test]
    fn output_overrides_the_default_image_path() {
        let args = parse(&["--output", "diagrams/run1"]);
        assert_eq!(args.image_path(), Path::new("diagrams/run1"));
    }
}
