use super::{Format, ImageSize, render};
use crate::display::{Artifacts, loaded, saved};
use clap::Args;
use nrn::data::Dataset;
use std::error::Error;

#[derive(Args, Debug)]
pub struct DatasetArgs {
    /// Dataset to plot (must have exactly two features)
    dataset: String,

    /// Output format
    #[arg(long, value_enum, default_value_t = Format::default())]
    format: Format,

    #[command(flatten)]
    size: ImageSize,
}

impl DatasetArgs {
    pub fn run(self) -> Result<(), Box<dyn Error>> {
        let dataset = Dataset::load(&self.dataset)?;
        loaded(&dataset);

        // A scatter is only defined for two features; the builder enforces it.
        let figure = dataset.figure()?;

        if let Some(path) = render(&figure, self.format, self.size, &self.dataset)? {
            saved(&Artifacts::single("Dataset Scatter", path));
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
        args: DatasetArgs,
    }

    fn parse(extra: &[&str]) -> DatasetArgs {
        let mut argv = vec!["dataset", "data.parquet"];
        argv.extend_from_slice(extra);
        Cli::parse_from(argv).args
    }

    #[test]
    fn format_defaults_to_console() {
        assert_eq!(parse(&[]).format, Format::Console);
    }

    #[test]
    fn format_image_is_parsed() {
        assert_eq!(parse(&["--format", "image"]).format, Format::Image);
    }

    #[test]
    fn size_defaults_when_omitted() {
        let args = parse(&[]);
        assert_eq!(
            (args.size.config().width, args.size.config().height),
            (1200, 900)
        );
    }

    #[test]
    fn size_flags_override_defaults() {
        let args = parse(&["--width", "640", "--height", "480"]);
        assert_eq!(
            (args.size.config().width, args.size.config().height),
            (640, 480)
        );
    }
}
