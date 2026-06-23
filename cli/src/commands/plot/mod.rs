//! The `plot` subcommand group: turn a dataset or a training run into a figure,
//! rendered inline in the terminal or saved to disk.
//!
//! Both subcommands share a [`Format`] (console vs image) and, for the image
//! format, an [`ImageSize`]. [`render`] is the one place a static figure becomes
//! either a terminal preview or a written PNG.

mod dataset;
mod run;

use crate::display::preview;
use clap::{Args, Subcommand, ValueEnum};
use dataset::DatasetArgs;
use nrn::plot::{Figure, ImageConfig};
use run::RunArgs;
use std::error::Error;
use std::path::{Path, PathBuf};

#[derive(Subcommand, Debug)]
pub enum PlotCommand {
    /// Plot a dataset's feature scatter
    #[command(visible_alias = "ds")]
    Dataset(DatasetArgs),
    /// Plot a training run's curves and decision boundary
    Run(RunArgs),
}

impl PlotCommand {
    pub fn run(self) -> Result<(), Box<dyn Error>> {
        match self {
            PlotCommand::Dataset(args) => args.run(),
            PlotCommand::Run(args) => args.run(),
        }
    }
}

/// Where a figure is rendered.
#[derive(ValueEnum, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum Format {
    /// Inline text in the terminal.
    #[default]
    Console,
    /// A file on disk — PNG for a still, GIF for an animation.
    Image,
}

/// Pixel dimensions for the rasterized output; ignored by the console format.
#[derive(Args, Debug, Clone, Copy)]
pub(super) struct ImageSize {
    /// Width of the plot in pixels
    #[arg(long, default_value_t = 1200, value_parser = clap::value_parser!(u16).range(100..=4096))]
    width: u16,

    /// Height of the plot in pixels
    #[arg(long, default_value_t = 900, value_parser = clap::value_parser!(u16).range(100..=4096))]
    height: u16,
}

impl ImageSize {
    /// The raster configuration for this size.
    pub(super) fn config(&self) -> ImageConfig<'static> {
        ImageConfig::new(self.width as u32, self.height as u32)
    }
}

/// Renders a still `figure`: previews it inline for [`Format::Console`], or saves
/// it as a PNG at `path` for [`Format::Image`], returning the written path.
pub(super) fn render(
    figure: &Figure,
    format: Format,
    size: ImageSize,
    path: impl AsRef<Path>,
) -> Result<Option<PathBuf>, Box<dyn Error>> {
    match format {
        Format::Console => {
            preview(figure);
            Ok(None)
        }
        Format::Image => Ok(Some(figure.to_image(&size.config())?.save(path)?)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct Cli {
        #[command(subcommand)]
        command: PlotCommand,
    }

    #[test]
    fn ds_is_an_alias_for_dataset() {
        let cli = Cli::try_parse_from(["plot", "ds", "data.safetensors"]).unwrap();
        assert!(matches!(cli.command, PlotCommand::Dataset(_)));
    }
}
