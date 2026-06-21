use crate::display::*;
use nrn::data::Dataset;
use nrn::io::png::save_rgb;
use nrn::plot::ImageConfig;
use std::error::Error;
use std::path::{Path, PathBuf};

/// The fraction of the data range added as whitespace around plot axes.
pub(crate) const DEFAULT_PADDING_FACTOR: f32 = 0.05;

/// The dataset's scatter plot saved to a file when it has exactly two features,
/// otherwise a warning and `None`.
pub(crate) fn plot_dataset<P: AsRef<Path>>(
    dataset: &Dataset,
    path: P,
) -> Result<Option<PathBuf>, Box<dyn Error>> {
    if dataset.n_features() != 2 {
        warning!("Plotting is only available for datasets with exactly two features");
        return Ok(None);
    }

    let cfg = ImageConfig::default();
    let figure = dataset.figure(DEFAULT_PADDING_FACTOR)?;
    let saved = save_rgb(figure.to_image(&cfg)?, path, cfg.width, cfg.height)?;

    Ok(Some(saved))
}
