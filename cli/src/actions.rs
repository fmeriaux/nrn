use crate::display::*;
use nrn::charts::RenderConfig;
use nrn::data::Dataset;
use nrn::io::png::save_rgb;
use std::error::Error;
use std::path::{Path, PathBuf};

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

    let render_cfg = RenderConfig::default();
    let saved = save_rgb(
        dataset.draw(&render_cfg)?,
        path,
        render_cfg.width,
        render_cfg.height,
    )?;

    Ok(Some(saved))
}
