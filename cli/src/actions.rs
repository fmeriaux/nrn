use crate::display::*;
use nrn::activations::RELU;
use nrn::charts::RenderConfig;
use nrn::data::Dataset;
use nrn::data::scalers::ScalerMethod;
use nrn::io::checkpoint::CheckpointArchive;
use nrn::io::png::save_rgb;
use nrn::io::scalers::ScalerRecord;
use nrn::model::{LayerPlan, NeuralNetwork, NeuronLayerSpec};
use std::error::Error;
use std::path::{Path, PathBuf};

pub(crate) fn load_dataset<P: AsRef<Path>>(path: P) -> Result<Dataset, Box<dyn Error>> {
    let dataset = Dataset::load(&path)?;
    loaded(&dataset);
    Ok(dataset)
}

pub(crate) fn load_scaler<P: AsRef<Path>>(path: P) -> Result<ScalerMethod, Box<dyn Error>> {
    let scaler: ScalerMethod = ScalerRecord::load(&path)?.into();
    loaded(&scaler);
    Ok(scaler)
}

/// The dataset's scatter plot saved to a file when it has exactly two features,
/// otherwise a warning and `None`.
pub(crate) fn plot_dataset<P: AsRef<Path>>(
    dataset: &Dataset,
    path: P,
) -> Result<Option<PathBuf>, Box<dyn Error>> {
    if dataset.n_features() != 2 {
        warning("Plotting is only available for datasets with exactly two features");
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

pub(crate) fn initialize_model_with(
    dataset: &Dataset,
    plan: LayerPlan,
    seed: u64,
) -> NeuralNetwork {
    let layer_specs = NeuronLayerSpec::plan(
        plan,
        dataset.n_features(),
        dataset.n_classes(),
        dataset.n_samples(),
        &*RELU,
    );
    let model = NeuralNetwork::initialization(dataset.n_features(), &layer_specs, seed);
    initialized(&model);
    model
}

pub(crate) fn load_model<P: AsRef<Path>>(path: P) -> Result<NeuralNetwork, Box<dyn Error>> {
    let model = NeuralNetwork::load(&path)?;
    loaded(&model);
    Ok(model)
}

pub(crate) fn load_history<P: AsRef<Path>>(path: P) -> Result<CheckpointArchive, Box<dyn Error>> {
    let archive = CheckpointArchive::load(&path)?;

    if archive.len() <= 2 {
        return Err("Training run must contain more than two checkpoints to plot.".into());
    }

    loaded(&archive);

    Ok(archive)
}
