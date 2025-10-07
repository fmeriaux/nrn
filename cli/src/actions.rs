use crate::display::*;
use nrn::charts::RenderConfig;
use nrn::data::Dataset;
use nrn::data::scalers::ScalerMethod;
use nrn::io::png::save_rgb;
use nrn::io::scalers::ScalerRecord;
use nrn::model::{NeuralNetwork, NeuronLayerSpec};
use nrn::training::History;
use std::error::Error;
use std::path::Path;

pub(crate) fn load_dataset<P: AsRef<Path>>(path: P) -> Result<Dataset, Box<dyn Error>> {
    let dataset = Dataset::load(&path)?;
    loaded(&dataset);
    Ok(dataset)
}

pub(crate) fn save_dataset<P: AsRef<Path>>(
    dataset: Dataset,
    name: &str,
    plot: bool,
    path: P,
) -> Result<(), Box<dyn Error>> {
    saved_at(DATASET_ICON, name, dataset.save(&path)?);

    if plot {
        plot_dataset(&dataset, path)?;
    }

    Ok(())
}

pub(crate) fn save_scaler<P: AsRef<Path>>(
    scaler: ScalerMethod,
    path: P,
) -> Result<(), Box<dyn Error>> {
    let record: ScalerRecord = scaler.into();
    saved_at(SCALER_ICON, "SCALER", record.save(path)?);
    Ok(())
}

pub(crate) fn load_scaler(filename: &str) -> Result<ScalerMethod, Box<dyn Error>> {
    let scaler: ScalerMethod = ScalerRecord::load(filename)?.into();
    loaded(&scaler);
    Ok(scaler)
}

/// Plots the dataset if it has exactly two features and saves the plot to a file.
pub(crate) fn plot_dataset<P: AsRef<Path>>(
    dataset: &Dataset,
    path: P,
) -> Result<(), Box<dyn Error>> {
    if dataset.n_features() == 2 {
        let render_cfg = RenderConfig::default();
        saved_at(
            PLOT_ICON,
            "VISUALIZATION",
            save_rgb(
                dataset.draw(&render_cfg)?,
                path,
                render_cfg.width,
                render_cfg.height,
            )?,
        );
    } else {
        warning("Plotting is only available for datasets with exactly two features");
    }

    Ok(())
}

pub(crate) fn initialize_model(
    input_size: usize,
    layer_specs: &Vec<NeuronLayerSpec>,
) -> NeuralNetwork {
    let model = NeuralNetwork::initialization(input_size, &layer_specs);
    initialized(&model);
    model
}

pub(crate) fn load_model(filename: &str) -> Result<NeuralNetwork, Box<dyn Error>> {
    let model = NeuralNetwork::load(filename)?;
    loaded(&model);
    Ok(model)
}

pub(crate) fn save_model(filename: &str, model: &NeuralNetwork) -> Result<(), Box<dyn Error>> {
    saved_at(MODEL_ICON, "NEURAL NETWORK", model.save(filename)?);
    Ok(())
}

pub(crate) fn load_training_history(filename: &str) -> Result<History, Box<dyn Error>> {
    let history = History::load(filename)?;

    assert!(
        history.model.len() > 2,
        "Training history must contain more than two checkpoints to plot."
    );

    loaded(&history);

    Ok(history)
}

pub(crate) fn save_training_history(
    filename: &str,
    history: &History,
) -> Result<(), Box<dyn Error>> {
    saved_at(HISTORY_ICON, "TRAINING HISTORY", history.save(filename)?);
    Ok(())
}
