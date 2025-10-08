use crate::display::*;
use nrn::activations::RELU;
use nrn::charts::RenderConfig;
use nrn::data::Dataset;
use nrn::data::scalers::ScalerMethod;
use nrn::io::png::save_rgb;
use nrn::io::scalers::ScalerRecord;
use nrn::model::{NeuralNetwork, NeuronLayerSpec};
use std::error::Error;
use std::path::Path;
use nrn::checkpoints::Checkpoints;

pub(crate) fn get_file_stem<P: AsRef<Path>>(path: P) -> String {
    let path = path.as_ref();
    path.file_stem()
        .unwrap_or_else(|| panic!("Failed to get file stem from path: {}", path.display()))
        .to_string_lossy()
        .to_string()
}

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

pub(crate) fn load_scaler<P: AsRef<Path>>(path: P) -> Result<ScalerMethod, Box<dyn Error>> {
    let scaler: ScalerMethod = ScalerRecord::load(&path)?.into();
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

pub(crate) fn initialize_model_with(
    dataset: &Dataset,
    layers: Option<Vec<usize>>,
    auto_layers: bool,
) -> NeuralNetwork {
    let layer_specs =if auto_layers {
        NeuronLayerSpec::infer_from(dataset.n_features(), dataset.n_classes(), dataset.n_samples(), &*RELU)
    } else {
        NeuronLayerSpec::network_for(layers.unwrap_or_default(), &*RELU, dataset.n_classes())
    };
    let model = NeuralNetwork::initialization(dataset.n_features(), &layer_specs);
    initialized(&model);
    model
}

pub(crate) fn load_model<P: AsRef<Path>>(path: P) -> Result<NeuralNetwork, Box<dyn Error>> {
    let model = NeuralNetwork::load(&path)?;
    loaded(&model);
    Ok(model)
}

pub(crate) fn save_model<P: AsRef<Path>>(
    path: P,
    model: &NeuralNetwork,
) -> Result<(), Box<dyn Error>> {
    saved_at(MODEL_ICON, "NEURAL NETWORK", model.save(&path)?);
    Ok(())
}

pub(crate) fn load_checkpoints<P: AsRef<Path>>(path: P) -> Result<Checkpoints, Box<dyn Error>> {
    let checkpoints = Checkpoints::load(&path)?;

    assert!(
        checkpoints.len() > 2,
        "Training checkpoints must contain more than two checkpoints to plot."
    );

    loaded(&checkpoints);

    Ok(checkpoints)
}

pub(crate) fn save_checkpoints<P: AsRef<Path>>(
    path: P,
    checkpoints: &Checkpoints,
) -> Result<(), Box<dyn Error>> {
    saved_at(HISTORY_ICON, "TRAINING HISTORY", checkpoints.save(&path)?);
    Ok(())
}
