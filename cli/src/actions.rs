use crate::display::*;
use nrn::activations::RELU;
use nrn::charts::RenderConfig;
use nrn::data::Dataset;
use nrn::data::scalers::ScalerMethod;
use nrn::io::png::save_rgb;
use nrn::io::scalers::ScalerRecord;
use nrn::io::training_history::SnapshotRecorder;
use nrn::model::{NeuralNetwork, NeuronLayerSpec};
use nrn::training_history::TrainingHistory;
use std::error::Error;
use std::path::Path;

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
    let layer_specs = if auto_layers {
        NeuronLayerSpec::infer_from(
            dataset.n_features(),
            dataset.n_classes(),
            dataset.n_samples(),
            &*RELU,
        )
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

pub(crate) fn load_history<P: AsRef<Path>>(path: P) -> Result<TrainingHistory, Box<dyn Error>> {
    let history = TrainingHistory::load(&path)?;

    if history.len() <= 2 {
        return Err("Training history must contain more than two snapshots to plot.".into());
    }

    loaded(&history);

    Ok(history)
}

/// Loads a model from a snapshot directory (`snapshot-{n}/model.safetensors`).
///
/// Errors if the path does not match the `snapshot-{n}` format, or if
/// `model.safetensors` / `evaluations.json` are missing.
/// Returns `(model, snapshot_index)`.
pub(crate) fn load_snapshot<P: AsRef<Path>>(
    path: P,
) -> Result<(NeuralNetwork, usize), Box<dyn Error>> {
    let path = path.as_ref();

    let snapshot_index = path
        .file_name()
        .and_then(|n| n.to_str())
        .and_then(|n| n.strip_prefix("snapshot-"))
        .and_then(|s| s.parse::<usize>().ok())
        .ok_or_else(|| -> Box<dyn Error> {
            format!(
                "'{}' is not a valid snapshot directory \
                 (expected a path ending in snapshot-{{n}})",
                path.display()
            )
            .into()
        })?;

    if !path.join("model.safetensors").exists() {
        return Err(format!("snapshot '{}' is missing model.safetensors", path.display()).into());
    }
    if !path.join("evaluations.json").exists() {
        return Err(format!("snapshot '{}' is missing evaluations.json", path.display()).into());
    }

    let model = load_model(path.join("model"))?;
    Ok((model, snapshot_index))
}

pub(crate) fn create_snapshot_recorder<P: AsRef<Path>>(
    path: P,
    interval: usize,
    overwrite: bool,
) -> Result<SnapshotRecorder, Box<dyn Error>> {
    Ok(SnapshotRecorder::create(path, interval, overwrite)?)
}

pub(crate) fn resume_snapshot_recorder<P: AsRef<Path>>(
    path: P,
    interval: usize,
    from_count: usize,
) -> Result<SnapshotRecorder, Box<dyn Error>> {
    Ok(SnapshotRecorder::resume(path, interval, from_count)?)
}
