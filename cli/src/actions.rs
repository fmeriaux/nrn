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

/// Tries to load a model and snapshot index from a snapshot directory
/// (`snapshot-{n}/model.safetensors` + `snapshot-{n}/evaluations.json`).
///
/// Falls back to loading the path as a plain model file if it is not a
/// snapshot directory, emitting:
/// - a `warning` when the path looks like a snapshot dir but cannot be parsed
///   (broken metadata — something unexpected)
/// - a `trace` otherwise (plain model file — intentional fallback)
///
/// Returns `(model, Some(snapshot_index))` on a successful snapshot load, or
/// `(model, None)` on fallback.
pub(crate) fn load_snapshot_or_model<P: AsRef<Path>>(
    path: P,
) -> Result<(NeuralNetwork, Option<usize>), Box<dyn Error>> {
    let path = path.as_ref();

    let snapshot_index = path
        .file_name()
        .and_then(|n| n.to_str())
        .and_then(|n| n.strip_prefix("snapshot-"))
        .and_then(|s| s.parse::<usize>().ok());

    let looks_like_snapshot = snapshot_index.is_some();

    if looks_like_snapshot
        && path.join("model.safetensors").exists()
        && path.join("evaluations.json").exists()
    {
        let model = load_model(path.join("model"))?;
        return Ok((model, snapshot_index));
    }

    // Fallback: load as a plain model file.
    if looks_like_snapshot {
        warning("Could not read snapshot metadata; starting count from 0.");
    } else {
        trace("No snapshot metadata found; starting count from 0.");
    }

    let model = load_model(path)?;
    Ok((model, None))
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
