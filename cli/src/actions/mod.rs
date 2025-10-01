pub mod chart;
mod display;

use crate::{display_initialization, display_success, display_warning};
use colored::Colorize;
use console::style;
use nrn::data::scalers::{Scaler, ScalerMethod};
use nrn::data::{Dataset, SplitDataset};
use nrn::io::data::SplitDatasetExt;
use nrn::io::scalers::ScalerRecord;
use nrn::model::NeuralNetwork;
use nrn::training::History;
use std::error::Error;
use std::path::Path;

pub(crate) fn load_dataset<P: AsRef<Path>>(path: P) -> Result<SplitDataset, Box<dyn Error>> {
    let dataset = SplitDataset::load(path)?;

    println!(
        "[{}] Dataset loaded ({} features, {} training samples, {} test samples)",
        style("✔").green(),
        style(dataset.train.n_features()).yellow(),
        style(dataset.train.n_samples()).yellow(),
        style(dataset.test.n_samples()).yellow()
    );

    Ok(dataset)
}

pub(crate) fn save_dataset<P: AsRef<Path>>(
    dataset: &SplitDataset,
    path: P,
) -> Result<(), Box<dyn Error>> {
    display::saved_at(dataset.save(&path)?, "Dataset");
    Ok(())
}

pub(crate) fn save_scaler(scaler: ScalerMethod, filename: &str) -> Result<(), Box<dyn Error>> {
    let record: ScalerRecord = scaler.into();
    record.save(filename)?;

    display_success!(
        "{} at {} {}",
        "Scaler saved".bright_green(),
        filename.bright_blue().italic(),
        "(JSON)".italic().dimmed()
    );

    Ok(())
}

pub(crate) fn load_scaler(filename: &str) -> Result<ScalerMethod, Box<dyn Error>> {
    let scaler: ScalerMethod = ScalerRecord::load(filename)?.into();

    display_initialization!("Scaler loaded ({})", scaler.name().yellow());

    Ok(scaler)
}

/// Plots the dataset if it has exactly two features and saves the plot to a file.
pub(crate) fn plot_dataset(
    filename: &str,
    dataset: &Dataset,
    width: u32,
    height: u32,
) -> Result<(), Box<dyn Error>> {
    if dataset.n_features() == 2 {
        display::saved_at(chart::of_data(&dataset, width, height, &filename)?, "Plot");
    } else {
        display_warning!("Plotting is only available for datasets with exactly two features");
    }

    Ok(())
}

pub(crate) fn load_model(filename: &str) -> Result<NeuralNetwork, Box<dyn Error>> {
    let model = NeuralNetwork::load(filename)?;

    display_initialization!("Neural network loaded ({})", model.summary().yellow());

    Ok(model)
}

pub(crate) fn save_model(filename: &str, model: &NeuralNetwork) -> Result<(), Box<dyn Error>> {
    model.save(filename)?;

    display_success!(
        "{} at {} {}",
        "Model saved".bright_green(),
        filename.bright_blue().italic(),
        "(HDF5)".italic().dimmed()
    );

    Ok(())
}

pub(crate) fn load_training_history(filename: &str) -> Result<History, Box<dyn Error>> {
    let history = History::load(filename)?;

    assert!(
        history.model.len() > 2,
        "Training history must contain more than two checkpoints to plot."
    );

    display_initialization!(
        "Training history loaded ({} checkpoints, one every {} epochs)",
        history.model.len().to_string().yellow(),
        history.interval.to_string().yellow()
    );

    Ok(history)
}

pub(crate) fn save_training_history(
    filename: &str,
    history: &History,
) -> Result<(), Box<dyn Error>> {
    history.save(filename)?;

    display_success!(
        "{} at {} {}",
        "Training history saved".bright_green(),
        filename.bright_blue().italic(),
        "(HDF5)".italic().dimmed()
    );

    Ok(())
}
