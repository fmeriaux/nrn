use crate::io::h5;
use crate::model::NeuralNetwork;
use crate::training::History;
use std::io::Result;
use std::path::{Path, PathBuf};

impl History {
    /// Saves the training history to an HDF5 file.
    /// # Arguments
    /// - `path`: The path to the file where the history will be saved.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<PathBuf> {
        let file = h5::create_file(path)?;

        file.new_attr::<usize>()
            .create("interval")?
            .write_scalar(&self.interval)?;

        let metrics_group = file.create_group("metrics")?;

        metrics_group
            .new_dataset_builder()
            .with_data(&self.loss)
            .create("loss")?;

        metrics_group
            .new_dataset_builder()
            .with_data(&self.train_accuracy)
            .create("train_accuracy")?;

        metrics_group
            .new_dataset_builder()
            .with_data(&self.test_accuracy)
            .create("test_accuracy")?;

        let model_group = file.create_group("model")?;

        for (i, model) in self.model.iter().enumerate() {
            let checkpoint_group = model_group.create_group(&format!("checkpoint{}", i))?;
            model.save_to_group(&checkpoint_group)?;
        }

        Ok(PathBuf::from(file.filename()))
    }

    /// Loads the training history from an HDF5 file.
    /// # Arguments
    /// - `path`: The path to the file to load the history from.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = h5::load_file(path)?;

        let interval: usize = file.attr("interval")?.read_scalar()?;

        let metrics_group = file.group("metrics")?;

        let loss: Vec<f32> = metrics_group.dataset("loss")?.read()?.to_vec();
        let train_accuracy: Vec<f32> = metrics_group.dataset("train_accuracy")?.read()?.to_vec();
        let test_accuracy: Vec<f32> = metrics_group.dataset("test_accuracy")?.read()?.to_vec();

        let model_group = file.group("model")?;

        let mut model = Vec::new();

        loop {
            let checkpoint_name = format!("checkpoint{}", model.len());
            match model_group.group(&checkpoint_name) {
                Ok(checkpoint_group) => {
                    model.push(NeuralNetwork::load_from_group(&checkpoint_group)?);
                }
                Err(_) => break,
            }
        }

        Ok(History {
            interval,
            model,
            loss,
            train_accuracy,
            test_accuracy,
        })
    }
}
