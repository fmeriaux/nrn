use crate::core::activation::ActivationMethod;
use crate::core::neuron_network::{NeuronLayer, NeuronNetwork};
use crate::core::training_history::TrainingHistory;
use crate::synth::{Dataset, SplitDataset};
use hdf5_metno::types::VarLenUnicode;
use hdf5_metno::{File, Group};
use ndarray::{Array1, Array2};
use std::error::Error;
use std::path::Path;

const EXTENSION: &'static str = "h5";

pub fn save_inputs<P: AsRef<Path>>(file: P, inputs: &Array1<f32>) -> Result<(), Box<dyn Error>> {
    let file = File::create(file.as_ref().with_extension(EXTENSION))?;

    file.new_dataset_builder()
        .with_data(inputs)
        .create("inputs")?;

    Ok(())
}

pub fn load_inputs<P: AsRef<Path>>(file: P) -> Result<Array1<f32>, Box<dyn Error>> {
    let file = File::open(file.as_ref().with_extension(EXTENSION))?;
    let inputs: Array1<f32> = file.dataset("inputs")?.read()?;
    Ok(inputs)
}

impl NeuronNetwork {
    /// Saves the neural network to an HDF5 group.
    /// # Arguments
    /// - `group`: The HDF5 group to save the network to.
    fn save_to_group(&self, group: &Group) -> Result<(), Box<dyn Error>> {
        for (i, layer) in self.layers.iter().enumerate() {
            let group = group.create_group(&format!("layer{}", i))?;

            group
                .new_attr::<VarLenUnicode>()
                .create("activation")?
                .write_scalar(&layer.activation.to_string().parse::<VarLenUnicode>()?)?;

            group
                .new_dataset_builder()
                .with_data(&layer.weights)
                .create("weights")?;

            group
                .new_dataset_builder()
                .with_data(&layer.bias)
                .create("bias")?;
        }

        Ok(())
    }

    /// Reads a neural network from an HDF5 group.
    /// # Arguments
    /// - `group`: The HDF5 group to read the network from.
    fn load_from_group(group: &Group) -> Result<Self, Box<dyn Error>> {
        let mut layers = Vec::new();

        loop {
            let layer_name = format!("layer{}", layers.len());
            match group.group(&layer_name) {
                Ok(layer_group) => {
                    let weights: Array2<f32> = layer_group.dataset("weights")?.read()?;
                    let bias: Array1<f32> = layer_group.dataset("bias")?.read()?;
                    let activation: ActivationMethod = layer_group
                        .attr("activation")?
                        .read_scalar::<VarLenUnicode>()?
                        .as_str()
                        .parse::<ActivationMethod>()?;

                    layers.push(NeuronLayer {
                        weights,
                        bias,
                        activation,
                    });
                }
                Err(_) => break,
            }
        }

        if layers.is_empty() {
            return Err("No layers found in the HDF5 group".into());
        }

        Ok(NeuronNetwork { layers })
    }

    /// Saves the current state of the neural network to an HDF5 file.
    /// # Arguments
    /// - `file`: The path to the file where the network will be saved.
    pub fn save<P: AsRef<Path>>(&self, file: P) -> Result<(), Box<dyn Error>> {
        let file = File::create(file.as_ref().with_extension(EXTENSION))?;
        self.save_to_group(&file)
    }

    /// Loads a neural network from an HDF5 file.
    /// # Arguments
    /// - `file`: The path to the file to load the network from.
    pub fn load<P: AsRef<Path>>(file: P) -> Result<Self, Box<dyn Error>> {
        let file = File::open(file.as_ref().with_extension(EXTENSION))?;
        Ok(NeuronNetwork::load_from_group(&file)?)
    }
}

impl TrainingHistory {
    /// Saves the training history to an HDF5 file.
    /// # Arguments
    /// - `file`: The path to the file where the history will be saved.
    pub fn save<P: AsRef<Path>>(&self, file: P) -> Result<(), Box<dyn Error>> {
        let file = File::create(file.as_ref().with_extension(EXTENSION))?;

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
            NeuronNetwork::save_to_group(model, &checkpoint_group)?;
        }

        Ok(())
    }

    /// Loads the training history from an HDF5 file.
    /// # Arguments
    /// - `file`: The path to the file to load the history from.
    pub fn load<P: AsRef<Path>>(file: P) -> Result<Self, Box<dyn Error>> {
        let file = File::open(file.as_ref().with_extension(EXTENSION))?;

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
                    model.push(NeuronNetwork::load_from_group(&checkpoint_group)?);
                }
                Err(_) => break,
            }
        }

        Ok(TrainingHistory {
            interval,
            model,
            loss,
            train_accuracy,
            test_accuracy,
        })
    }
}

impl Dataset {
    /// Writes the dataset to an HDF5 group.
    /// # Arguments
    /// - `group`: The HDF5 group to write the dataset to.
    fn write_dataset(&self, group: &Group) -> Result<(), Box<dyn Error>> {
        group
            .new_dataset::<f32>()
            .shape(self.features.dim())
            .create("features")?
            .write(&self.features)?;

        group
            .new_dataset::<f32>()
            .shape(&[self.labels.len()])
            .create("labels")?
            .write(&self.labels)?;

        Ok(())
    }

    /// Reads a dataset from an HDF5 group.
    /// # Arguments
    /// - `group`: The HDF5 group to read the dataset from.
    fn read_dataset(group: &Group) -> Result<Dataset, Box<dyn Error>> {
        let features: Array2<f32> = group.dataset("features")?.read()?;
        let labels: Array1<f32> = group.dataset("labels")?.read()?;

        Ok(Dataset { features, labels })
    }
}

impl SplitDataset {
    /// Saves the split dataset to an HDF5 file.
    /// # Arguments
    /// - `file`: The path to the file where the dataset will be saved.
    pub fn save<P: AsRef<Path>>(&self, file: P) -> Result<(), Box<dyn Error>> {
        let file = File::create(file.as_ref().with_extension(EXTENSION))?;

        for (usage, dataset) in self.groups().iter() {
            let group = file.create_group(usage)?;
            dataset.write_dataset(&group)?;
        }

        Ok(())
    }

    /// Loads a split dataset from an HDF5 file.
    /// # Arguments
    /// - `file`: The file path to load the dataset from.
    pub fn load<P: AsRef<Path>>(file: P) -> Result<SplitDataset, Box<dyn Error>> {
        let file = File::open(file.as_ref().with_extension(EXTENSION))?;

        let train = Dataset::read_dataset(&file.group("train")?)?;
        let test = Dataset::read_dataset(&file.group("test")?)?;

        Ok(SplitDataset { train, test })
    }
}
