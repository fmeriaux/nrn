use crate::activations::ActivationProvider;
use crate::model::{NeuralNetwork, NeuronLayer};
use crate::io::h5;
use hdf5_metno::Group;
use hdf5_metno::types::VarLenUnicode;
use ndarray::{Array1, Array2};
use std::io::ErrorKind::InvalidData;
use std::io::{Error, Result};
use std::path::Path;

impl NeuralNetwork {
    /// Saves the neural network to an HDF5 group.
    /// # Arguments
    /// - `group`: The HDF5 group to save the network to.
    pub fn save_to_group(&self, group: &Group) -> Result<()> {
        for (i, layer) in self.layers.iter().enumerate() {
            let group = group.create_group(&format!("layer{}", i))?;

            group
                .new_attr::<VarLenUnicode>()
                .create("activation")?
                .write_scalar(
                    &layer
                        .activation
                        .name()
                        .parse::<VarLenUnicode>()
                        .map_err(|e| Error::new(InvalidData, e))?,
                )?;

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
    pub fn load_from_group(group: &Group) -> Result<Self> {
        let mut layers = Vec::new();

        loop {
            let layer_name = format!("layer{}", layers.len());
            match group.group(&layer_name) {
                Ok(layer_group) => {
                    let weights: Array2<f32> = layer_group.dataset("weights")?.read()?;
                    let bias: Array1<f32> = layer_group.dataset("bias")?.read()?;

                    let activation_name: String = layer_group
                        .attr("activation")?
                        .read_scalar::<VarLenUnicode>()?
                        .as_str()
                        .to_string();

                    let activation =
                        ActivationProvider::get_by_name(&activation_name).ok_or_else(|| {
                            Error::new(
                                InvalidData,
                                format!("Unknown activation function: {}", activation_name),
                            )
                        })?;

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
            return Err(Error::new(InvalidData, "No layers found in the HDF5 group"));
        }

        Ok(NeuralNetwork { layers })
    }

    /// Saves the current state of the neural network to an HDF5 file.
    /// # Arguments
    /// - `path`: The path to the file where the network will be saved.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = h5::create_file(path)?;
        self.save_to_group(&file)
    }

    /// Loads a neural network from an HDF5 file.
    /// # Arguments
    /// - `path`: The path to the file to load the network from.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = h5::load_file(path)?;
        Ok(NeuralNetwork::load_from_group(&file)?)
    }
}
