use crate::activations::ActivationProvider;
use crate::io::h5;
use crate::model::{NeuralNetwork, NeuronLayer};
use hdf5_metno::Group;
use hdf5_metno::types::VarLenUnicode;
use ndarray::{Array1, Array2};
use std::io::ErrorKind::InvalidData;
use std::io::{Error, Result};
use std::path::{Path, PathBuf};

#[cfg(test)]
mod tests {
    use crate::activations::RELU;
    use crate::model::{NeuralNetwork, NeuronLayerSpec};
    use ndarray::Array2;

    #[test]
    fn save_load_roundtrip_predictions_are_identical() {
        let specs = NeuronLayerSpec::network_for(vec![4], &*RELU, 2);
        let model = NeuralNetwork::initialization(3, &specs);

        let inputs = Array2::from_shape_fn((3, 5), |(i, j)| (i * 5 + j) as f32 * 0.1);
        let predictions_before = model.predict(inputs.view());

        let path =
            std::path::PathBuf::from(format!("target/nrn_test_model_{}", std::process::id()));
        model.save(&path).unwrap();

        let loaded = NeuralNetwork::load(&path).unwrap();
        let predictions_after = loaded.predict(inputs.view());

        let _ = std::fs::remove_file(path.with_extension("h5"));

        assert_eq!(predictions_before, predictions_after);
    }
}

impl NeuralNetwork {
    /// Saves the neural network to an HDF5 group.
    /// # Arguments
    /// - `group`: The HDF5 group to save the network to.
    pub fn save_to(&self, group: &Group) -> Result<()> {
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
                .with_data(&layer.biases)
                .create("biases")?;
        }

        Ok(())
    }

    /// Reads a neural network from an HDF5 group.
    /// # Arguments
    /// - `group`: The HDF5 group to read the network from.
    pub fn load_from_group(group: &Group) -> Result<Self> {
        let n_layers = group
            .member_names()
            .map_err(Error::from)?
            .into_iter()
            .filter(|name| name.starts_with("layer"))
            .count();

        if n_layers == 0 {
            return Err(Error::new(InvalidData, "No layers found in the HDF5 group"));
        }

        let mut layers = Vec::with_capacity(n_layers);

        for i in 0..n_layers {
            let layer_group = group.group(&format!("layer{}", i))?;

            let weights: Array2<f32> = layer_group.dataset("weights")?.read()?;
            let biases: Array1<f32> = layer_group.dataset("biases")?.read()?;

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
                biases,
                activation,
            });
        }

        Ok(NeuralNetwork { layers })
    }

    /// Saves the current state of the neural network to an HDF5 file.
    /// # Arguments
    /// - `path`: The path to the file where the network will be saved.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<PathBuf> {
        let file = h5::create_file(path)?;
        self.save_to(&file)?;
        Ok(PathBuf::from(file.filename()))
    }

    /// Loads a neural network from an HDF5 file.
    /// # Arguments
    /// - `path`: The path to the file to load the network from.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = h5::load_file(path)?;
        NeuralNetwork::load_from_group(&file)
    }
}
