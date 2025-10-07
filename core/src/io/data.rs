use crate::data::Dataset;
use crate::io::h5;
use ndarray::{Array1, Array2};
use std::io::Result;
use std::path::{Path, PathBuf};

pub fn save_inputs<P: AsRef<Path>>(path: P, inputs: &Array1<f32>) -> Result<()> {
    let file = h5::create_file(path)?;

    file.new_dataset_builder()
        .with_data(inputs)
        .create("inputs")?;

    Ok(())
}

pub fn load_inputs<P: AsRef<Path>>(path: P) -> Result<Array1<f32>> {
    let file = h5::load_file(path)?;
    let inputs: Array1<f32> = file.dataset("inputs")?.read()?;
    Ok(inputs)
}

impl Dataset {
    /// Saves the dataset to an HDF5 file.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<PathBuf> {
        let file = h5::create_file(path)?;

        file.new_dataset::<f32>()
            .shape(self.features.dim())
            .create("features")?
            .write(&self.features)?;

        file.new_dataset::<f32>()
            .shape(&[self.labels.len()])
            .create("labels")?
            .write(&self.labels)?;

        Ok(PathBuf::from(file.filename()))
    }

    /// Loads a dataset from an HDF5 file.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Dataset> {
        let file = h5::load_file(path)?;

        let features: Array2<f32> = file.dataset("features")?.read()?;
        let labels: Array1<f32> = file.dataset("labels")?.read()?;

        Ok(Dataset { features, labels })
    }
}
