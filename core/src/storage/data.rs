use crate::storage::h5;
use hdf5_metno::Group;
use ndarray::{Array1, Array2};
use crate::data::{Dataset, SplitDataset};
use std::io::Result;
use std::path::Path;

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

pub trait DatasetExt {
    fn save_to_group(&self, group: &Group) -> Result<()>;
    fn load_from_group(group: &Group) -> Result<Dataset>;
}

impl DatasetExt for Dataset {
    /// Writes the dataset to an HDF5 group.
    /// # Arguments
    /// - `group`: The HDF5 group to write the dataset to.
    fn save_to_group(&self, group: &Group) -> Result<()> {
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
    fn load_from_group(group: &Group) -> Result<Dataset> {
        let features: Array2<f32> = group.dataset("features")?.read()?;
        let labels: Array1<f32> = group.dataset("labels")?.read()?;

        Ok(Dataset { features, labels })
    }
}

pub trait SplitDatasetExt {
    fn save<P: AsRef<Path>>(&self, path: P) -> Result<()>;
    fn load<P: AsRef<Path>>(path: P) -> Result<SplitDataset>;
}

impl SplitDatasetExt for SplitDataset {
    /// Saves the split dataset to an HDF5 file.
    /// # Arguments
    /// - `path`: The path to the file where the dataset will be saved.
    fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = h5::create_file(path)?;

        for (usage, dataset) in self.groups().iter() {
            let group = file.create_group(usage)?;
            dataset.save_to_group(&group)?;
        }

        Ok(())
    }

    /// Loads a split dataset from an HDF5 file.
    /// # Arguments
    /// - `path`: The file path to load the dataset from.
    fn load<P: AsRef<Path>>(path: P) -> Result<SplitDataset> {
        let file = h5::load_file(path)?;

        let train = Dataset::load_from_group(&file.group("train")?)?;
        let test = Dataset::load_from_group(&file.group("test")?)?;

        Ok(SplitDataset { train, test })
    }
}
