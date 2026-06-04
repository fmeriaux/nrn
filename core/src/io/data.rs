use crate::data::Dataset;
use crate::io::tensors;
use ndarray::Array1;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::io::ErrorKind::InvalidData;
use std::io::{Error, Result};
use std::path::{Path, PathBuf};

pub fn save_inputs<P: AsRef<Path>>(path: P, inputs: &Array1<f32>) -> Result<()> {
    let entries = vec![("inputs".to_string(), tensors::tensor(inputs))];
    tensors::save(path, entries, HashMap::new())?;
    Ok(())
}

pub fn load_inputs<P: AsRef<Path>>(path: P) -> Result<Array1<f32>> {
    let bytes = tensors::load(path)?;
    let st =
        SafeTensors::deserialize(&bytes).map_err(|e| Error::new(InvalidData, e.to_string()))?;
    tensors::read_array1("inputs", &st)
}

impl Dataset {
    /// Saves the dataset to a `.safetensors` file.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<PathBuf> {
        let entries = vec![
            ("features".to_string(), tensors::tensor(&self.features)),
            ("labels".to_string(), tensors::tensor(&self.labels)),
        ];
        tensors::save(path, entries, HashMap::new())
    }

    /// Loads a dataset from a `.safetensors` file.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Dataset> {
        let bytes = tensors::load(path)?;
        let st =
            SafeTensors::deserialize(&bytes).map_err(|e| Error::new(InvalidData, e.to_string()))?;

        let features = tensors::read_array2("features", &st)?;
        let labels = tensors::read_array1("labels", &st)?;

        Ok(Dataset { features, labels })
    }
}

#[cfg(test)]
mod tests {
    use super::{load_inputs, save_inputs};
    use crate::data::Dataset;
    use ndarray::{Array2, array};
    use std::path::{Path, PathBuf};

    fn temp_path(tag: &str) -> PathBuf {
        let dir = PathBuf::from("target/nrn_tests");
        std::fs::create_dir_all(&dir).ok();
        dir.join(format!("nrn_test_{tag}_{}", std::process::id()))
    }

    fn cleanup(path: &Path) {
        let _ = std::fs::remove_file(path.with_extension("safetensors"));
    }

    #[test]
    fn dataset_save_load_roundtrip() {
        let dataset = Dataset {
            features: Array2::from_shape_fn((3, 5), |(i, j)| (i * 5 + j) as f32 * 0.25),
            labels: array![0.0, 1.0, 0.0, 2.0, 1.0],
        };

        let path = temp_path("dataset");
        dataset.save(&path).unwrap();
        let loaded = Dataset::load(&path).unwrap();
        cleanup(&path);

        assert_eq!(dataset.features, loaded.features);
        assert_eq!(dataset.labels, loaded.labels);
    }

    #[test]
    fn inputs_save_load_roundtrip() {
        let inputs = array![0.1, -0.2, 3.5, 42.0];

        let path = temp_path("inputs");
        save_inputs(&path, &inputs).unwrap();
        let loaded = load_inputs(&path).unwrap();
        cleanup(&path);

        assert_eq!(inputs, loaded);
    }

    #[test]
    fn load_rejects_corrupt_file() {
        let path = temp_path("data_corrupt");
        std::fs::write(
            path.with_extension("safetensors"),
            b"not a safetensors buffer",
        )
        .unwrap();

        assert!(Dataset::load(&path).is_err());
        assert!(load_inputs(&path).is_err());

        cleanup(&path);
    }
}
