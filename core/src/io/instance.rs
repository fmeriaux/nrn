//! safetensors serializer for an inference [`Instance`]: its feature vector
//! stored under a single `instance` tensor.

use crate::data::Instance;
use crate::io::tensors;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::io::ErrorKind::InvalidData;
use std::io::{Error, Result};
use std::path::{Path, PathBuf};

/// safetensors tensor name for an inference instance.
const INSTANCE: &str = "instance";

fn invalid<E: std::fmt::Display>(error: E) -> Error {
    Error::new(InvalidData, error.to_string())
}

impl Instance {
    /// Saves the instance's feature vector to a `.safetensors` file, returning the
    /// written path.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<PathBuf> {
        let entries = vec![(INSTANCE.to_string(), tensors::tensor(self.values()))];
        tensors::save(path, entries, HashMap::new())
    }

    /// Loads an instance from a `.safetensors` file written by [`Instance::save`].
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Instance> {
        let bytes = tensors::load(path)?;
        let st = SafeTensors::deserialize(&bytes).map_err(invalid)?;
        Ok(Instance::new(tensors::read_array1(INSTANCE, &st)?))
    }
}

#[cfg(test)]
mod tests {
    use crate::data::Instance;
    use ndarray::array;
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
    fn instance_save_load_roundtrip() {
        let instance = Instance::new(array![0.1, -0.2, 3.5, 42.0]);

        let path = temp_path("instance");
        instance.save(&path).unwrap();
        let loaded = Instance::load(&path).unwrap();
        cleanup(&path);

        assert_eq!(instance, loaded);
    }

    #[test]
    fn load_rejects_corrupt_file() {
        let path = temp_path("instance_corrupt");
        std::fs::write(
            path.with_extension("safetensors"),
            b"not a safetensors buffer",
        )
        .unwrap();

        assert!(Instance::load(&path).is_err());

        cleanup(&path);
    }
}
