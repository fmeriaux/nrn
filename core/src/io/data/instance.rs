//! JSON serializer for an inference [`Instance`]: its feature vector stored
//! under a single `instance` field.

use crate::data::Instance;
use crate::io::json;
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::io::Result;
use std::path::{Path, PathBuf};

/// Serializable mirror of an [`Instance`]'s feature vector.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct InstanceRecord {
    pub instance: Array1<f32>,
}

impl Instance {
    /// Saves the instance's feature vector to a `.json` file, returning the
    /// written path.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<PathBuf> {
        let record = InstanceRecord {
            instance: self.values().clone(),
        };
        json::save(&record, path)
    }

    /// Loads an instance from a `.json` file written by [`Instance::save`].
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Instance> {
        let record: InstanceRecord = json::load(path)?;
        Ok(Instance::new(record.instance))
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
        let _ = std::fs::remove_file(path.with_extension("json"));
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
        std::fs::write(path.with_extension("json"), b"not a json instance").unwrap();

        assert!(Instance::load(&path).is_err());

        cleanup(&path);
    }
}
