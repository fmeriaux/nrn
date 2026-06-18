use crate::io::tensors;
use crate::optimizers::OptimizerState;
use safetensors::SafeTensors;
use std::io::ErrorKind::InvalidData;
use std::io::{Error, Result};
use std::path::{Path, PathBuf};

/// Saves an [`OptimizerState`] to a `.safetensors` file: its tensors as
/// entries, its metadata as the `__metadata__` string map.
pub fn save<P: AsRef<Path>>(state: &OptimizerState, path: P) -> Result<PathBuf> {
    let entries = state
        .tensors
        .iter()
        .map(|(name, array)| (name.clone(), tensors::tensor(array)))
        .collect();
    tensors::save(path, entries, state.metadata.clone())
}

/// Loads an [`OptimizerState`] previously written by [`save`].
pub fn load<P: AsRef<Path>>(path: P) -> Result<OptimizerState> {
    let bytes = tensors::load(path)?;
    let st =
        SafeTensors::deserialize(&bytes).map_err(|e| Error::new(InvalidData, e.to_string()))?;
    let metadata = tensors::read_metadata(&bytes)?;

    let tensors = st
        .tensors()
        .into_iter()
        .map(|(name, view)| Ok((name, tensors::read_arrayd(&view)?)))
        .collect::<Result<Vec<_>>>()?;

    Ok(OptimizerState { tensors, metadata })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use std::collections::HashMap;
    use std::path::PathBuf;

    fn temp_path(tag: &str) -> PathBuf {
        PathBuf::from(format!(
            "target/nrn_test_optimizer_{tag}_{}",
            std::process::id()
        ))
    }

    fn cleanup(path: &Path) {
        let _ = std::fs::remove_file(path.with_extension("safetensors"));
    }

    #[test]
    fn save_load_roundtrip_preserves_tensors_and_metadata() {
        let state = OptimizerState {
            tensors: vec![
                (
                    "layer0.m_weights".to_string(),
                    array![[1.0_f32, 2.0], [3.0, 4.0]].into_dyn(),
                ),
                ("layer0.m_biases".to_string(), array![0.5_f32].into_dyn()),
            ],
            metadata: HashMap::from([("time_step".to_string(), "5".to_string())]),
        };

        let path = temp_path("roundtrip");
        save(&state, &path).unwrap();
        let loaded = load(&path).unwrap();
        cleanup(&path);

        assert_eq!(
            loaded.metadata.get("time_step").map(String::as_str),
            Some("5")
        );
        assert_eq!(loaded.tensors.len(), 2);
        let m_weights = loaded
            .tensors
            .iter()
            .find(|(name, _)| name == "layer0.m_weights")
            .map(|(_, array)| array)
            .unwrap();
        assert_eq!(m_weights, &array![[1.0_f32, 2.0], [3.0, 4.0]].into_dyn());
    }

    #[test]
    fn load_rejects_missing_file() {
        // No file is written: the read itself must fail, before any parsing.
        let path = temp_path("absent");
        assert!(load(&path).is_err());
    }

    #[test]
    fn load_rejects_corrupt_file() {
        let path = temp_path("corrupt");
        std::fs::write(path.with_extension("safetensors"), b"not safetensors").unwrap();

        assert!(load(&path).is_err());
        cleanup(&path);
    }

    #[test]
    fn load_rejects_non_f32_tensor() {
        use safetensors::{Dtype, View, serialize};
        use std::borrow::Cow;

        // A state file whose tensor is not f32 must be refused: the dtype guard in
        // `read_arrayd` rejects it as the entries are read back.
        struct U8Tensor;
        impl View for U8Tensor {
            fn dtype(&self) -> Dtype {
                Dtype::U8
            }
            fn shape(&self) -> &[usize] {
                &[1]
            }
            fn data(&self) -> Cow<'_, [u8]> {
                Cow::Owned(vec![0])
            }
            fn data_len(&self) -> usize {
                1
            }
        }

        let bytes = serialize(vec![("layer0.m_weights".to_string(), U8Tensor)], None).unwrap();
        let path = temp_path("non_f32");
        std::fs::write(path.with_extension("safetensors"), bytes).unwrap();

        assert!(load(&path).is_err());
        cleanup(&path);
    }
}
