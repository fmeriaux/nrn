//! safetensors helpers shared by the model, dataset and checkpoint serializers.
//!
//! Tensors are stored as little-endian `f32`; auxiliary scalars and strings
//! (activation names, intervals, evaluations) travel in the `__metadata__`
//! string map. The format carries no `ndarray` types, so it is unaffected by
//! `ndarray` version bumps.

use crate::io::bytes::secure_read;
use crate::io::path::PathExt;
use ndarray::{Array, Array1, Array2, Dimension};
use safetensors::{Dtype, SafeTensors, View, serialize};
use std::borrow::Cow;
use std::collections::HashMap;
use std::fs;
use std::io::ErrorKind::InvalidData;
use std::io::{Error, Result};
use std::path::{Path, PathBuf};

/// File extension used for every safetensors artifact (models, datasets, checkpoints).
const EXTENSION: &str = "safetensors";

fn invalid<E: std::fmt::Display>(error: E) -> Error {
    Error::new(InvalidData, error.to_string())
}

/// Owned `f32` tensor ready to be serialized as a safetensors entry.
pub struct F32Tensor {
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

impl View for F32Tensor {
    fn dtype(&self) -> Dtype {
        Dtype::F32
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(&self.bytes)
    }
    fn data_len(&self) -> usize {
        self.bytes.len()
    }
}

/// Converts any `f32` array into a serializable tensor, copying it into a
/// standard (row-major) layout so the byte buffer matches the declared shape.
pub fn tensor<D: Dimension>(array: &Array<f32, D>) -> F32Tensor {
    let standard = array.as_standard_layout();
    let shape = standard.shape().to_vec();
    let data = standard
        .as_slice()
        .expect("as_standard_layout yields a contiguous array");

    let mut bytes = Vec::with_capacity(data.len() * 4);
    for value in data {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    F32Tensor { shape, bytes }
}

fn read_f32(
    name: &str,
    tensors: &SafeTensors,
    expected_rank: usize,
) -> Result<(Vec<usize>, Vec<f32>)> {
    let view = tensors.tensor(name).map_err(invalid)?;

    if view.dtype() != Dtype::F32 {
        return Err(invalid(format!("tensor `{name}` is not f32")));
    }
    if view.shape().len() != expected_rank {
        return Err(invalid(format!(
            "tensor `{name}` has rank {}, expected {expected_rank}",
            view.shape().len()
        )));
    }

    let data = view
        .data()
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    Ok((view.shape().to_vec(), data))
}

/// Reads a 1D `f32` tensor by name.
pub fn read_array1(name: &str, tensors: &SafeTensors) -> Result<Array1<f32>> {
    let (shape, data) = read_f32(name, tensors, 1)?;
    Array1::from_shape_vec(shape[0], data).map_err(invalid)
}

/// Reads a 2D `f32` tensor by name.
pub fn read_array2(name: &str, tensors: &SafeTensors) -> Result<Array2<f32>> {
    let (shape, data) = read_f32(name, tensors, 2)?;
    Array2::from_shape_vec((shape[0], shape[1]), data).map_err(invalid)
}

/// Looks up a required entry in the `__metadata__` map.
pub fn meta<'a>(metadata: &'a HashMap<String, String>, key: &str) -> Result<&'a str> {
    metadata
        .get(key)
        .map(String::as_str)
        .ok_or_else(|| invalid(format!("missing metadata key `{key}`")))
}

/// Serializes named tensors and a metadata map to a `.safetensors` file,
/// resolving the path safely (no traversal) and creating parent directories.
pub fn save<P: AsRef<Path>>(
    path: P,
    tensors: Vec<(String, F32Tensor)>,
    metadata: HashMap<String, String>,
) -> Result<PathBuf> {
    let filepath = path.as_ref().with_extension(EXTENSION);
    let filepath = Path::combine_safe_with_cwd(filepath)?;
    filepath.create_parents()?;

    let bytes = serialize(tensors, Some(metadata)).map_err(invalid)?;
    fs::write(&filepath, bytes)?;
    Ok(filepath)
}

/// Reads the raw bytes of a `.safetensors` file (path-safe).
///
/// The caller owns the buffer and borrows it through [`SafeTensors::deserialize`]
/// and [`read_metadata`], which must outlive their views.
pub fn load<P: AsRef<Path>>(path: P) -> Result<Vec<u8>> {
    secure_read(path.as_ref().with_extension(EXTENSION))
}

/// Extracts the `__metadata__` string map from a safetensors buffer.
pub fn read_metadata(bytes: &[u8]) -> Result<HashMap<String, String>> {
    let (_, metadata) = SafeTensors::read_metadata(bytes).map_err(invalid)?;
    Ok(metadata.metadata().clone().unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn read_metadata_rejects_corrupt_bytes() {
        assert!(read_metadata(b"not a safetensors header").is_err());
    }

    #[test]
    fn read_rejects_missing_tensor() {
        let bytes = serialize(
            vec![("present".to_string(), tensor(&array![1.0_f32, 2.0]))],
            None,
        )
        .unwrap();
        let st = SafeTensors::deserialize(&bytes).unwrap();

        assert!(read_array1("absent", &st).is_err());
    }

    #[test]
    fn read_rejects_wrong_rank() {
        // A 2D tensor read back as 1D must fail the rank check.
        let bytes = serialize(
            vec![("matrix".to_string(), tensor(&array![[1.0_f32, 2.0]]))],
            None,
        )
        .unwrap();
        let st = SafeTensors::deserialize(&bytes).unwrap();

        assert!(read_array1("matrix", &st).is_err());
    }

    #[test]
    fn read_rejects_non_f32_dtype() {
        // A tensor stored with a non-f32 dtype must be refused on read.
        struct U8Tensor {
            shape: Vec<usize>,
        }
        impl View for U8Tensor {
            fn dtype(&self) -> Dtype {
                Dtype::U8
            }
            fn shape(&self) -> &[usize] {
                &self.shape
            }
            fn data(&self) -> Cow<'_, [u8]> {
                Cow::Owned(vec![0])
            }
            fn data_len(&self) -> usize {
                1
            }
        }

        let bytes = serialize(
            vec![("byte".to_string(), U8Tensor { shape: vec![1] })],
            None,
        )
        .unwrap();
        let st = SafeTensors::deserialize(&bytes).unwrap();

        assert!(read_array1("byte", &st).is_err());
    }

    #[test]
    fn meta_reports_missing_key() {
        let metadata = HashMap::new();
        assert!(meta(&metadata, "interval").is_err());
    }
}
