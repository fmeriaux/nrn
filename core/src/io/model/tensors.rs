//! safetensors helpers shared by the model and optimizer serializers.
//!
//! Tensors are stored as little-endian `f32`; the optimizer's scalar state
//! (e.g. Adam's time step) travels in the `__metadata__` string map. The format
//! carries no `ndarray` types, so it is unaffected by `ndarray` version bumps.

use crate::io::bytes::secure_read;
use crate::io::path::PathExt;
use ndarray::{Array, ArrayD, Dimension, IxDyn};
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

/// Reads a tensor of arbitrary rank as an `ArrayD<f32>`, e.g. for optimizer
/// state tensors whose rank depends on the layer they belong to.
pub fn read_arrayd<V: View>(view: &V) -> Result<ArrayD<f32>> {
    if view.dtype() != Dtype::F32 {
        return Err(invalid("tensor is not f32"));
    }

    let data: Vec<f32> = view
        .data()
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    ArrayD::from_shape_vec(IxDyn(view.shape()), data).map_err(invalid)
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

    // `read_metadata`'s guard has no caller that reaches it on corrupt bytes
    // (every caller gets here after a successful `deserialize`), so it is tested
    // directly here.
    #[test]
    fn read_metadata_rejects_corrupt_bytes() {
        assert!(read_metadata(b"not a safetensors header").is_err());
    }
}
