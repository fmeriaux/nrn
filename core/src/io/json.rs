use std::fs;
use std::path::{Path, PathBuf};
use serde::Serialize;
use std::io::{Result, Error, ErrorKind::InvalidData};
use serde::de::DeserializeOwned;
use crate::io::path::PathExt;

pub fn save<T: Serialize, P: AsRef<Path>>(value: &T, path: P) -> Result<PathBuf> {
    let filepath = path.as_ref().with_extension("json");
    let filepath = Path::combine_safe_with_cwd(filepath)?;
    filepath.create_parents()?;

    let data = serde_json::to_vec(value).map_err(|e| Error::new(InvalidData, e))?;
    fs::write(&filepath, data)?;
    Ok(filepath)
}

pub fn load<T: DeserializeOwned, P: AsRef<Path>>(path: P) -> Result<T> {
    let filepath = path.as_ref().with_extension("json");
    let filepath = Path::combine_safe_with_cwd(filepath)?;
    let data = fs::read(&filepath)?;
    serde_json::from_slice(&data).map_err(|e| Error::new(InvalidData, e))
}