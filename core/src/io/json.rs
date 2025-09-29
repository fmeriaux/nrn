use std::fs;
use std::path::Path;
use serde::Serialize;
use std::io::{Result, Error, ErrorKind::InvalidData};
use serde::de::DeserializeOwned;
use crate::io::path::PathExt;

pub fn save<T: Serialize, P: AsRef<Path>>(value: &T, path: P) -> Result<()> {
    let file_path = path.as_ref().with_extension("json");
    let file_path = Path::combine_safe_with_cwd(file_path)?;
    file_path.create_parents()?;

    let data = serde_json::to_vec(value).map_err(|e| Error::new(InvalidData, e))?;
    fs::write(file_path, data)
}

pub fn load<T: DeserializeOwned, P: AsRef<Path>>(path: P) -> Result<T> {
    let file_path = path.as_ref().with_extension("json");
    let file_path = Path::combine_safe_with_cwd(file_path)?;
    let data = fs::read(&file_path)?;
    serde_json::from_slice(&data).map_err(|e| Error::new(InvalidData, e))
}