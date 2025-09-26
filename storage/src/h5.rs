use crate::path::PathExt;
use hdf5_metno::File;
use std::io::{Error, Result};
use std::path::Path;

pub fn create_file<P: AsRef<Path>>(path: P) -> Result<File> {
    let file_path = path.as_ref().with_extension("h5");
    let file_path = Path::combine_safe_with_cwd(file_path)?;
    file_path.create_parents()?;

    File::create(file_path).map_err(|e| Error::from(e))
}

pub fn load_file<P: AsRef<Path>>(path: P) -> Result<File> {
    let file_path = path.as_ref().with_extension("h5");
    let file_path = Path::combine_safe_with_cwd(file_path)?;
    File::open(file_path).map_err(|e| Error::from(e))
}
