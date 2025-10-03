use crate::io::path::PathExt;
use hdf5_metno::File;
use std::io::{Error, Result};
use std::path::Path;

pub fn create_file<P: AsRef<Path>>(path: P) -> Result<File> {
    let filepath = path.as_ref().with_extension("h5");
    let filepath = Path::combine_safe_with_cwd(filepath)?;
    filepath.create_parents()?;

    File::create(filepath).map_err(|e| Error::from(e))
}

pub fn load_file<P: AsRef<Path>>(path: P) -> Result<File> {
    let filepath = path.as_ref().with_extension("h5");
    let filepath = Path::combine_safe_with_cwd(filepath)?;
    File::open(filepath).map_err(|e| Error::from(e))
}
