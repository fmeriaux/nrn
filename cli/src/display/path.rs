//! Small `Path` convenience for display.

use pathdiff::diff_paths;
use std::env;
use std::path::{Path, PathBuf};

/// Path convenience used by CLI display.
pub(crate) trait PathExt {
    /// This path expressed relative to the current directory, falling back to
    /// the path as-is when that can't be determined.
    fn to_relative(&self) -> PathBuf;
}

impl PathExt for Path {
    fn to_relative(&self) -> PathBuf {
        env::current_dir()
            .ok()
            .and_then(|cwd| diff_paths(self, cwd))
            .unwrap_or_else(|| self.to_path_buf())
    }
}
