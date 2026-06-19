//! Small `Path` conveniences shared across the CLI.

use pathdiff::diff_paths;
use std::env;
use std::path::{Path, PathBuf};

/// Path conveniences used by CLI commands and console display alike.
pub(crate) trait PathExt {
    /// This path expressed relative to the current directory, falling back to
    /// the path as-is when that can't be determined.
    fn to_relative(&self) -> PathBuf;

    /// The file stem as an owned `String`. Panics if the path has none.
    fn file_stem_string(&self) -> String;

    /// A sibling path in the same directory, named `{prefix}-{stem}` from this
    /// path's file stem. Panics if the path has no stem.
    fn sibling(&self, prefix: &str) -> PathBuf;
}

impl PathExt for Path {
    fn to_relative(&self) -> PathBuf {
        env::current_dir()
            .ok()
            .and_then(|cwd| diff_paths(self, cwd))
            .unwrap_or_else(|| self.to_path_buf())
    }

    fn file_stem_string(&self) -> String {
        self.file_stem()
            .unwrap_or_else(|| panic!("Failed to get file stem from path: {}", self.display()))
            .to_string_lossy()
            .to_string()
    }

    fn sibling(&self, prefix: &str) -> PathBuf {
        self.with_file_name(format!("{prefix}-{}", self.file_stem_string()))
    }
}
