use std::{env, fs};
use std::io::ErrorKind::PermissionDenied;
use std::io::{Error, Result};
use std::path::{Path, PathBuf};

/// Extension trait for adding secure path manipulation methods to `Path`.
pub trait PathExt {
    /// Combines `self` with `user_path`, resolving the absolute path,
    /// normalizing, and validating no directory traversal outside `self`.
    /// Returns an error if the result is outside the base path.
    fn combine_safe<P: AsRef<Path>>(&self, user_path: P) -> Result<PathBuf>;

    /// Ensures all parent directories of `self` exist,
    /// creating them if necessary.
    fn create_parents(&self) -> Result<()>;

    /// Combines the current working directory with `user_path`, resolving the absolute path,
    /// normalizing, and validating no directory traversal outside the current directory.
    /// Returns an error if the result is outside the current working directory.
    fn combine_safe_with_cwd<P: AsRef<Path>>(user_path: P) -> Result<PathBuf>;
}

impl PathExt for Path {
    fn combine_safe<P: AsRef<Path>>(&self, user_path: P) -> Result<PathBuf> {
        let base_path = fs::canonicalize(self)?;
        let combined = self.join(user_path.as_ref());

        // Find the nearest existing parent directory
        let mut current = combined.as_path();
        let mut segments = Vec::new();
        while !current.exists() {
            if let Some(parent) = current.parent() {
                if let Some(name) = current.file_name() {
                    segments.push(name.to_os_string());
                }
                current = parent;
            } else {
                return Err(Error::new(PermissionDenied, "No existing parent directory found for path traversal check"));
            }
        }
        // Canonicalize the nearest existing parent directory
        let mut canonical = fs::canonicalize(current)?;
        // Reapply the non-existing segments in reverse order
        for segment in segments.iter().rev() {
            canonical = canonical.join(segment);
        }
        // Validate that the final path is within the base path
        if !canonical.starts_with(&base_path) {
            return Err(Error::new(
                PermissionDenied,
                "Attempted directory traversal detected",
            ));
        }
        Ok(canonical)
    }

    fn create_parents(&self) -> Result<()> {
        if let Some(parent) = self.parent() {
            fs::create_dir_all(parent)?;
        }
        Ok(())
    }

    fn combine_safe_with_cwd<P: AsRef<Path>>(user_path: P) -> Result<PathBuf> {
        let cwd = env::current_dir()?;
        cwd.combine_safe(user_path)
    }
}
