use std::collections::BTreeSet;
use std::io::ErrorKind::PermissionDenied;
use std::io::{Error, Result};
use std::path::{Path, PathBuf};
use std::{env, fs};

/// Extension trait adding path manipulation conveniences to `Path`.
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

    /// Returns `self` if a sidecar file exists at `self` with `ext` appended, `None` otherwise —
    /// for optional files whose presence gates whether they get loaded at all.
    fn optional_sidecar(&self, ext: &str) -> Option<PathBuf>;

    /// The file stem as an owned `String`.
    ///
    /// # Panics
    /// Panics if the path has no stem.
    fn file_stem_string(&self) -> String;

    /// A sibling path in the same directory, named `{prefix}-{stem}` from this
    /// path's file stem.
    ///
    /// # Panics
    /// Panics if the path has no stem.
    fn sibling(&self, prefix: &str) -> PathBuf;

    /// Scans `self` for its immediate subdirectories, returning their names
    /// sorted alphabetically. Files are ignored.
    fn scan_dir(&self) -> Result<Vec<String>>;
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
                return Err(Error::new(
                    PermissionDenied,
                    "No existing parent directory found for path traversal check",
                ));
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

    fn optional_sidecar(&self, ext: &str) -> Option<PathBuf> {
        self.with_extension(ext)
            .exists()
            .then(|| self.to_path_buf())
    }

    fn file_stem_string(&self) -> String {
        self.file_stem()
            .unwrap_or_else(|| panic!("Failed to get file stem from path: {}.", self.display()))
            .to_string_lossy()
            .to_string()
    }

    fn sibling(&self, prefix: &str) -> PathBuf {
        self.with_file_name(format!("{prefix}-{}", self.file_stem_string()))
    }

    fn scan_dir(&self) -> Result<Vec<String>> {
        let names: BTreeSet<String> = fs::read_dir(Path::combine_safe_with_cwd(self)?)?
            .filter_map(std::result::Result::ok)
            .filter(|entry| entry.path().is_dir())
            .filter_map(|entry| entry.file_name().to_str().map(str::to_string))
            .collect();

        Ok(names.into_iter().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        env::temp_dir().join(format!("nrn_path_{tag}_{}", std::process::id()))
    }

    #[test]
    fn optional_sidecar_returns_the_stem_when_the_file_exists() {
        let dir = temp_dir("sidecar_present");
        fs::create_dir_all(&dir).unwrap();
        let stem = dir.join("preprocessor");
        fs::write(stem.with_extension("json"), "{}").unwrap();

        let found = stem.optional_sidecar("json");

        fs::remove_dir_all(&dir).ok();

        assert_eq!(found, Some(stem));
    }

    #[test]
    fn optional_sidecar_returns_none_when_the_file_is_absent() {
        let dir = temp_dir("sidecar_absent");
        fs::create_dir_all(&dir).unwrap();
        let stem = dir.join("preprocessor");

        let found = stem.optional_sidecar("json");

        fs::remove_dir_all(&dir).ok();

        assert_eq!(found, None);
    }

    #[test]
    fn file_stem_string_drops_the_directory_and_extension() {
        assert_eq!(Path::new("runs/data.parquet").file_stem_string(), "data");
        assert_eq!(Path::new("model").file_stem_string(), "model");
    }

    #[test]
    fn sibling_prefixes_the_stem_in_the_same_directory() {
        assert_eq!(
            Path::new("runs/data.parquet").sibling("curves"),
            Path::new("runs/curves-data")
        );
    }

    #[test]
    fn sibling_of_a_bare_name_has_no_directory() {
        assert_eq!(
            Path::new("run").sibling("boundary"),
            Path::new("boundary-run")
        );
    }

    // `scan_dir` resolves its argument against the current working directory
    // (`combine_safe_with_cwd`), so these use a relative path under `target/`
    // rather than the `temp_dir` helper's absolute, out-of-tree paths.
    fn relative_temp_dir(tag: &str) -> PathBuf {
        PathBuf::from(format!("target/nrn_path_{tag}_{}", std::process::id()))
    }

    #[test]
    fn scan_dir_returns_subdirectories_as_sorted_names() {
        let dir = relative_temp_dir("scan_sorted");
        fs::create_dir_all(&dir).unwrap();
        for name in ["dog", "bird", "cat"] {
            fs::create_dir_all(dir.join(name)).unwrap();
        }
        // A stray file is ignored: only directories are returned.
        fs::write(dir.join("notes.txt"), b"ignored").unwrap();

        let names = dir.scan_dir().unwrap();

        fs::remove_dir_all(&dir).ok();

        assert_eq!(
            names,
            vec!["bird".to_string(), "cat".to_string(), "dog".to_string()]
        );
    }

    #[test]
    fn scan_dir_errors_when_the_root_is_missing() {
        let missing = relative_temp_dir("scan_missing");
        fs::remove_dir_all(&missing).ok();

        assert!(missing.scan_dir().is_err());
    }
}
