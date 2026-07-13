use crate::data::Classes;
use crate::io::path::PathExt;
use fs::read_dir;
use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::path::Path;

impl Classes {
    /// Scans `root` for class subdirectories, mapping each subdirectory name to a
    /// contiguous 0-indexed label assigned in sorted name order. Stray files are
    /// ignored: only subdirectories become classes.
    pub fn scan<P: AsRef<Path>>(root: P) -> Result<Self, Box<dyn Error>> {
        let mut directories = read_dir(Path::combine_safe_with_cwd(root)?)?
            .filter_map(Result::ok)
            .filter(|e| e.path().is_dir())
            .collect::<Vec<_>>();

        directories.sort_by_key(|e| e.file_name());

        let mut classes = BTreeMap::new();
        for (index, entry) in directories.into_iter().enumerate() {
            if let Some(name) = entry.file_name().to_str() {
                classes.insert(name.to_string(), index);
            }
        }

        Ok(Classes::new(classes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = PathBuf::from(format!("target/nrn_classes_{}_{}", tag, std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn scans_subdirectories_as_sorted_indexed_classes() {
        let root = temp_dir("sorted");
        for name in ["dog", "bird", "cat"] {
            fs::create_dir_all(root.join(name)).unwrap();
        }
        // A stray file is ignored: only directories become classes.
        fs::write(root.join("notes.txt"), b"ignored").unwrap();

        let classes = Classes::scan(&root).unwrap();
        fs::remove_dir_all(&root).unwrap();

        assert_eq!(
            classes,
            Classes::new(BTreeMap::from([
                ("bird".to_string(), 0),
                ("cat".to_string(), 1),
                ("dog".to_string(), 2),
            ]))
        );
    }

    #[test]
    fn errors_when_root_is_missing() {
        let missing = temp_dir("missing");
        fs::remove_dir_all(&missing).unwrap();
        assert!(Classes::scan(&missing).is_err());
    }
}
