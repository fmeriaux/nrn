use crate::io::path::PathExt;
use fs::read_dir;
use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::path::Path;

type ClassMap = BTreeMap<String, usize>;

/// Extract class names from subdirectory names within the given root directory.
pub fn extract_classes<P: AsRef<Path>>(root: &P) -> Result<ClassMap, Box<dyn Error>> {
    let mut classes = BTreeMap::new();

    let mut directories = read_dir(Path::combine_safe_with_cwd(root)?)?
        .filter_map(Result::ok)
        .filter(|e| e.path().is_dir())
        .collect::<Vec<_>>();

    directories.sort_by_key(|e| e.file_name());

    directories
        .into_iter()
        .enumerate()
        .for_each(|(index, entry)| {
            if let Some(name) = entry.file_name().to_str() {
                classes.insert(name.to_string(), index);
            }
        });

    Ok(classes)
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
    fn extracts_subdirectories_as_sorted_indexed_classes() {
        let root = temp_dir("sorted");
        for name in ["dog", "bird", "cat"] {
            fs::create_dir_all(root.join(name)).unwrap();
        }
        // A stray file is ignored: only directories become classes.
        fs::write(root.join("notes.txt"), b"ignored").unwrap();

        let classes = extract_classes(&root).unwrap();
        fs::remove_dir_all(&root).unwrap();

        assert_eq!(classes.get("bird"), Some(&0));
        assert_eq!(classes.get("cat"), Some(&1));
        assert_eq!(classes.get("dog"), Some(&2));
        assert_eq!(classes.len(), 3);
    }

    #[test]
    fn errors_when_root_is_missing() {
        let missing = temp_dir("missing");
        fs::remove_dir_all(&missing).unwrap();
        assert!(extract_classes(&missing).is_err());
    }
}
