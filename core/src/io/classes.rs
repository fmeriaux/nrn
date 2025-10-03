use fs::read_dir;
use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::path::Path;
use crate::io::path::PathExt;

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
