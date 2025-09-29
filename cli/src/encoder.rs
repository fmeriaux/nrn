use fs::read_dir;
use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::path::Path;

type CategoryMap = BTreeMap<String, usize>;

/// Extracts categories with their indices from the directory structure.
pub fn extract_categories<P: AsRef<Path>>(root: &P) -> Result<CategoryMap, Box<dyn Error>> {
    let mut categories = BTreeMap::new();

    let mut directories = read_dir(root)?
        .filter_map(Result::ok)
        .filter(|e| e.path().is_dir())
        .collect::<Vec<_>>();

    directories.sort_by_key(|e| e.file_name());

    directories
        .into_iter()
        .enumerate()
        .for_each(|(index, entry)| {
            if let Some(name) = entry.file_name().to_str() {
                categories.insert(name.to_string(), index);
            }
        });

    Ok(categories)
}
