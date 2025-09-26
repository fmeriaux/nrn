use fs::read_dir;
use image::imageops::FilterType::Nearest;
use image::ImageReader;
use ndarray::Array1;
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

pub fn encode_image<P: AsRef<Path>>(
    img_shape: (u32, u32),
    grayscale: bool,
    path: &P,
) -> Result<Array1<u8>, Box<dyn Error>> {
    let img = ImageReader::open(path)?.decode()?;
    let img = img.resize_exact(img_shape.0, img_shape.1, Nearest);
    let len: usize = (img_shape.0 * img_shape.1) as usize;

    if grayscale {
        return Ok(Array1::from_shape_vec(len, img.to_luma8().into_raw())?);
    }

    Ok(Array1::from_shape_vec(len * 3, img.to_rgb8().into_raw())?)
}
