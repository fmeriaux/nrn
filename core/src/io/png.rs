use crate::io::path::PathExt;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::io::Result;

pub fn save_rgb<P: AsRef<Path>>(
    frame_rgb: Vec<u8>,
    path: P,
    width: u32,
    height: u32,
) -> Result<PathBuf> {
    let filepath = path.as_ref().with_extension("png");
    let filepath = Path::combine_safe_with_cwd(filepath)?;
    filepath.create_parents()?;

    let file = File::create(&filepath)?;
    let mut encoder = png::Encoder::new(file, width, height);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;
    writer.write_image_data(&frame_rgb)?;
    Ok(filepath.to_path_buf())
}
