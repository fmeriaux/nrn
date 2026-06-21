use crate::io::path::PathExt;
use crate::plot::RasterImage;
use std::fs::File;
use std::io::Result;
use std::path::{Path, PathBuf};

impl RasterImage {
    /// Saves the image as a PNG at `path` (the `.png` extension is set automatically),
    /// returning the written path.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<PathBuf> {
        let filepath = path.as_ref().with_extension("png");
        let filepath = Path::combine_safe_with_cwd(filepath)?;
        filepath.create_parents()?;

        let file = File::create(&filepath)?;
        let mut encoder = png::Encoder::new(file, self.width, self.height);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header()?;
        writer.write_image_data(&self.bytes)?;
        Ok(filepath)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(tag: &str) -> PathBuf {
        let dir = PathBuf::from("target/nrn_tests");
        std::fs::create_dir_all(&dir).ok();
        dir.join(format!("nrn_test_{tag}_{}", std::process::id()))
    }

    #[test]
    fn save_writes_a_png_with_the_extension_appended() {
        let image = RasterImage {
            bytes: vec![128u8; 4 * 3 * 3],
            width: 4,
            height: 3,
        };

        let path = temp_path("raster_png");
        let written = image.save(&path).unwrap();
        assert_eq!(written.extension().unwrap(), "png");
        // A real PNG file landed on disk.
        let header = std::fs::read(&written).unwrap();
        assert_eq!(&header[1..4], b"PNG");

        std::fs::remove_file(&written).ok();
    }

    #[test]
    fn save_rejects_paths_outside_the_working_directory() {
        let image = RasterImage {
            bytes: vec![0u8; 3],
            width: 1,
            height: 1,
        };
        assert!(image.save("../nrn_traversal_image").is_err());
    }
}
