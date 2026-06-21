use crate::io::path::PathExt;
use crate::plot::RasterAnimation;
use gif::{Encoder, Frame, Repeat};
use std::error::Error;
use std::fs::File;
use std::path::{Path, PathBuf};

impl RasterAnimation {
    /// Saves the frames as a looping GIF at `path` (the `.gif` extension is set
    /// automatically), returning the written path.
    ///
    /// # Errors
    /// When the animation has no frames.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<PathBuf, Box<dyn Error>> {
        let first = self.frames.first().ok_or("No frames to save")?;
        let (width, height) = (first.width as u16, first.height as u16);

        let filepath = path.as_ref().with_extension("gif");
        let filepath = Path::combine_safe_with_cwd(filepath)?;
        filepath.create_parents()?;

        let file = File::create(&filepath)?;
        let mut encoder = Encoder::new(file, width, height, &[])?;
        encoder.set_repeat(Repeat::Infinite)?;

        for image in &self.frames {
            let mut frame = Frame::from_rgb(width, height, &image.bytes);
            frame.delay = self.frame_delay / 10; // milliseconds to centiseconds
            encoder.write_frame(&frame)?;
        }

        Ok(filepath)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::RasterImage;

    fn temp_path(tag: &str) -> PathBuf {
        let dir = PathBuf::from("target/nrn_tests");
        std::fs::create_dir_all(&dir).ok();
        dir.join(format!("nrn_test_{tag}_{}", std::process::id()))
    }

    fn frame() -> RasterImage {
        RasterImage {
            bytes: vec![64u8; 2 * 2 * 3],
            width: 2,
            height: 2,
        }
    }

    #[test]
    fn save_writes_a_gif_with_the_extension_appended() {
        let animation = RasterAnimation {
            frames: vec![frame(), frame()],
            frame_delay: 50,
        };

        let path = temp_path("raster_gif");
        let written = animation.save(&path).unwrap();
        assert_eq!(written.extension().unwrap(), "gif");
        let header = std::fs::read(&written).unwrap();
        assert_eq!(&header[0..3], b"GIF");

        std::fs::remove_file(&written).ok();
    }

    #[test]
    fn save_rejects_an_empty_animation() {
        let animation = RasterAnimation {
            frames: Vec::new(),
            frame_delay: 50,
        };
        let error = animation.save(temp_path("raster_gif_empty")).unwrap_err();
        assert!(error.to_string().contains("No frames"));
    }

    #[test]
    fn save_rejects_paths_outside_the_working_directory() {
        let animation = RasterAnimation {
            frames: vec![frame()],
            frame_delay: 50,
        };
        assert!(animation.save("../nrn_traversal_anim").is_err());
    }
}
