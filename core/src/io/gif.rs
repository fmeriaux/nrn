use crate::io::path::PathExt;
use crate::plot::RasterImage;
use gif::{Encoder, Frame, Repeat};
use std::error::Error;
use std::fs::File;
use std::path::{Path, PathBuf};

/// A streaming encoder for a looping GIF: frames are written one at a time as
/// they are produced, so a caller can render-and-encode in a single pass without
/// ever holding the whole animation in memory.
///
/// The trailer is written when the writer is dropped; [`finish`](GifWriter::finish)
/// returns the written path.
pub struct GifWriter {
    encoder: Encoder<File>,
    width: u16,
    height: u16,
    frame_delay: u16,
    path: PathBuf,
}

impl GifWriter {
    /// Opens a GIF at `path` (the `.gif` extension is set automatically) sized to
    /// `width × height`, looping forever, with `frame_delay` milliseconds between
    /// frames.
    pub fn create(
        path: impl AsRef<Path>,
        width: u32,
        height: u32,
        frame_delay: u16,
    ) -> Result<Self, Box<dyn Error>> {
        let (width, height) = (width as u16, height as u16);

        let path = path.as_ref().with_extension("gif");
        let path = Path::combine_safe_with_cwd(path)?;
        path.create_parents()?;

        let file = File::create(&path)?;
        let mut encoder = Encoder::new(file, width, height, &[])?;
        encoder.set_repeat(Repeat::Infinite)?;

        Ok(Self {
            encoder,
            width,
            height,
            frame_delay,
            path,
        })
    }

    /// Encodes one frame, in display order.
    ///
    /// # Errors
    /// When the image's dimensions differ from the writer's, or encoding fails.
    pub fn write_frame(&mut self, image: &RasterImage) -> Result<(), Box<dyn Error>> {
        if image.width != self.width as u32 || image.height != self.height as u32 {
            return Err(format!(
                "frame is {}x{} but the GIF is {}x{}",
                image.width, image.height, self.width, self.height
            )
            .into());
        }

        let mut frame = Frame::from_rgb(self.width, self.height, &image.bytes);
        frame.delay = self.frame_delay / 10; // milliseconds to centiseconds
        self.encoder.write_frame(&frame)?;
        Ok(())
    }

    /// Finalizes the GIF (flushing on drop) and returns the written path.
    pub fn finish(self) -> PathBuf {
        self.path
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

    fn frame(value: u8) -> RasterImage {
        RasterImage {
            bytes: vec![value; 2 * 2 * 3],
            width: 2,
            height: 2,
        }
    }

    #[test]
    fn writes_a_looping_gif_with_the_extension_appended() {
        let path = temp_path("gif_writer");
        let mut writer = GifWriter::create(&path, 2, 2, 50).unwrap();
        writer.write_frame(&frame(64)).unwrap();
        writer.write_frame(&frame(128)).unwrap();
        let written = writer.finish();

        assert_eq!(written.extension().unwrap(), "gif");
        let bytes = std::fs::read(&written).unwrap();
        assert_eq!(&bytes[0..3], b"GIF");

        std::fs::remove_file(&written).ok();
    }

    #[test]
    fn rejects_a_frame_of_the_wrong_size() {
        let path = temp_path("gif_writer_mismatch");
        let mut writer = GifWriter::create(&path, 2, 2, 50).unwrap();
        let error = writer.write_frame(&frame_of(4, 4)).unwrap_err();
        assert!(error.to_string().contains("4x4"));

        std::fs::remove_file(writer.finish()).ok();
    }

    #[test]
    fn rejects_paths_outside_the_working_directory() {
        assert!(GifWriter::create("../nrn_traversal_anim", 2, 2, 50).is_err());
    }

    fn frame_of(width: u32, height: u32) -> RasterImage {
        RasterImage {
            bytes: vec![0; (width * height * 3) as usize],
            width,
            height,
        }
    }
}
