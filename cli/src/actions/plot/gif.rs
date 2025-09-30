use gif::{Encoder, Frame, Repeat};
use std::error::Error;
use std::fs::File;

/// Represents a training view that can display decision boundaries of a neural network
/// and save them as GIF frames.
/// # Properties
/// - `width`: The width of the view in pixels.
/// - `height`: The height of the view in pixels.
/// - `scale`: A tuple representing the minimum and maximum values for scaling the decision boundary.
pub struct GifBuilder<'a> {
    pub width: u16,
    pub height: u16,
    frames: Vec<Frame<'a>>,
}

impl<'a> GifBuilder<'a> {
    /// Creates a new `GifBuilder` instance with the specified width and height.
    /// # Arguments
    /// - `width`: The width of the frame in pixels.
    /// - `height`: The height of the frame in pixels.
    pub fn new(width: u16, height: u16) -> GifBuilder<'a> {
        GifBuilder {
            width,
            height,
            frames: Vec::new(),
        }
    }

    /// Adds a frame to the GIF using the provided RGB pixel data.
    pub fn add_frame(&mut self, rgb_frame: Vec<u8>) -> Result<(), Box<dyn Error>> {
        let mut frame = Frame::from_rgb(self.width, self.height, &rgb_frame);
        frame.delay = 20;
        self.frames.push(frame);
        Ok(())
    }

    /// Saves the frames as a GIF file with the given name.
    /// # Arguments
    /// - `name`: The name of the file to save the GIF as, without the `.gif` extension.
    pub fn save(&mut self, name: &str) -> Result<(), Box<dyn Error>> {
        let mut encoder = Encoder::new(
            File::create(&format!("{}.gif", name))?,
            self.width,
            self.height,
            &[],
        )?;

        encoder.set_repeat(Repeat::Infinite)?;

        for frame in self.frames.iter() {
            encoder.write_frame(&frame)?;
        }

        Ok(())
    }
}
