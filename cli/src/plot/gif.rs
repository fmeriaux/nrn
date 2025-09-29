use crate::plot::chart::draw_data;
use gif::{Encoder, Frame, Repeat};
use nrn::data::Dataset;
use nrn::model::NeuralNetwork;
use std::fs::File;

/// Represents a training view that can display decision boundaries of a neural network
/// and save them as GIF frames.
/// # Properties
/// - `width`: The width of the view in pixels.
/// - `height`: The height of the view in pixels.
/// - `scale`: A tuple representing the minimum and maximum values for scaling the decision boundary.
pub struct DecisionBoundaryView {
    pub width: u32,
    pub height: u32,
    dataset: Dataset,
    frames: Vec<Vec<u8>>,
}

impl DecisionBoundaryView {
    /// Creates a new `DecisionBoundaryView`, initializing the background with a scatter plot of the dataset features and labels.
    /// # Arguments
    /// - `width`: The width of the view in pixels.
    /// - `height`: The height of the view in pixels.
    /// - `dataset`: A reference to the `Dataset` containing features and labels.
    pub fn new(width: u32, height: u32, dataset: Dataset) -> DecisionBoundaryView {
        DecisionBoundaryView {
            width,
            height,
            dataset,
            frames: Vec::new(),
        }
    }

    /// Adds a frame to the decision boundary view by drawing it based on the provided `NeuronNetwork`.
    /// # Arguments
    /// - `model`: A reference to the `NeuronNetwork` whose decision boundary will be drawn.
    pub fn add_frame(&mut self, model: &NeuralNetwork) -> Result<(), Box<dyn std::error::Error>> {
        let buffer = draw_data(&self.dataset, self.width, self.height, Some(model), false)?;
        self.frames.push(buffer.clone());
        Ok(())
    }

    /// Saves the frames as a GIF file with the given name.
    /// # Arguments
    /// - `name`: The name of the file to save the GIF as, without the `.gif` extension.
    pub fn save(&mut self, name: &str) -> Result<(), Box<dyn std::error::Error>> {
        let (width, height) = (self.width as u16, self.height as u16);

        let mut encoder =
            Encoder::new(File::create(&format!("{}.gif", name))?, width, height, &[])?;

        encoder.set_repeat(Repeat::Infinite)?;

        for frame in self.frames.iter() {
            let mut gif_frame = Frame::from_rgb(width, height, frame);
            gif_frame.delay = 20;
            encoder.write_frame(&gif_frame)?;
        }

        Ok(())
    }
}
