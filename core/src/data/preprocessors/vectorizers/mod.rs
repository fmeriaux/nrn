mod image;

pub use image::ImageEncoder;

use ndarray::Array1;
use std::error::Error;

/// Trait defining an interface for encoding data from a file path into a one-dimensional numerical vector.
///
/// This trait is designed for flexible implementations that convert raw data (e.g., images, audio)
/// into `Array1<f32>`, suitable for use in neural network.
///
/// Implementors must ensure that the input data is correctly loaded and transformed into a flat float vector.
/// Note that this trait does not impose any normalization or scaling on the output data; such preprocessing should
/// be handled by dedicated scalers or transformers downstream in the pipeline.
///
/// # Errors
///
/// Implementations may return an error if the data cannot be loaded, decoded, or converted.
///
pub trait VectorEncoder: Send + Sync {
    fn encode(&self, input: &[u8]) -> Result<Array1<f32>, Box<dyn Error>>;
}
