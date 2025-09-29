//! Module providing the `ImageEncoder` implementation of the `VectorEncoder` trait.
//!
//! `ImageEncoder` transforms image files into one-dimensional raw pixel vectors suitable for machine learning pipelines.
//! It supports resizing and conversion to grayscale or RGB representation.
//!
//! # Important
//! The output vector contains raw pixel values as floats converted from u8 without normalization.
//! Normalization (e.g., min-max scaling to [0, 1]) should be applied separately, typically through a dedicated scaler.
//!
//! # Usage Example
//!
//! ```
//!
//! let encoder = ImageEncoder { img_shape: (64, 64), grayscale: true };
//! let img = std::fs::read("image.png")?;
//! let mut data = encoder.encode(&img)?;
//! let normalized = scaler.apply_inplace(&mut data)?;
//! ```
//!
use crate::data::vectorizers::VectorEncoder;
use image::imageops::Nearest;
use ndarray::Array1;
use std::error::Error;


/// Resizes and converts images to raw float pixel vectors.
///
/// Images are resized to the configured `img_shape` and converted to grayscale or RGB mode.
/// The output vector contains raw pixel intensities as floats in `[0, 255]` without normalization.
///
/// Normalization (e.g., min-max scaling) should be applied separately for machine learning tasks.
pub struct ImageEncoder {
    /// Target image size `(width, height)` after resizing.
    pub img_shape: (u32, u32),
    /// Whether to convert images to grayscale (`true`) or keep RGB (`false`).
    pub grayscale: bool,
}

/// Encodes an image from raw bytes into a one-dimensional float vector.
///
/// # Errors
///
/// Returns an error if the image cannot be processed (e.g., invalid format).
///
/// # Notes
///
/// Output values are raw, unnormalized pixel intensities in `[0, 255]`.
/// Normalize externally as needed.
impl VectorEncoder for ImageEncoder {
    fn encode(&self, input: &[u8]) -> Result<Array1<f32>, Box<dyn Error>> {
        let img = image::load_from_memory(input)?;
        let img = img.resize_exact(self.img_shape.0, self.img_shape.1, Nearest);

        let pixels: Vec<u8> = if self.grayscale {
            img.to_luma8().into_raw()
        } else {
            img.to_rgb8().into_raw()
        };

        Ok(Array1::from(pixels).map(|&p| p as f32))
    }
}
