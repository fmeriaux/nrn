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
//! ```ignore
//! let encoder = ImageEncoder { img_shape: (64, 64), grayscale: true };
//! let img = std::fs::read("image.png")?;
//! let data = encoder.encode(&img)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageFormat, RgbImage};
    use std::io::Cursor;

    /// Encodes a small solid-colour RGB image to in-memory PNG bytes.
    fn png_bytes(width: u32, height: u32, color: [u8; 3]) -> Vec<u8> {
        let img = RgbImage::from_pixel(width, height, image::Rgb(color));
        let mut bytes = Vec::new();
        img.write_to(&mut Cursor::new(&mut bytes), ImageFormat::Png)
            .expect("PNG encoding should succeed");
        bytes
    }

    #[test]
    fn encode_rgb_resizes_and_flattens_to_three_channels() {
        let bytes = png_bytes(8, 8, [10, 20, 30]);
        let encoder = ImageEncoder {
            img_shape: (2, 2),
            grayscale: false,
        };

        let vector = encoder.encode(&bytes).unwrap();

        // 2x2 pixels x 3 channels, raw intensities in [0, 255].
        assert_eq!(vector.len(), 12);
        assert!(vector.iter().all(|&p| (0.0..=255.0).contains(&p)));
        // Solid colour survives the nearest-neighbour resize: every pixel is (10, 20, 30).
        assert_eq!(vector[0], 10.0);
        assert_eq!(vector[1], 20.0);
        assert_eq!(vector[2], 30.0);
    }

    #[test]
    fn encode_grayscale_collapses_to_one_channel() {
        let bytes = png_bytes(8, 8, [255, 255, 255]);
        let encoder = ImageEncoder {
            img_shape: (4, 4),
            grayscale: true,
        };

        let vector = encoder.encode(&bytes).unwrap();

        // 4x4 pixels x 1 channel; a fully white image maps to luma 255.
        assert_eq!(vector.len(), 16);
        assert!(vector.iter().all(|&p| p == 255.0));
    }

    #[test]
    fn encode_rejects_non_image_bytes() {
        let encoder = ImageEncoder {
            img_shape: (2, 2),
            grayscale: false,
        };
        assert!(encoder.encode(b"not an image").is_err());
    }
}
