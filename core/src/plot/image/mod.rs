//! Rasterizing plot IRs to RGB with `plotters`: [`Figure`](crate::plot::Figure)
//! charts in [`figure`] and [`ActivationDiagram`](crate::plot::ActivationDiagram)
//! node-link graphs in [`diagram`]. This module owns the shared output types and
//! the scene-to-`plotters` color conversion.

mod diagram;
mod figure;

use crate::plot::scene::Color as SceneColor;
use plotters::prelude::RGBColor;

/// Rendering options for rasterizing a [`Figure`](crate::plot::Figure).
pub struct ImageConfig<'a> {
    /// Width of the image in pixels.
    pub width: u32,
    /// Height of the image in pixels.
    pub height: u32,
    /// Font family for titles, labels and legends, e.g. "sans-serif".
    pub font_style: &'a str,
    /// Font size for text elements.
    pub font_size: u32,
    /// Size of the area reserved for axis labels and legends.
    pub area_size: u32,
}

impl Default for ImageConfig<'_> {
    fn default() -> Self {
        Self {
            width: 1200,
            height: 900,
            font_style: "sans-serif",
            font_size: 20,
            area_size: 40,
        }
    }
}

impl ImageConfig<'_> {
    /// An image configuration of the given size, with default fonts and spacing.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            ..Default::default()
        }
    }
}

/// A rendered RGB image: a row-major buffer of `width × height` 8-bit RGB triples.
pub struct RasterImage {
    /// Row-major RGB pixel data, three bytes per pixel.
    pub bytes: Vec<u8>,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
}

/// The plotters color for a scene color.
fn rgb(color: SceneColor) -> RGBColor {
    RGBColor(color.red, color.green, color.blue)
}
