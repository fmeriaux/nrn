//! Rendering plot IRs to text: [`Figure`](crate::plot::Figure) charts in
//! [`figure`] and [`ActivationDiagram`](crate::plot::ActivationDiagram) layer
//! lists in [`diagram`]. This module owns the [`ConsoleConfig`] and the
//! truecolor-escape helpers shared by both renderers.

mod diagram;
mod figure;

use crate::plot::scene::Color as SceneColor;
use std::error::Error;

/// The smallest canvas `textplots` accepts, in dots.
const MIN_WIDTH: u32 = 32;
const MIN_HEIGHT: u32 = 3;

/// Rendering options for drawing a [`Figure`](crate::plot::Figure) as text.
#[derive(Debug)]
pub struct ConsoleConfig {
    /// Canvas width in dots (two dots per character column).
    width: u32,
    /// Canvas height in dots (four dots per character row).
    height: u32,
    /// How many panels to place side by side per row; 1 stacks them.
    columns: usize,
}

impl Default for ConsoleConfig {
    fn default() -> Self {
        Self {
            width: 180,
            height: 60,
            columns: 1,
        }
    }
}

impl ConsoleConfig {
    /// A console configuration of the given canvas size.
    ///
    /// # Errors
    /// When the canvas is smaller than `textplots` allows (width below 32 or height below 3).
    pub fn new(width: u32, height: u32) -> Result<Self, Box<dyn Error>> {
        if width < MIN_WIDTH || height < MIN_HEIGHT {
            return Err("Console canvas must be at least 32 by 3 dots".into());
        }
        Ok(Self {
            width,
            height,
            columns: 1,
        })
    }

    /// The largest canvas that honours `aspect` (data width / height) within a
    /// budget of `max_cols` by `max_rows` character cells.
    ///
    /// Braille dots are square — two per column, four per row over a cell that
    /// is twice as tall as wide — so a square data domain yields a square
    /// canvas. The result is floored to the smallest size `textplots` accepts.
    pub fn fitted(aspect: f32, max_cols: u16, max_rows: u16) -> Self {
        let max_width = u32::from(max_cols) * 2;
        let max_height = u32::from(max_rows) * 4;

        let mut height = max_height;
        let mut width = (height as f32 * aspect).round() as u32;
        if width > max_width {
            width = max_width;
            height = (width as f32 / aspect).round() as u32;
        }

        Self {
            width: width.max(MIN_WIDTH),
            height: height.max(MIN_HEIGHT),
            columns: 1,
        }
    }

    /// The largest canvas filling a budget of `max_cols` by `max_rows` character
    /// cells, floored to the smallest size `textplots` accepts.
    pub fn filling(max_cols: u16, max_rows: u16) -> Self {
        Self {
            width: (u32::from(max_cols) * 2).max(MIN_WIDTH),
            height: (u32::from(max_rows) * 4).max(MIN_HEIGHT),
            columns: 1,
        }
    }

    /// The same configuration arranging panels into rows of `columns` placed
    /// side by side (at least one), rather than a single stacked column.
    pub fn with_columns(mut self, columns: usize) -> Self {
        self.columns = columns.max(1);
        self
    }

    /// How many panels are placed side by side per row.
    pub fn columns(&self) -> usize {
        self.columns
    }

    /// The canvas width in dots.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// The canvas height in dots.
    pub fn height(&self) -> u32 {
        self.height
    }
}

/// A filled circle in `color`, as a truecolor-escaped string matching the dots
/// `textplots` draws for that series.
fn swatch(color: SceneColor) -> String {
    colored('\u{25cf}', color)
}

/// `glyph` wrapped in a truecolor escape so it prints in `color`, reset after.
fn colored(glyph: char, color: SceneColor) -> String {
    colored_str(&glyph.to_string(), color)
}

/// `text` wrapped in a truecolor escape so it prints in `color`, reset after.
fn colored_str(text: &str, color: SceneColor) -> String {
    format!(
        "\u{1b}[38;2;{};{};{}m{text}\u{1b}[0m",
        color.red, color.green, color.blue
    )
}
