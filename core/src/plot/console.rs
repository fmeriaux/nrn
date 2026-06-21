//! Rendering a [`Figure`] to a text canvas with `textplots`.

use crate::plot::scene::{Color as SceneColor, Figure, Panel, Series};
use rgb::RGB8;
use std::error::Error;
use textplots::{Chart, ColorPlot, Shape};

/// Rendering options for drawing a [`Figure`] as text.
pub struct ConsoleConfig {
    /// Canvas width in dots (two dots per character column).
    pub width: u32,
    /// Canvas height in dots (four dots per character row).
    pub height: u32,
}

impl Default for ConsoleConfig {
    fn default() -> Self {
        Self {
            width: 180,
            height: 60,
        }
    }
}

impl ConsoleConfig {
    /// A console configuration of the given canvas size.
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }
}

impl Figure {
    /// Renders the figure as text, stacking its panels top to bottom.
    ///
    /// # Errors
    /// When the canvas is smaller than `textplots` allows (width below 32 or height below 3).
    pub fn to_console(&self, cfg: &ConsoleConfig) -> Result<String, Box<dyn Error>> {
        if cfg.width < 32 || cfg.height < 3 {
            return Err("Console canvas must be at least 32 by 3 dots".into());
        }

        let mut output = String::new();
        for panel in &self.panels {
            output.push_str(&panel.title);
            output.push('\n');
            output.push_str(&render_panel(panel, cfg));
        }
        Ok(output)
    }
}

/// The `textplots` color for a scene color.
fn rgb(color: SceneColor) -> RGB8 {
    RGB8::new(color.red, color.green, color.blue)
}

/// Renders a single panel's series onto a text canvas the size of `cfg`.
fn render_panel(panel: &Panel, cfg: &ConsoleConfig) -> String {
    let (x_min, x_max) = panel.x_range;
    let (y_min, y_max) = panel.y_range;

    let shapes: Vec<(Shape, RGB8)> = panel
        .series
        .iter()
        .map(|series| match series {
            Series::Line { points, color, .. } => (Shape::Lines(points), rgb(*color)),
            Series::Points { points, color, .. } => (Shape::Points(points), rgb(*color)),
        })
        .collect();

    let mut chart = Chart::new_with_y_range(cfg.width, cfg.height, x_min, x_max, y_min, y_max);
    let mut chart = &mut chart;
    for (shape, color) in &shapes {
        chart = chart.linecolorplot(shape, *color);
    }
    chart.axis();
    chart.figures();
    chart.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::scene::{Color, Figure, Panel, Series};

    fn line_panel() -> Panel {
        Panel {
            title: "Loss".to_string(),
            x_range: (0.0, 10.0),
            y_range: (0.0, 1.0),
            show_legend: true,
            series: vec![
                Series::Line {
                    points: vec![(0.0, 0.9), (10.0, 0.1)],
                    color: Color::TRAIN,
                    label: Some("Train".to_string()),
                },
                Series::Points {
                    points: vec![(2.0, 0.5), (8.0, 0.3)],
                    color: Color::category(0),
                    label: None,
                    radius: 1,
                },
            ],
        }
    }

    #[test]
    fn to_console_renders_line_and_point_series_under_the_title() {
        let figure = Figure {
            panels: vec![line_panel()],
        };
        let text = figure.to_console(&ConsoleConfig::new(64, 32)).unwrap();
        assert!(text.starts_with("Loss\n"));
        // The plotted series draw braille dots: non-blank cells past U+2800.
        assert!(
            text.chars().any(|c| ('\u{2801}'..='\u{28FF}').contains(&c)),
            "expected braille dots in the rendered panel"
        );
    }

    #[test]
    fn to_console_stacks_every_panel_title() {
        let mut top = line_panel();
        top.title = "Top".to_string();
        let mut bottom = line_panel();
        bottom.title = "Bottom".to_string();
        let figure = Figure {
            panels: vec![top, bottom],
        };
        let text = figure.to_console(&ConsoleConfig::default()).unwrap();
        assert!(text.contains("Top\n"));
        assert!(text.contains("Bottom\n"));
    }

    #[test]
    fn to_console_of_an_empty_figure_is_empty() {
        let figure = Figure { panels: Vec::new() };
        let text = figure.to_console(&ConsoleConfig::new(64, 32)).unwrap();
        assert!(text.is_empty());
    }

    #[test]
    fn to_console_rejects_a_too_narrow_canvas() {
        let figure = Figure { panels: Vec::new() };
        let error = figure.to_console(&ConsoleConfig::new(16, 32)).unwrap_err();
        assert!(error.to_string().contains("at least 32"));
    }

    #[test]
    fn to_console_rejects_a_too_short_canvas() {
        let figure = Figure { panels: Vec::new() };
        // Width is fine, so the height bound is what rejects it.
        let error = figure.to_console(&ConsoleConfig::new(64, 2)).unwrap_err();
        assert!(error.to_string().contains("at least 32"));
    }
}
