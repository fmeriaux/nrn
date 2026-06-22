//! Rendering a [`Figure`] to a text canvas with `textplots`.

use crate::plot::scene::{Color as SceneColor, Figure, Panel, Series};
use rgb::RGB8;
use std::error::Error;
use textplots::{Chart, ColorPlot, Shape};

/// The smallest canvas `textplots` accepts, in dots.
const MIN_WIDTH: u32 = 32;
const MIN_HEIGHT: u32 = 3;

/// Rendering options for drawing a [`Figure`] as text.
#[derive(Debug)]
pub struct ConsoleConfig {
    /// Canvas width in dots (two dots per character column).
    width: u32,
    /// Canvas height in dots (four dots per character row).
    height: u32,
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
    ///
    /// # Errors
    /// When the canvas is smaller than `textplots` allows (width below 32 or height below 3).
    pub fn new(width: u32, height: u32) -> Result<Self, Box<dyn Error>> {
        if width < MIN_WIDTH || height < MIN_HEIGHT {
            return Err("Console canvas must be at least 32 by 3 dots".into());
        }
        Ok(Self { width, height })
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
        }
    }

    /// The largest canvas filling a budget of `max_cols` by `max_rows` character
    /// cells, floored to the smallest size `textplots` accepts.
    pub fn filling(max_cols: u16, max_rows: u16) -> Self {
        Self {
            width: (u32::from(max_cols) * 2).max(MIN_WIDTH),
            height: (u32::from(max_rows) * 4).max(MIN_HEIGHT),
        }
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

impl Figure {
    /// Renders the figure as text, stacking its panels top to bottom. Each panel
    /// is titled, drawn, and — when it carries a legend — followed by a colored
    /// key mapping each series color to its label.
    pub fn to_console(&self, cfg: &ConsoleConfig) -> String {
        let mut output = String::new();
        for panel in &self.panels {
            output.push_str(&panel.title);
            output.push('\n');
            output.push_str(&render_panel(panel, cfg));
            if panel.show_legend
                && let Some(legend) = legend_line(panel)
            {
                output.push_str(&legend);
                output.push('\n');
            }
        }
        output
    }
}

/// A one-line color key for a panel's labeled series — a swatch in each series'
/// own color followed by its label — or `None` when no series carries a label.
fn legend_line(panel: &Panel) -> Option<String> {
    let entries: Vec<String> = panel
        .series
        .iter()
        .filter_map(|series| {
            let (color, label) = match series {
                Series::Line { color, label, .. } | Series::Points { color, label, .. } => {
                    (color, label.as_deref()?)
                }
            };
            Some(format!("{} {label}", swatch(*color)))
        })
        .collect();

    (!entries.is_empty()).then(|| entries.join("   "))
}

/// A filled circle in `color`, as a truecolor-escaped string matching the dots
/// `textplots` draws for that series.
fn swatch(color: SceneColor) -> String {
    format!(
        "\u{1b}[38;2;{};{};{}m\u{25cf}\u{1b}[0m",
        color.red, color.green, color.blue
    )
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
        let figure = Figure::spatial(vec![line_panel()]);
        let text = figure.to_console(&ConsoleConfig::new(64, 32).unwrap());
        assert!(text.starts_with("Loss\n"));
        // The plotted series draw braille dots: non-blank cells past U+2800.
        assert!(
            text.chars().any(|c| ('\u{2801}'..='\u{28FF}').contains(&c)),
            "expected braille dots in the rendered panel"
        );
    }

    #[test]
    fn to_console_appends_a_colored_legend_for_labeled_series() {
        let figure = Figure::chart(vec![line_panel()]);
        let text = figure.to_console(&ConsoleConfig::new(64, 32).unwrap());
        // The labeled line shows up in the key, in its own truecolor swatch.
        assert!(text.contains("\u{1b}[38;2;214;39;40m\u{25cf}\u{1b}[0m Train"));
    }

    #[test]
    fn legend_omits_unlabeled_series_and_legend_free_panels() {
        // line_panel carries one labeled line and one unlabeled point series.
        let labeled = legend_line(&line_panel()).unwrap();
        assert!(labeled.contains("Train"));
        assert_eq!(labeled.matches('\u{25cf}').count(), 1);

        // With the legend turned off, nothing is appended past the chart.
        let mut hidden = line_panel();
        hidden.show_legend = false;
        let text = Figure::chart(vec![hidden]).to_console(&ConsoleConfig::new(64, 32).unwrap());
        assert!(!text.contains("Train"));
    }

    #[test]
    fn to_console_stacks_every_panel_title() {
        let mut top = line_panel();
        top.title = "Top".to_string();
        let mut bottom = line_panel();
        bottom.title = "Bottom".to_string();
        let figure = Figure::chart(vec![top, bottom]);
        let text = figure.to_console(&ConsoleConfig::default());
        assert!(text.contains("Top\n"));
        assert!(text.contains("Bottom\n"));
    }

    #[test]
    fn to_console_of_an_empty_figure_is_empty() {
        let figure = Figure::chart(Vec::new());
        let text = figure.to_console(&ConsoleConfig::new(64, 32).unwrap());
        assert!(text.is_empty());
    }

    #[test]
    fn new_rejects_a_too_narrow_canvas() {
        let error = ConsoleConfig::new(16, 32).unwrap_err();
        assert!(error.to_string().contains("at least 32"));
    }

    #[test]
    fn new_rejects_a_too_short_canvas() {
        // Width is fine, so the height bound is what rejects it.
        let error = ConsoleConfig::new(64, 2).unwrap_err();
        assert!(error.to_string().contains("at least 32"));
    }

    #[test]
    fn fitted_floors_to_the_minimum_canvas() {
        let cfg = ConsoleConfig::fitted(1.0, 1, 1);
        assert!(cfg.width >= MIN_WIDTH && cfg.height >= MIN_HEIGHT);
    }

    #[test]
    fn filling_uses_the_whole_cell_budget() {
        // Two dots per column, four per row.
        let cfg = ConsoleConfig::filling(40, 12);
        assert_eq!((cfg.width, cfg.height), (80, 48));
    }

    #[test]
    fn filling_floors_to_the_minimum_canvas() {
        let cfg = ConsoleConfig::filling(1, 0);
        assert!(cfg.width >= MIN_WIDTH && cfg.height >= MIN_HEIGHT);
    }
}
