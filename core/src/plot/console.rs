//! Rendering a [`Figure`] to a text canvas with `textplots`, and an
//! [`ActivationDiagram`] to a colored list of layers.

use crate::classification::Classification;
use crate::plot::activations::{ActivationDiagram, DiagramLayer, Unit};
use crate::plot::scene::{Color as SceneColor, Figure, Panel, Series};
use rgb::RGB8;
use std::error::Error;
use textplots::{Chart, ColorPlot, Shape};

/// The smallest canvas `textplots` accepts, in dots.
const MIN_WIDTH: u32 = 32;
const MIN_HEIGHT: u32 = 3;

/// Blank columns separating panels placed side by side.
const GUTTER: &str = "   ";

/// Rendering options for drawing a [`Figure`] as text.
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

impl Figure {
    /// Renders the figure as text. Panels are laid out in rows of
    /// [`ConsoleConfig::columns`] placed side by side, each row stacked beneath
    /// the previous. Each panel is titled, drawn, and — when it carries a legend
    /// — followed by a colored key mapping each series color to its label.
    pub fn to_console(&self, cfg: &ConsoleConfig) -> String {
        let blocks: Vec<String> = self
            .panels
            .iter()
            .map(|panel| render_block(panel, cfg))
            .collect();

        if cfg.columns <= 1 {
            return blocks.concat();
        }

        let mut output = String::new();
        for row in blocks.chunks(cfg.columns) {
            output.push_str(&join_row(row));
        }
        output
    }
}

/// Renders one panel as a titled block: its title, the plotted canvas, and —
/// when the panel carries a legend — a colored key beneath it.
fn render_block(panel: &Panel, cfg: &ConsoleConfig) -> String {
    let mut block = String::new();
    block.push_str(&panel.title);
    block.push('\n');
    block.push_str(&render_panel(panel, cfg));
    if panel.show_legend
        && let Some(legend) = legend_line(panel)
    {
        block.push_str(&legend);
        block.push('\n');
    }
    block
}

/// Joins panel blocks side by side, padding each to its own visible width and
/// separating columns with a gutter. Shorter blocks are padded with blank lines.
fn join_row(blocks: &[String]) -> String {
    let columns: Vec<Vec<&str>> = blocks.iter().map(|block| block.lines().collect()).collect();
    let widths: Vec<usize> = columns
        .iter()
        .map(|lines| {
            lines
                .iter()
                .map(|line| display_width(line))
                .max()
                .unwrap_or(0)
        })
        .collect();
    let height = columns.iter().map(Vec::len).max().unwrap_or(0);

    let mut output = String::new();
    for row in 0..height {
        for (column, lines) in columns.iter().enumerate() {
            if column > 0 {
                output.push_str(GUTTER);
            }
            let line = lines.get(row).copied().unwrap_or("");
            output.push_str(line);
            // Pad to the column width so the next column stays aligned; the last
            // column needs no trailing padding.
            if column + 1 < columns.len() {
                output.extend(std::iter::repeat_n(
                    ' ',
                    widths[column] - display_width(line),
                ));
            }
        }
        output.push('\n');
    }
    output
}

/// The visible width of `line` in columns, skipping ANSI CSI escape sequences
/// (the color codes `textplots` emits for plotted points).
fn display_width(line: &str) -> usize {
    let mut width = 0;
    let mut chars = line.chars();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' {
            // Consume up to the escape sequence's final letter byte.
            for c in chars.by_ref() {
                if c.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            width += 1;
        }
    }
    width
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
    colored('\u{25cf}', color)
}

/// `glyph` wrapped in a truecolor escape so it prints in `color`, reset after.
fn colored(glyph: char, color: SceneColor) -> String {
    format!(
        "\u{1b}[38;2;{};{};{}m{glyph}\u{1b}[0m",
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

/// The hollow marker color of a silent (non-firing) neuron.
const SILENT_COLOR: SceneColor = SceneColor::rgb(90, 90, 90);

impl ActivationDiagram {
    /// Renders the forward pass as a vertical list of layers from input to
    /// output: each neuron a colored marker beside its value — filled and tinted
    /// by activation intensity when firing, hollow when silent — followed by the
    /// predicted class ranking. No connections are drawn.
    pub fn to_console(&self) -> String {
        let mut output = String::new();
        for layer in &self.layers {
            output.push_str(&render_layer(layer));
            output.push('\n');
        }
        output.push_str(&render_prediction(&self.prediction));
        output
    }
}

/// One layer as a heading naming its role and neuron count, then one line per
/// shown neuron.
fn render_layer(layer: &DiagramLayer) -> String {
    let mut block = layer.heading();
    block.push('\n');
    for unit in &layer.units {
        block.push_str(&render_unit(unit));
        block.push('\n');
    }
    block
}

/// One neuron: a filled marker tinted by intensity when firing, or a hollow
/// marker flagged silent, beside its index and value.
fn render_unit(unit: &Unit) -> String {
    if unit.firing {
        format!(
            "  {} n{:<3} {:>9.4}",
            colored('\u{25cf}', unit.marker_color()),
            unit.index,
            unit.value
        )
    } else {
        format!(
            "  {} n{:<3} {:>9.4}  (silent)",
            colored('\u{25cb}', SILENT_COLOR),
            unit.index,
            unit.value
        )
    }
}

/// The predicted class ranking: one line per class with its category swatch and
/// probability, the most likely first and arrow-marked.
fn render_prediction(prediction: &Classification) -> String {
    let mut block = String::from("Prediction\n");
    for (rank, &(class, probability)) in prediction.ranking().iter().enumerate() {
        let marker = if rank == 0 { "  <-" } else { "" };
        block.push_str(&format!(
            "  {} class {class}  {:>5.1}%{marker}\n",
            swatch(SceneColor::category(class)),
            probability * 100.0
        ));
    }
    block
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

    mod activation_diagram {
        use super::*;
        use crate::activations::{Activation, RELU};
        use crate::model::{NeuralNetwork, NeuronLayer};
        use crate::plot::activations::DiagramOptions;
        use ndarray::{Array1, Array2, array};

        /// A single ReLU layer; neuron 1's negative pre-activation goes silent.
        fn diagram() -> ActivationDiagram {
            let net = NeuralNetwork {
                layers: vec![NeuronLayer {
                    weights: array![[1.0, 0.0], [-1.0, 0.0], [2.0, 0.0]],
                    biases: array![0.0, 0.0, 0.0],
                    activation: RELU.clone(),
                }],
            };
            net.activation_diagram(array![1.0, 1.0].view(), &DiagramOptions::default())
                .unwrap()
        }

        #[test]
        fn to_console_heads_each_layer_with_its_role_and_count() {
            let text = diagram().to_console();
            assert!(text.contains("Input (2 features)"));
            assert!(text.contains(&format!("{} (3 units)", RELU.name())));
        }

        #[test]
        fn to_console_marks_firing_neurons_filled_and_silent_neurons_hollow() {
            let text = diagram().to_console();
            // The dead neuron (value 0) is hollow and flagged; active ones are filled.
            assert!(text.contains('\u{25cb}'));
            assert!(text.contains("(silent)"));
            assert!(text.contains('\u{25cf}'));
            // Concrete activation values are printed beside their neuron.
            assert!(text.contains("n2"));
            assert!(text.contains("2.0000"));
        }

        #[test]
        fn to_console_flags_a_sampled_layer_in_its_heading() {
            let weights = Array2::from_shape_fn((50, 2), |(r, _)| r as f32);
            let net = NeuralNetwork {
                layers: vec![NeuronLayer {
                    weights,
                    biases: Array1::zeros(50),
                    activation: RELU.clone(),
                }],
            };
            let options = DiagramOptions {
                max_units: 8,
                ..DiagramOptions::default()
            };
            let text = net
                .activation_diagram(array![1.0, 1.0].view(), &options)
                .unwrap()
                .to_console();
            assert!(text.contains("showing 8 of 50 units"));
        }

        #[test]
        fn to_console_lists_the_ranked_prediction_with_the_top_class_marked() {
            // Output activations [1, 0, 2]: class 2 leads.
            let text = diagram().to_console();
            let prediction = text.split("Prediction").nth(1).unwrap();
            let first = prediction.lines().nth(1).unwrap();
            assert!(first.contains("class 2"));
            assert!(first.contains("<-"));
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

    #[test]
    fn display_width_ignores_ansi_color_escapes() {
        // A truecolor swatch is one visible glyph despite its escape codes.
        assert_eq!(display_width(&format!("{}x", swatch(Color::TRAIN))), 2);
    }

    #[test]
    fn join_row_pads_each_column_to_align_the_next() {
        // The left block's lines differ in width and it has more lines than the
        // right; the left column pads to its widest line so the right column
        // starts at the same offset, and the missing right line is blank.
        let left = "ab\nlongerline\n".to_string();
        let right = "X\n".to_string();
        let lines: Vec<String> = join_row(&[left, right])
            .lines()
            .map(str::to_owned)
            .collect();
        assert_eq!(lines[0], format!("ab{}   X", " ".repeat(8)));
        assert_eq!(lines[1], "longerline   ");
    }

    #[test]
    fn columns_arrange_panels_side_by_side() {
        let mut left = line_panel();
        left.title = "Loss".to_string();
        let mut right = line_panel();
        right.title = "Accuracy".to_string();
        let figure = Figure::chart(vec![left, right]);

        let cfg = ConsoleConfig::new(48, 16).unwrap().with_columns(2);
        let text = figure.to_console(&cfg);

        // Both titles share the first line — the panels sit beside each other
        // rather than stacking.
        let first = text.lines().next().unwrap();
        assert!(first.contains("Loss") && first.contains("Accuracy"));
    }

    #[test]
    fn a_single_column_stacks_the_panels() {
        let mut top = line_panel();
        top.title = "Loss".to_string();
        let mut bottom = line_panel();
        bottom.title = "Accuracy".to_string();
        let figure = Figure::chart(vec![top, bottom]);

        // Stacked (the default), each title opens its own line.
        let text = figure.to_console(&ConsoleConfig::new(48, 16).unwrap());
        assert!(text.contains("Loss\n"));
        assert!(text.contains("Accuracy\n"));
    }
}
