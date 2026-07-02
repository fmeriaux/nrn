//! Rendering a [`Figure`] to a text canvas with `textplots`, laying panels out
//! in rows and appending a colored key for labeled series.

use super::{ConsoleConfig, swatch};
use crate::plot::scene::{Color as SceneColor, Figure, Panel, Series};
use rgb::RGB8;
use textplots::{Chart, ColorPlot, Shape};

/// Blank columns separating panels placed side by side.
const GUTTER: &str = "   ";

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

        if cfg.columns() <= 1 {
            return blocks.concat();
        }

        let mut output = String::new();
        for row in blocks.chunks(cfg.columns()) {
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

    let mut chart = Chart::new_with_y_range(cfg.width(), cfg.height(), x_min, x_max, y_min, y_max);
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
