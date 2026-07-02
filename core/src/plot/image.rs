//! Rasterizing a [`Figure`] to an RGB image with `plotters`, and an
//! [`ActivationDiagram`] to a horizontal node-link graph.

use crate::plot::activations::ActivationDiagram;
use crate::plot::scene::{Color as SceneColor, Figure, Panel, Series};
use plotters::backend::BitMapBackend;
use plotters::chart::{ChartBuilder, ChartContext};
use plotters::coord::Shift;
use plotters::coord::cartesian::Cartesian2d;
use plotters::coord::types::RangedCoordf32;
use plotters::prelude::*;
use plotters::style::text_anchor::{HPos, Pos, VPos};
use std::error::Error;

/// Rendering options for rasterizing a [`Figure`].
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

impl Figure {
    /// Rasterizes the figure to a [`RasterImage`], stacking its panels vertically in order.
    pub fn to_image(&self, cfg: &ImageConfig) -> Result<RasterImage, Box<dyn Error>> {
        let mut bytes = vec![255u8; (cfg.width * cfg.height * 3) as usize];
        {
            let root =
                BitMapBackend::with_buffer(&mut bytes, (cfg.width, cfg.height)).into_drawing_area();
            root.fill(&WHITE)?;

            let regions = root.split_evenly((self.panels.len(), 1));
            for (panel, region) in self.panels.iter().zip(regions) {
                draw_panel(&region, panel, cfg)?;
            }

            root.present()?;
        }

        Ok(RasterImage {
            bytes,
            width: cfg.width,
            height: cfg.height,
        })
    }
}

/// The plotters color for a scene color.
fn rgb(color: SceneColor) -> RGBColor {
    RGBColor(color.red, color.green, color.blue)
}

/// Blank pixels reserved around the node field on every side.
const DIAGRAM_MARGIN: i32 = 60;
/// Extra vertical room above the node field for the per-column layer labels.
const LABEL_BAND: i32 = 30;
/// Extra vertical room below the node field for the two-row legend.
const LEGEND_BAND: i32 = 54;
/// Extra horizontal room on each side for the input and output value labels.
const VALUE_GUTTER: i32 = 52;
/// The outline color of a silent (non-firing) neuron on the white canvas.
const SILENT_NODE: SceneColor = SceneColor::rgb(150, 150, 150);

/// One entry's marker in the legend row.
enum LegendMark {
    /// A filled neuron marker in the given color.
    Filled(SceneColor),
    /// A hollow neuron marker (a silent neuron).
    Hollow,
    /// A connection segment in the given color.
    Line(SceneColor),
    /// No marker, just the label.
    None,
}

impl ActivationDiagram {
    /// Rasterizes the diagram as a horizontal node-link graph: one column of
    /// neurons per layer from input (left) to output (right), each neuron a
    /// circle tinted by activation intensity (hollow when silent) and each
    /// connection a line colored by weight sign, its width and opacity scaled by
    /// magnitude. The input and output neurons are annotated with their values,
    /// the predicted top class is captioned above, and a legend runs along the
    /// bottom.
    pub fn to_image(&self, cfg: &ImageConfig) -> Result<RasterImage, Box<dyn Error>> {
        let mut bytes = vec![255u8; (cfg.width * cfg.height * 3) as usize];
        {
            let root =
                BitMapBackend::with_buffer(&mut bytes, (cfg.width, cfg.height)).into_drawing_area();
            root.fill(&WHITE)?;

            let positions = self.node_positions(cfg);
            let radius = self.node_radius(cfg);
            self.draw_edges(&root, &positions)?;
            self.draw_nodes(&root, &positions, radius)?;
            self.draw_values(&root, &positions, radius, cfg)?;
            self.draw_labels(&root, &positions, cfg)?;
            self.draw_legend(&root, cfg)?;

            root.present()?;
        }

        Ok(RasterImage {
            bytes,
            width: cfg.width,
            height: cfg.height,
        })
    }

    /// The pixel center of every shown neuron, indexed `[layer][unit]`: layers
    /// span the width left to right, neurons span the height top to bottom.
    fn node_positions(&self, cfg: &ImageConfig) -> Vec<Vec<(i32, i32)>> {
        let left = DIAGRAM_MARGIN + VALUE_GUTTER;
        let right = cfg.width as i32 - DIAGRAM_MARGIN - VALUE_GUTTER;
        let top = DIAGRAM_MARGIN + LABEL_BAND;
        let bottom = cfg.height as i32 - DIAGRAM_MARGIN - LEGEND_BAND;

        self.layers
            .iter()
            .enumerate()
            .map(|(column, layer)| {
                let x = spread(column, self.layers.len(), left, right);
                (0..layer.units.len())
                    .map(|row| (x, spread(row, layer.units.len(), top, bottom)))
                    .collect()
            })
            .collect()
    }

    /// The neuron radius: a third of the densest column's vertical spacing,
    /// clamped to a legible range.
    fn node_radius(&self, cfg: &ImageConfig) -> i32 {
        let field = cfg.height as i32 - 2 * DIAGRAM_MARGIN - LABEL_BAND - LEGEND_BAND;
        let densest = self.layers.iter().map(|l| l.units.len()).max().unwrap_or(1);
        let gap = if densest > 1 {
            field / (densest as i32 - 1)
        } else {
            field
        };
        (gap / 3).clamp(3, 14)
    }

    /// Draws each connection as a line between the neurons it joins, colored by
    /// weight sign and scaled in width and opacity by magnitude.
    fn draw_edges(
        &self,
        root: &DrawingArea<BitMapBackend, Shift>,
        positions: &[Vec<(i32, i32)>],
    ) -> Result<(), Box<dyn Error>> {
        for (column, layer) in self.layers.iter().enumerate().skip(1) {
            for edge in &layer.edges {
                let from = positions[column - 1][edge.source];
                let to = positions[column][edge.target];
                let alpha = 0.1 + 0.9 * edge.magnitude as f64;
                let width = 1 + (edge.magnitude * 3.0).round() as u32;
                let style = ShapeStyle::from(rgb(edge.color()).mix(alpha)).stroke_width(width);
                root.draw(&PathElement::new(vec![from, to], style))?;
            }
        }
        Ok(())
    }

    /// Draws each neuron as a filled circle tinted by intensity, or a hollow
    /// outline when silent.
    fn draw_nodes(
        &self,
        root: &DrawingArea<BitMapBackend, Shift>,
        positions: &[Vec<(i32, i32)>],
        radius: i32,
    ) -> Result<(), Box<dyn Error>> {
        for (column, layer) in self.layers.iter().enumerate() {
            for (row, unit) in layer.units.iter().enumerate() {
                let center = positions[column][row];
                if unit.firing {
                    root.draw(&Circle::new(
                        center,
                        radius,
                        rgb(unit.marker_color()).filled(),
                    ))?;
                } else {
                    let outline = ShapeStyle::from(rgb(SILENT_NODE)).stroke_width(2);
                    root.draw(&Circle::new(center, radius, outline))?;
                }
            }
        }
        Ok(())
    }

    /// Captions each column with its layer role and writes the predicted top
    /// class along the top edge.
    fn draw_labels(
        &self,
        root: &DrawingArea<BitMapBackend, Shift>,
        positions: &[Vec<(i32, i32)>],
        cfg: &ImageConfig,
    ) -> Result<(), Box<dyn Error>> {
        let last = self.layers.len().saturating_sub(1);

        for (column, (layer, positions)) in self.layers.iter().zip(positions).enumerate() {
            if let Some(&(x, _)) = positions.first() {
                // Anchor the edge columns inward so the input and output headings
                // stay on the canvas rather than overflowing the margins.
                let hpos = match column {
                    0 => HPos::Left,
                    c if c == last => HPos::Right,
                    _ => HPos::Center,
                };
                let font = (cfg.font_style, cfg.font_size)
                    .into_font()
                    .color(&BLACK)
                    .pos(Pos::new(hpos, VPos::Top));
                root.draw(&Text::new(layer.heading(), (x, DIAGRAM_MARGIN), font))?;
            }
        }

        let (class, probability) = self.prediction.top();
        let caption = format!("Prediction: class {class} ({:.1}%)", probability * 100.0);
        let corner = (cfg.font_style, cfg.font_size).into_font().color(&BLACK);
        root.draw(&Text::new(caption, (DIAGRAM_MARGIN, 16), corner))?;
        Ok(())
    }

    /// Annotates the input and output neurons with their numeric values: input
    /// features to the left of their column, output activations to the right.
    fn draw_values(
        &self,
        root: &DrawingArea<BitMapBackend, Shift>,
        positions: &[Vec<(i32, i32)>],
        radius: i32,
        cfg: &ImageConfig,
    ) -> Result<(), Box<dyn Error>> {
        let last = self.layers.len().saturating_sub(1);
        let gap = radius + 8;

        for (column, hpos, dx) in [(0, HPos::Right, -gap), (last, HPos::Left, gap)] {
            // A single-layer diagram has no distinct output column to label twice.
            if column == last && last == 0 {
                break;
            }
            let font = (cfg.font_style, cfg.font_size)
                .into_font()
                .color(&BLACK)
                .pos(Pos::new(hpos, VPos::Center));
            for (unit, &(x, y)) in self.layers[column].units.iter().zip(&positions[column]) {
                let value = format!("{:.4}", unit.value);
                root.draw(&Text::new(value, (x + dx, y), font.clone()))?;
            }
        }
        Ok(())
    }

    /// Draws a two-row legend along the bottom: the neuron encoding (blue for a
    /// positive value, orange for negative, hollow when silent) and the
    /// connection encoding (blue/orange by weight sign, thicker with magnitude).
    fn draw_legend(
        &self,
        root: &DrawingArea<BitMapBackend, Shift>,
        cfg: &ImageConfig,
    ) -> Result<(), Box<dyn Error>> {
        let top = cfg.height as i32 - LEGEND_BAND;
        self.draw_legend_row(
            root,
            cfg,
            top,
            "neuron",
            &[
                (LegendMark::Filled(SceneColor::POSITIVE), "positive value"),
                (LegendMark::Filled(SceneColor::NEGATIVE), "negative value"),
                (LegendMark::Hollow, "silent"),
            ],
        )?;
        self.draw_legend_row(
            root,
            cfg,
            top + 24,
            "weight",
            &[
                (LegendMark::Line(SceneColor::POSITIVE), "positive"),
                (LegendMark::Line(SceneColor::NEGATIVE), "negative"),
                (LegendMark::None, "thicker = larger magnitude"),
            ],
        )?;
        Ok(())
    }

    /// Draws one legend row at `y`: a prefix, then each marker beside its label.
    fn draw_legend_row(
        &self,
        root: &DrawingArea<BitMapBackend, Shift>,
        cfg: &ImageConfig,
        y: i32,
        prefix: &str,
        entries: &[(LegendMark, &str)],
    ) -> Result<(), Box<dyn Error>> {
        let font = (cfg.font_style, cfg.font_size)
            .into_font()
            .color(&BLACK)
            .pos(Pos::new(HPos::Left, VPos::Center));
        let char_width = cfg.font_size as i32 * 6 / 10;
        let radius = 6;
        let segment = 22;

        // Align both rows' first marker past a fixed-width prefix column.
        root.draw(&Text::new(
            format!("{prefix}:"),
            (DIAGRAM_MARGIN, y),
            font.clone(),
        ))?;
        let mut x = DIAGRAM_MARGIN + 8 * char_width;

        for (mark, text) in entries {
            let text_x = match mark {
                LegendMark::Filled(color) => {
                    root.draw(&Circle::new((x + radius, y), radius, rgb(*color).filled()))?;
                    x + 2 * radius + 8
                }
                LegendMark::Hollow => {
                    let outline = ShapeStyle::from(rgb(SILENT_NODE)).stroke_width(2);
                    root.draw(&Circle::new((x + radius, y), radius, outline))?;
                    x + 2 * radius + 8
                }
                LegendMark::Line(color) => {
                    let style = ShapeStyle::from(rgb(*color)).stroke_width(3);
                    root.draw(&PathElement::new(vec![(x, y), (x + segment, y)], style))?;
                    x + segment + 8
                }
                LegendMark::None => x,
            };
            root.draw(&Text::new(text.to_string(), (text_x, y), font.clone()))?;
            x = text_x + text.len() as i32 * char_width + 28;
        }
        Ok(())
    }
}

/// The evenly spaced position of item `i` of `count` between `start` and `end`,
/// centered when there is only one.
fn spread(i: usize, count: usize, start: i32, end: i32) -> i32 {
    if count <= 1 {
        return (start + end) / 2;
    }
    start + (i as i32) * (end - start) / (count as i32 - 1)
}

/// Draws a single panel — mesh, series and optional legend — onto its region.
fn draw_panel(
    area: &DrawingArea<BitMapBackend, Shift>,
    panel: &Panel,
    cfg: &ImageConfig,
) -> Result<(), Box<dyn Error>> {
    let (x_min, x_max) = panel.x_range;
    let (y_min, y_max) = panel.y_range;
    let mut chart = ChartBuilder::on(area)
        .caption(&panel.title, (cfg.font_style, cfg.font_size).into_font())
        .x_label_area_size(cfg.area_size)
        .y_label_area_size(cfg.area_size)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    chart.configure_mesh().draw()?;

    for series in &panel.series {
        draw_series(&mut chart, series)?;
    }

    if panel.show_legend {
        chart
            .configure_series_labels()
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .label_font((cfg.font_style, cfg.font_size))
            .legend_area_size(cfg.area_size)
            .position(SeriesLabelPosition::LowerRight)
            .draw()?;
    }

    Ok(())
}

/// Draws one series, registering it for the legend when it carries a label.
fn draw_series(
    chart: &mut ChartContext<BitMapBackend, Cartesian2d<RangedCoordf32, RangedCoordf32>>,
    series: &Series,
) -> Result<(), Box<dyn Error>> {
    match series {
        Series::Line {
            points,
            color,
            label,
        } => {
            let color = rgb(*color);
            let annotation = chart.draw_series(LineSeries::new(points.iter().copied(), &color))?;
            if let Some(label) = label {
                annotation
                    .label(label)
                    .legend(move |(x, y)| Circle::new((x, y), 2, color.filled()));
            }
        }
        Series::Points {
            points,
            color,
            label,
            radius,
        } => {
            let color = rgb(*color).filled();
            let radius = *radius as i32;
            let annotation =
                chart.draw_series(points.iter().map(|&p| Circle::new(p, radius, color)))?;
            if let Some(label) = label {
                annotation
                    .label(label)
                    .legend(move |(x, y)| Circle::new((x, y), radius, color));
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::scene::{Color, Figure, Panel, Series};

    fn pixel_count(buffer: &[u8]) -> usize {
        buffer.len() / 3
    }

    mod activation_diagram {
        use super::*;
        use crate::activations::RELU;
        use crate::model::{NeuralNetwork, NeuronLayer};
        use crate::plot::activations::DiagramOptions;
        use ndarray::array;

        fn diagram() -> crate::plot::activations::ActivationDiagram {
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
        fn to_image_draws_the_node_link_graph_onto_the_canvas() {
            let image = diagram().to_image(&ImageConfig::new(300, 200)).unwrap();
            assert_eq!((image.width, image.height), (300, 200));
            assert_eq!(pixel_count(&image.bytes), 300 * 200);
            // Nodes, edges and labels leave non-white pixels behind.
            assert!(image.bytes.iter().any(|&byte| byte != 255));
        }

        #[test]
        fn node_positions_lay_layers_left_to_right() {
            let diagram = diagram();
            let cfg = ImageConfig::new(300, 200);
            let positions = diagram.node_positions(&cfg);
            // Two columns (input, output), the output sitting right of the input.
            assert_eq!(positions.len(), 2);
            assert_eq!(positions[0].len(), 2);
            assert_eq!(positions[1].len(), 3);
            assert!(positions[0][0].0 < positions[1][0].0);
        }
    }

    #[test]
    fn to_image_draws_a_legend_for_labeled_line_and_point_series() {
        // A labeled line and a labeled scatter, with the legend on: both legend
        // markers (line and points) are rendered.
        let figure = Figure::spatial(vec![Panel {
            title: "Lines".to_string(),
            x_range: (0.0, 10.0),
            y_range: (0.0, 1.0),
            show_legend: true,
            series: vec![
                Series::Line {
                    points: vec![(0.0, 0.0), (10.0, 1.0)],
                    color: Color::TRAIN,
                    label: Some("Train".to_string()),
                },
                Series::Points {
                    points: vec![(2.0, 0.3), (8.0, 0.7)],
                    color: Color::category(0),
                    label: Some("Class 0".to_string()),
                    radius: 2,
                },
            ],
        }]);

        let cfg = ImageConfig::new(200, 150);
        let image = figure.to_image(&cfg).unwrap();
        assert_eq!((image.width, image.height), (200, 150));
        assert_eq!(pixel_count(&image.bytes), 200 * 150);
    }

    #[test]
    fn to_image_renders_points_and_unlabeled_series_without_legend() {
        let figure = Figure::spatial(vec![Panel {
            title: "Scatter".to_string(),
            x_range: (0.0, 1.0),
            y_range: (0.0, 1.0),
            show_legend: false,
            series: vec![
                Series::Line {
                    points: vec![(0.0, 0.0), (1.0, 1.0)],
                    color: Color::TRAIN,
                    label: None,
                },
                Series::Points {
                    points: vec![(0.2, 0.2), (0.8, 0.8)],
                    color: Color::category(0),
                    label: Some("Class 0".to_string()),
                    radius: 2,
                },
                Series::Points {
                    points: vec![(0.5, 0.5)],
                    color: Color::BOUNDARY,
                    label: None,
                    radius: 1,
                },
            ],
        }]);

        let cfg = ImageConfig::default();
        let image = figure.to_image(&cfg).unwrap();
        assert_eq!(pixel_count(&image.bytes), (cfg.width * cfg.height) as usize);
    }

    #[test]
    fn to_image_stacks_multiple_panels() {
        let panel = || Panel {
            title: "Panel".to_string(),
            x_range: (0.0, 1.0),
            y_range: (0.0, 1.0),
            show_legend: false,
            series: Vec::new(),
        };
        let figure = Figure::spatial(vec![panel(), panel()]);

        let image = figure.to_image(&ImageConfig::new(120, 120)).unwrap();
        assert_eq!(pixel_count(&image.bytes), 120 * 120);
    }

    #[test]
    fn to_image_of_an_empty_figure_is_blank() {
        let figure = Figure::spatial(Vec::new());
        let image = figure.to_image(&ImageConfig::new(64, 64)).unwrap();
        // No panels drawn: every pixel stays white.
        assert!(image.bytes.iter().all(|&byte| byte == 255));
    }
}
