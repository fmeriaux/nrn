//! Rasterizing an [`ActivationDiagram`] to a horizontal node-link graph: one
//! column of neurons per layer, edges colored by weight sign, values annotated
//! at the ends, and a legend along the bottom.

use super::{ImageConfig, RasterImage, rgb};
use crate::plot::activations::ActivationDiagram;
use crate::plot::scene::Color as SceneColor;
use plotters::backend::BitMapBackend;
use plotters::coord::Shift;
use plotters::prelude::*;
use plotters::style::text_anchor::{HPos, Pos, VPos};
use std::error::Error;

/// Blank pixels reserved around the node field on every side.
const DIAGRAM_MARGIN: i32 = 60;
/// Extra vertical room above the node field for the per-column layer labels.
const LABEL_BAND: i32 = 30;
/// Extra vertical room below the node field for the two-row legend.
const LEGEND_BAND: i32 = 54;
/// Extra horizontal room on each side for the input and output value labels.
const VALUE_GUTTER: i32 = 52;
/// The outline color of a silent (non-firing) neuron on the diagram canvas.
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
    /// magnitude. The input neurons are annotated with their feature values and the
    /// output neurons with their class probability, and a legend runs along the
    /// bottom.
    pub fn to_image(&self, cfg: &ImageConfig) -> Result<RasterImage, Box<dyn Error>> {
        let mut bytes = vec![255u8; (cfg.width * cfg.height * 3) as usize];
        {
            let root =
                BitMapBackend::with_buffer(&mut bytes, (cfg.width, cfg.height)).into_drawing_area();
            root.fill(&rgb(SceneColor::CANVAS))?;

            let positions = self.node_positions(cfg);
            let radius = self.node_radius(cfg);
            self.draw_bands(&root, &positions, cfg)?;
            self.draw_edges(&root, &positions)?;
            self.draw_nodes(&root, &positions, radius)?;
            self.draw_indices(&root, &positions, radius, cfg)?;
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

    /// Shades every other layer column with a faint band, giving the node field
    /// a vertical rhythm that reads as discrete layers rather than a flat canvas.
    /// Each band spans from the midpoint to its neighbours, widening to the value
    /// gutters at the input and output edges.
    fn draw_bands(
        &self,
        root: &DrawingArea<BitMapBackend, Shift>,
        positions: &[Vec<(i32, i32)>],
        cfg: &ImageConfig,
    ) -> Result<(), Box<dyn Error>> {
        let xs: Vec<i32> = positions
            .iter()
            .filter_map(|column| column.first().map(|&(x, _)| x))
            .collect();
        let top = DIAGRAM_MARGIN;
        let bottom = cfg.height as i32 - LEGEND_BAND;
        let style = rgb(SceneColor::LAYER_BAND).filled();

        for (column, &x) in xs.iter().enumerate().step_by(2) {
            let left = if column == 0 {
                x - VALUE_GUTTER
            } else {
                (xs[column - 1] + x) / 2
            };
            let right = match xs.get(column + 1) {
                Some(&next) => (x + next) / 2,
                None => x + VALUE_GUTTER,
            };
            root.draw(&Rectangle::new([(left, top), (right, bottom)], style))?;
        }
        Ok(())
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

    /// Labels every neuron below its marker in a reduced font: an output neuron by
    /// its class, the rest by their index, so the sampled neurons of a large layer
    /// stay identifiable.
    fn draw_indices(
        &self,
        root: &DrawingArea<BitMapBackend, Shift>,
        positions: &[Vec<(i32, i32)>],
        radius: i32,
        cfg: &ImageConfig,
    ) -> Result<(), Box<dyn Error>> {
        let font = (cfg.font_style, (cfg.font_size * 2 / 3).max(10))
            .into_font()
            .color(&BLACK)
            .pos(Pos::new(HPos::Center, VPos::Top));
        for (layer, positions) in self.layers.iter().zip(positions) {
            for (unit, &(x, y)) in layer.units.iter().zip(positions) {
                let label = match unit.class {
                    Some(class) => format!("class {class}"),
                    None => format!("n{}", unit.index),
                };
                root.draw(&Text::new(label, (x, y + radius + 2), font.clone()))?;
            }
        }
        Ok(())
    }

    /// Captions each column with its layer role along the top edge.
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
                root.draw(&Text::new(layer.short_heading(), (x, DIAGRAM_MARGIN), font))?;
            }
        }
        Ok(())
    }

    /// Annotates the input and output neurons alongside their markers: input
    /// features to the left of their column as raw values, output neurons to the
    /// right as their class probability.
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
                let label = match unit.class {
                    Some(_) => format!("{:.1}%", unit.value * 100.0),
                    None => format!("{:.4}", unit.value),
                };
                root.draw(&Text::new(label, (x + dx, y), font.clone()))?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::RELU;
    use crate::model::{NeuralNetwork, NeuronLayer};
    use crate::plot::activations::DiagramOptions;
    use ndarray::array;

    fn pixel_count(buffer: &[u8]) -> usize {
        buffer.len() / 3
    }

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
    fn to_image_draws_the_node_link_graph_onto_the_canvas() {
        let image = diagram().to_image(&ImageConfig::new(300, 200)).unwrap();
        assert_eq!((image.width, image.height), (300, 200));
        assert_eq!(pixel_count(&image.bytes), 300 * 200);
        // A corner outside the node field carries the soft canvas fill, not white.
        let canvas = SceneColor::CANVAS;
        assert_eq!(&image.bytes[0..3], &[canvas.red, canvas.green, canvas.blue]);
        // Nodes, edges and labels are drawn darker than the canvas and its bands.
        assert!(image.bytes.iter().any(|&byte| byte < 200));
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
