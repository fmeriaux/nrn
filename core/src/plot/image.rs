//! Rasterizing a [`Figure`] to an RGB image with `plotters`.

use crate::plot::scene::{Color as SceneColor, Figure, Panel, Series};
use plotters::backend::BitMapBackend;
use plotters::chart::{ChartBuilder, ChartContext};
use plotters::coord::Shift;
use plotters::coord::cartesian::Cartesian2d;
use plotters::coord::types::RangedCoordf32;
use plotters::prelude::*;
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

impl Figure {
    /// Rasterizes the figure to a `width × height` RGB byte buffer, stacking its panels
    /// vertically in order.
    pub fn to_image(&self, cfg: &ImageConfig) -> Result<Vec<u8>, Box<dyn Error>> {
        let mut buffer = vec![255u8; (cfg.width * cfg.height * 3) as usize];
        {
            let root = BitMapBackend::with_buffer(&mut buffer, (cfg.width, cfg.height))
                .into_drawing_area();
            root.fill(&WHITE)?;

            let regions = root.split_evenly((self.panels.len(), 1));
            for (panel, region) in self.panels.iter().zip(regions) {
                draw_panel(&region, panel, cfg)?;
            }

            root.present()?;
        }

        Ok(buffer)
    }
}

/// The plotters color for a scene color.
fn rgb(color: SceneColor) -> RGBColor {
    RGBColor(color.red, color.green, color.blue)
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

    #[test]
    fn to_image_draws_a_legend_for_labeled_line_and_point_series() {
        // A labeled line and a labeled scatter, with the legend on: both legend
        // markers (line and points) are rendered.
        let figure = Figure {
            panels: vec![Panel {
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
            }],
        };

        let cfg = ImageConfig::new(200, 150);
        let buffer = figure.to_image(&cfg).unwrap();
        assert_eq!(pixel_count(&buffer), 200 * 150);
    }

    #[test]
    fn to_image_renders_points_and_unlabeled_series_without_legend() {
        let figure = Figure {
            panels: vec![Panel {
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
            }],
        };

        let cfg = ImageConfig::default();
        let buffer = figure.to_image(&cfg).unwrap();
        assert_eq!(pixel_count(&buffer), (cfg.width * cfg.height) as usize);
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
        let figure = Figure {
            panels: vec![panel(), panel()],
        };

        let buffer = figure.to_image(&ImageConfig::new(120, 120)).unwrap();
        assert_eq!(pixel_count(&buffer), 120 * 120);
    }

    #[test]
    fn to_image_of_an_empty_figure_is_blank() {
        let figure = Figure { panels: Vec::new() };
        let buffer = figure.to_image(&ImageConfig::new(64, 64)).unwrap();
        // No panels drawn: every pixel stays white.
        assert!(buffer.iter().all(|&byte| byte == 255));
    }
}
