mod dataset;
mod training;

use plotters::backend::BitMapBackend;
use plotters::chart::{ChartBuilder, ChartContext};
use plotters::coord::Shift;
use plotters::coord::ranged1d::{AsRangedCoord, ValueFormatter};
use plotters::prelude::*;
use std::error::Error;

/// Provides configuration options for rendering charts.
pub struct RenderConfig<'a> {
    /// Width of the chart in pixels.
    pub width: u32,
    /// Height of the chart in pixels.
    pub height: u32,
    /// Padding factor to add space around the data points, e.g., 0.05 for 5% padding.
    pub padding_factor: f32,
    /// Font style for text elements in the chart, e.g., "sans-serif".
    pub font_style: &'a str,
    /// Font size for text elements in the chart.
    pub font_size: u32,
    /// Size of the area allocated for axis labels and legends.
    pub area_size: u32,
}

impl Default for RenderConfig<'_> {
    fn default() -> Self {
        Self {
            width: 1200,
            height: 900,
            padding_factor: 0.05,
            font_style: "sans-serif",
            font_size: 20,
            area_size: 40,
        }
    }
}

impl RenderConfig<'_> {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            ..Default::default()
        }
    }
}

/// Draws a chart with the specified width and height using the provided drawing function.
fn draw_with<F>(cfg: &RenderConfig, draw: F) -> Result<Vec<u8>, Box<dyn Error>>
where
    F: FnOnce(&DrawingArea<BitMapBackend<'_>, Shift>) -> Result<(), Box<dyn Error>>,
{
    let mut buffer = vec![255u8; (cfg.width * cfg.height * 3) as usize];
    {
        let root =
            BitMapBackend::with_buffer(&mut buffer, (cfg.width, cfg.height)).into_drawing_area();
        root.fill(&WHITE)?;

        draw(&root)?;

        root.present()?;
    }

    Ok(buffer)
}

fn draw_chart<'a, 'b, F, X: AsRangedCoord, Y: AsRangedCoord>(
    area: &'a DrawingArea<BitMapBackend<'b>, Shift>,
    title: &str,
    x_range: X,
    y_range: Y,
    cfg: &RenderConfig,
    show_legend: bool,
    draw: F,
) -> Result<(), Box<dyn Error>>
where
    F: FnOnce(
        &mut ChartContext<'a, BitMapBackend<'b>, Cartesian2d<X::CoordDescType, Y::CoordDescType>>,
    ) -> Result<(), Box<dyn Error>>,
    X::CoordDescType: ValueFormatter<X::Value>,
    Y::CoordDescType: ValueFormatter<Y::Value>,
{
    let mut chart = ChartBuilder::on(area)
        .caption(title, (cfg.font_style, cfg.font_size).into_font())
        .x_label_area_size(cfg.area_size)
        .y_label_area_size(cfg.area_size)
        .build_cartesian_2d(x_range, y_range)?;

    chart.configure_mesh().draw()?;

    draw(&mut chart)?;

    if show_legend {
        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .label_font((cfg.font_style, cfg.font_size))
            .legend_area_size(cfg.area_size)
            .position(SeriesLabelPosition::LowerRight)
            .draw()?;
    }

    Ok(())
}

/// Adds padding to the given min and max values by the specified padding factor.
fn add_padding(mins: &[f32], maxs: &[f32], padding_factor: f32) -> (Vec<f32>, Vec<f32>) {
    let mut padded_mins = Vec::with_capacity(mins.len());
    let mut padded_maxs = Vec::with_capacity(maxs.len());

    for (&min, &max) in mins.iter().zip(maxs.iter()) {
        let range = max - min;
        padded_mins.push(min - range * padding_factor);
        padded_maxs.push(max + range * padding_factor);
    }

    (padded_mins, padded_maxs)
}
