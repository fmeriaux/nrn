mod dataset;
mod training;

pub use dataset::*;
pub use training::*;

use plotters::backend::BitMapBackend;
use plotters::chart::{ChartBuilder, ChartContext};
use plotters::coord::Shift;
use plotters::coord::ranged1d::{AsRangedCoord, ValueFormatter};
use plotters::prelude::*;
use std::error::Error;

pub struct RenderConfig<'a> {
    pub width: u32,
    pub height: u32,
    pub padding_factor: f32,
    pub text_style: &'a str,
    pub text_size: u32,
    pub area_size: u32,
}

impl Default for RenderConfig<'_> {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            padding_factor: 0.05,
            text_style: "sans-serif",
            text_size: 20,
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
        let root = BitMapBackend::with_buffer(&mut buffer, (cfg.width, cfg.height)).into_drawing_area();
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
        .caption(title, (cfg.text_style, cfg.text_size).into_font())
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
