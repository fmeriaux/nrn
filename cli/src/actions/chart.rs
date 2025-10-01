use nrn::analysis::decision_boundary;
use nrn::data::Dataset;
use nrn::io::png;
use nrn::model::NeuralNetwork;
use plotters::chart::ChartBuilder;
use plotters::prelude::full_palette::RED_900;
use plotters::prelude::*;
use std::error::Error;
use std::path::{Path, PathBuf};

const CHART_MARGIN: u32 = 30;
const LABEL_AREA_SIZE: u32 = 30;

/// Computes the minimum and maximum values for a slice of f32 values.
fn compute_min_max(slice: &[f32]) -> (f32, f32) {
    let min = slice.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    (min, max)
}

/// Creates a buffer filled with white pixels for the scatter plot background.
fn create_rgb_buffer(width: u32, height: u32) -> Vec<u8> {
    vec![255u8; (width * height * 3) as usize]
}

/// Creates a marker for the scatter plot with the specified coordinates and label.
fn mk_marker<C>(x: C, y: usize) -> Circle<C, i32> {
    Circle::new(x, 2, Palette100::pick(y).stroke_width(2).filled())
}

/// Draws a scatter plot of the features with labels.
pub fn draw_data(
    dataset: &Dataset,
    width: u32,
    height: u32,
    model: Option<&NeuralNetwork>,
    show_legend: bool,
) -> Result<Vec<u8>, Box<dyn Error>> {
    let mut buffer = create_rgb_buffer(width, height);
    {
        let root = BitMapBackend::with_buffer(&mut buffer, (width, height)).into_drawing_area();
        root.fill(&WHITE)?;
        let (min, max) = compute_min_max(dataset.features.as_slice().unwrap());
        let mut chart = ChartBuilder::on(&root)
            .margin(CHART_MARGIN)
            .caption("Scatter Plot", ("Arial", 20))
            .x_label_area_size(LABEL_AREA_SIZE)
            .y_label_area_size(LABEL_AREA_SIZE)
            .build_cartesian_2d(min..max, min..max)?;

        chart.configure_mesh().draw()?;

        for label in dataset.unique_labels() {
            let color = Palette100::pick(label as usize).to_rgba();
            chart
                .draw_series(
                    dataset
                        .features
                        .outer_iter()
                        .zip(dataset.labels.iter())
                        .filter(|(_, l)| **l == label)
                        .map(|(row, _)| mk_marker((row[0], row[1]), label as usize)),
                )?
                .label(format!("Cluster {}", label))
                .legend(move |(x, y)| Circle::new((x, y), 2, color.filled()));
        }

        if let Some(model) = model {
            let decision_boundary = decision_boundary(&[min, min], &[max, max], 500, model);

            chart.draw_series(
                decision_boundary
                    .iter()
                    .map(|pt| Circle::new((pt[0], pt[1]), 1, RED_900.filled())),
            )?;
        }

        if show_legend {
            chart
                .configure_series_labels()
                .background_style(&WHITE.mix(0.8))
                .border_style(&BLACK)
                .draw()?;
        }

        root.present()?;
    }

    Ok(buffer)
}

/// Generates a PNG file representing the training history of a neural network.
/// # Arguments
/// - `name`: The name of the training session, used as chart title and filename.
/// - `width`: The width of the chart in pixels.
/// - `height`: The height of the chart in pixels.
/// - `series`: A slice of tuples, where each tuple contains a label and a slice of f32 values representing the training history for that label.
pub(crate) fn of_history(
    name: &str,
    width: u32,
    height: u32,
    series: &[(&str, &[f32])],
) -> Result<PathBuf, Box<dyn Error>> {
    let title = format!("Training History - {}", name);
    let mut frame = create_rgb_buffer(width, height);
    {
        let root = BitMapBackend::with_buffer(&mut frame, (width, height)).into_drawing_area();
        root.fill(&WHITE)?;

        // Find the minimum and maximum values across all series to set the y-axis range
        let (min_val, max_val) = series
            .iter()
            .flat_map(|(_, s)| s.iter())
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), &v| {
                (f32::min(min, v), f32::max(max, v))
            });
        let max_len = series.iter().map(|(_, s)| s.len()).max().unwrap_or(0);

        let mut chart = ChartBuilder::on(&root)
            .margin(CHART_MARGIN)
            .caption(title, ("Arial", 20))
            .x_label_area_size(LABEL_AREA_SIZE)
            .y_label_area_size(LABEL_AREA_SIZE)
            .build_cartesian_2d(0..max_len, min_val..max_val)?;

        chart.configure_mesh().draw()?;

        for (idx, (label, s)) in series.iter().enumerate() {
            let color = Palette99::pick(idx).to_rgba();
            chart
                .draw_series(LineSeries::new(
                    s.iter().enumerate().map(|(i, &l)| (i, l)),
                    &color,
                ))?
                .label(*label)
                .legend(move |(x, y)| Circle::new((x, y), 2, color.filled()));
        }

        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()?;

        root.present()?;
    }
    png::save_rgb(frame, name, width, height).map_err(|e| e.into())
}

/// Generates a PNG file representing the dataset features and labels.
pub(crate) fn of_data<P: AsRef<Path>>(
    dataset: &Dataset,
    width: u32,
    height: u32,
    path: P,
) -> Result<PathBuf, Box<dyn Error>> {
    let frame = draw_data(&dataset, width, height, None, true)?;
    png::save_rgb(frame, path, width, height).map_err(|e| e.into())
}
