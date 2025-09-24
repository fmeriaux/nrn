use crate::core::neuron_network::NeuronNetwork;
use crate::core::data::Dataset;
use gif::{Encoder, Frame, Repeat};
use ndarray::{Array1, Array2};
use plotters::chart::ChartBuilder;
use plotters::prelude::LineSeries;
use plotters::prelude::full_palette::BLUE_900;
use plotters::prelude::{BLACK, Circle, Color, IntoDrawingArea, RGBColor};
use plotters::prelude::{BitMapBackend, WHITE};
use plotters::style::full_palette::*;
use std::collections::HashSet;
use std::fs::File;

const PALETTE_900: &[RGBColor] = &[
    ORANGE_900, BLUE_900, PINK_900, PURPLE_900, INDIGO_900, CYAN_900, TEAL_900, GREEN_900,
    LIME_900, YELLOW_900, AMBER_900, RED_900, BROWN_900, GREY_900,
];

const CHART_MARGIN: u32 = 30;
const LABEL_AREA_SIZE: u32 = 30;

/// Computes the minimum and maximum values for a slice of f32 values.
fn compute_min_max(slice: &[f32]) -> (f32, f32) {
    let min = slice.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    (min, max)
}

/// Returns a color from the palette based on the label index.
fn color_for_label(label: usize) -> RGBColor {
    PALETTE_900[label % PALETTE_900.len()]
}

/// Creates a buffer filled with white pixels for the scatter plot background.
fn create_rgb_buffer(width: u32, height: u32) -> Vec<u8> {
    vec![255u8; (width * height * 3) as usize]
}

/// Creates a marker for the scatter plot with the specified coordinates and label.
fn mk_marker<C>(x: C, y: usize) -> Circle<C, i32> {
    Circle::new(x, 2, color_for_label(y).filled())
}

/// Returns a vector of unique labels from the given array.
fn unique_labels(y: &Array1<f32>) -> Vec<f32> {
    let mut set = HashSet::new();
    let mut uniques = Vec::new();
    for &v in y.iter() {
        // Use to_bits to handle NaN and -0.0 correctly
        if set.insert(v.to_bits()) {
            uniques.push(v);
        }
    }
    uniques
}

/// Creates a grid of points and corresponding inputs for a neural network.
fn make_grid_and_inputs(min: f32, max: f32, resolution: usize) -> (Vec<(f32, f32)>, Array2<f32>) {
    let step = (max - min) / resolution as f32;
    let mut grid_points = Vec::with_capacity((resolution + 1) * (resolution + 1));
    for j in 0..resolution {
        for i in 0..resolution {
            let x_val = min + (i as f32) * step;
            let y_val = min + (j as f32) * step;
            grid_points.push((x_val, y_val));
        }
    }
    let xs: Vec<f32> = grid_points.iter().map(|&(x, _)| x).collect();
    let ys: Vec<f32> = grid_points.iter().map(|&(_, y)| y).collect();
    let mut flat: Vec<f32> = Vec::with_capacity(2 * grid_points.len());
    flat.extend(xs);
    flat.extend(ys);
    let inputs = Array2::from_shape_vec((2, grid_points.len()), flat).unwrap();
    (grid_points, inputs)
}

/// Draws a scatter plot of the features with labels.
fn draw_data(
    dataset: &Dataset,
    width: u32,
    height: u32,
    model: Option<&NeuronNetwork>,
    show_legend: bool,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
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

        for label in unique_labels(&dataset.labels) {
            let color = color_for_label(label as usize);
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
            let (grid_points, inputs) = make_grid_and_inputs(min, max, 500);
            let predictions = model.predict(inputs.view());

            let decision_boundary: Vec<(f32, f32)> = grid_points
                .iter()
                .enumerate()
                .filter(|(i, _)| {
                    let pred = predictions.column(*i);
                    pred.iter().any(|&p| (p - 0.5).abs() < 0.001)
                })
                .map(|(_, &pt)| pt)
                .collect();

            chart.draw_series(
                decision_boundary
                    .iter()
                    .map(|&(x, y)| Circle::new((x, y), 1, RED_900.filled())),
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
/// - `name`: The name of the training session, used in the filename.
/// - `width`: The width of the chart in pixels.
/// - `height`: The height of the chart in pixels.
/// - `series`: A slice of tuples, where each tuple contains a label and a slice of f32 values representing the training history for that label.
pub(crate) fn of_history(
    name: &str,
    width: u32,
    height: u32,
    series: &[(&str, &[f32])],
) -> Result<(), Box<dyn std::error::Error>> {
    let filepath = &format!("{}.png", name);
    let title = format!("Training History - {}", name);
    let root = BitMapBackend::new(&filepath, (width, height)).into_drawing_area();
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
        let color = PALETTE_900[idx % PALETTE_900.len()];
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
    Ok(())
}

/// Generates a PNG file representing the dataset features and labels.
pub(crate) fn of_data(
    name: &str,
    dataset: &Dataset,
    width: u32,
    height: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let buffer = draw_data(&dataset, width, height, None, true)?;
    let filepath = format!("{}.png", name);
    let file = File::create(filepath)?;
    let mut encoder = png::Encoder::new(file, width, height);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;
    writer.write_image_data(&buffer)?;
    Ok(())
}

/// Represents a training view that can display decision boundaries of a neural network
/// and save them as GIF frames.
/// # Properties
/// - `width`: The width of the view in pixels.
/// - `height`: The height of the view in pixels.
/// - `scale`: A tuple representing the minimum and maximum values for scaling the decision boundary.
pub struct DecisionBoundaryView {
    pub width: u32,
    pub height: u32,
    dataset: Dataset,
    frames: Vec<Vec<u8>>,
}

impl DecisionBoundaryView {
    /// Creates a new `DecisionBoundaryView`, initializing the background with a scatter plot of the dataset features and labels.
    /// # Arguments
    /// - `width`: The width of the view in pixels.
    /// - `height`: The height of the view in pixels.
    /// - `dataset`: A reference to the `Dataset` containing features and labels.
    pub fn new(width: u32, height: u32, dataset: Dataset) -> DecisionBoundaryView {
        DecisionBoundaryView {
            width,
            height,
            dataset,
            frames: Vec::new(),
        }
    }

    /// Adds a frame to the decision boundary view by drawing it based on the provided `NeuronNetwork`.
    /// # Arguments
    /// - `model`: A reference to the `NeuronNetwork` whose decision boundary will be drawn.
    pub fn add_frame(&mut self, model: &NeuronNetwork) -> Result<(), Box<dyn std::error::Error>> {
        let buffer = draw_data(&self.dataset, self.width, self.height, Some(model), false)?;
        self.frames.push(buffer.clone());
        Ok(())
    }

    /// Saves the frames as a GIF file with the given name.
    /// # Arguments
    /// - `name`: The name of the file to save the GIF as, without the `.gif` extension.
    pub fn save(&mut self, name: &str) -> Result<(), Box<dyn std::error::Error>> {
        let (width, height) = (self.width as u16, self.height as u16);

        let mut encoder =
            Encoder::new(File::create(&format!("{}.gif", name))?, width, height, &[])?;

        encoder.set_repeat(Repeat::Infinite)?;

        for frame in self.frames.iter() {
            let mut gif_frame = Frame::from_rgb(width, height, frame);
            gif_frame.delay = 20;
            encoder.write_frame(&gif_frame)?;
        }

        Ok(())
    }
}
