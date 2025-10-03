use crate::analysis::decision_boundary;
use crate::charts::{RenderConfig, add_padding, draw_chart, draw_with};
use crate::data::Dataset;
use crate::model::NeuralNetwork;
use plotters::element::Circle;
use plotters::prelude::full_palette::RED_900;
use plotters::prelude::*;
use std::error::Error;

impl Dataset {
    /// Draws a scatter plot of the dataset features with labels.
    /// Returns the plot as a vector of bytes in RGB format.
    pub fn draw(&self, cfg: &RenderConfig) -> Result<Vec<u8>, Box<dyn Error>> {
        draw_data_with(self, None, cfg, true)
    }
}

impl NeuralNetwork {
    /// Draws the decision boundary of the neural network over the provided dataset.
    /// The dataset must have exactly two features.
    /// Returns the plot as a vector of bytes in RGB format.
    pub fn draw_decision_boundary(
        &self,
        dataset: &Dataset,
        cfg: &RenderConfig,
    ) -> Result<Vec<u8>, Box<dyn Error>> {
        draw_data_with(&dataset, Some(self), cfg, false)
    }
}

/// Draws a scatter plot of the features with labels.
/// If a model is provided, it also draws the decision boundary.
fn draw_data_with(
    dataset: &Dataset,
    model: Option<&NeuralNetwork>,
    cfg: &RenderConfig,
    show_legend: bool,
) -> Result<Vec<u8>, Box<dyn Error>> {
    assert_eq!(
        dataset.n_features(),
        2,
        "Scatter plot can only be generated for datasets with exactly two features"
    );

    let (mins, maxs) = dataset
        .feature_range()
        .and_then(|(mins, maxs)| add_padding(&mins, &maxs, cfg.padding_factor).into())
        .ok_or("Dataset is empty, cannot determine feature range")?;

    draw_with(cfg, |root| {
        draw_chart(
            &root,
            "Scatter Plot of Dataset Features",
            mins[0]..maxs[0],
            mins[1]..maxs[1],
            cfg,
            show_legend,
            |chart| {
                for label in dataset.unique_labels() {
                    let color = Palette100::pick(label as usize).to_rgba().filled();
                    chart
                        .draw_series(
                            dataset
                                .get_features_for_label(label)
                                .outer_iter()
                                .map(|pt| (pt[0], pt[1]))
                                .map(|c| Circle::new(c, 2, color)),
                        )?
                        .label(format!("Class {}", label))
                        .legend(move |(x, y)| Circle::new((x, y), 2, color));
                }

                if let Some(model) = model {
                    let decision_boundary = decision_boundary(&mins, &maxs, model, 800, 0.01);
                    let color = RED_900.to_rgba().filled();

                    chart.draw_series(
                        decision_boundary
                            .outer_iter()
                            .map(|pt| (pt[0], pt[1]))
                            .map(|c| Circle::new(c, 1, color)),
                    )?;
                }
                Ok(())
            },
        )
    })
}
