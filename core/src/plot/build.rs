//! Builders turning domain objects into a [`Figure`].
//!
//! These are pure and backend-agnostic: they shape the data into panels and series,
//! leaving rasterization to a renderer.

use crate::data::Dataset;
use crate::evaluation_history::EvaluationHistory;
use crate::model::Predictor;
use crate::plot::scene::{Color, Figure, Panel, Series, add_padding};
use std::error::Error;

/// Fraction of each axis range added as whitespace around a figure's axes.
const DEFAULT_PADDING_FACTOR: f32 = 0.05;

impl Dataset {
    /// A scatter figure of the dataset's two features, with default axis padding.
    ///
    /// # Errors
    /// When the dataset does not have exactly two features.
    pub fn figure(&self) -> Result<Figure, Box<dyn Error>> {
        self.figure_with_padding(DEFAULT_PADDING_FACTOR)
    }

    /// A scatter figure widening each axis by `padding_factor` of its range.
    ///
    /// # Errors
    /// When the dataset does not have exactly two features.
    pub fn figure_with_padding(&self, padding_factor: f32) -> Result<Figure, Box<dyn Error>> {
        Ok(Figure::spatial(vec![scatter_panel(
            self,
            None,
            padding_factor,
            true,
        )?]))
    }
}

impl Predictor {
    /// A scatter figure overlaid with this predictor's decision boundary, with
    /// default axis padding.
    ///
    /// # Errors
    /// When the dataset does not have exactly two features.
    pub fn boundary_figure(
        &self,
        dataset: &Dataset,
        resolution: usize,
    ) -> Result<Figure, Box<dyn Error>> {
        self.boundary_figure_with_padding(dataset, resolution, DEFAULT_PADDING_FACTOR)
    }

    /// A scatter figure overlaid with this predictor's decision boundary,
    /// widening each axis by `padding_factor` of its range.
    ///
    /// `resolution` is the number of grid points per axis used to trace the boundary;
    /// higher values yield a smoother curve at a quadratic cost in forward passes.
    ///
    /// # Errors
    /// When the dataset does not have exactly two features.
    pub fn boundary_figure_with_padding(
        &self,
        dataset: &Dataset,
        resolution: usize,
        padding_factor: f32,
    ) -> Result<Figure, Box<dyn Error>> {
        Ok(Figure::spatial(vec![scatter_panel(
            dataset,
            Some((self, resolution)),
            padding_factor,
            false,
        )?]))
    }
}

impl EvaluationHistory {
    /// A two-panel figure of training loss and accuracy over epochs, with
    /// default axis padding.
    ///
    /// # Errors
    /// When the history is empty.
    pub fn figure(&self) -> Result<Figure, Box<dyn Error>> {
        self.figure_with_padding(DEFAULT_PADDING_FACTOR)
    }

    /// A two-panel figure: training loss (top) and accuracy (bottom) over epochs,
    /// widening each value axis by `padding_factor` of its range.
    ///
    /// # Errors
    /// When the history is empty.
    pub fn figure_with_padding(&self, padding_factor: f32) -> Result<Figure, Box<dyn Error>> {
        let epochs = self.epochs();
        let ((loss, accuracy), last_epoch) = self
            .loss_range()
            .zip(self.accuracy_range())
            .zip(epochs.last())
            .ok_or("No data to plot")?;

        let (mins, maxs) =
            add_padding(&[loss.0, accuracy.0], &[loss.1, accuracy.1], padding_factor);

        let x_range = (0.0, (last_epoch + 1) as f32);

        let loss_panel = Panel {
            title: "Training Loss Over Epochs".to_string(),
            x_range,
            y_range: (mins[0], maxs[0]),
            x_label: Some("Epoch".to_string()),
            y_label: Some("Loss".to_string()),
            show_legend: true,
            series: line_series(
                &epochs,
                [
                    (self.train_losses(), Color::TRAIN, "Train"),
                    (self.validation_losses(), Color::VALIDATION, "Validation"),
                    (self.test_losses(), Color::TEST, "Test"),
                ],
            ),
        };

        let accuracy_panel = Panel {
            title: "Training and Test Accuracy Over Epochs".to_string(),
            x_range,
            y_range: (mins[1], maxs[1]),
            x_label: Some("Epoch".to_string()),
            y_label: Some("Accuracy".to_string()),
            show_legend: true,
            series: line_series(
                &epochs,
                [
                    (self.train_accuracies(), Color::TRAIN, "Train"),
                    (
                        self.validation_accuracies(),
                        Color::VALIDATION,
                        "Validation",
                    ),
                    (self.test_accuracies(), Color::TEST, "Test"),
                ],
            ),
        };

        Ok(Figure::chart(vec![loss_panel, accuracy_panel]))
    }
}

/// Builds the scatter panel, optionally overlaying a predictor's decision boundary.
fn scatter_panel(
    dataset: &Dataset,
    boundary: Option<(&Predictor, usize)>,
    padding_factor: f32,
    show_legend: bool,
) -> Result<Panel, Box<dyn Error>> {
    if dataset.n_features() != 2 {
        return Err("Scatter plot requires a dataset with exactly two features".into());
    }

    let (raw_mins, raw_maxs) = dataset.feature_range();
    let (mins, maxs) = add_padding(&raw_mins, &raw_maxs, padding_factor);

    let mut series: Vec<Series> = dataset
        .unique_labels()
        .into_iter()
        .map(|label| Series::Points {
            points: dataset
                .get_features_for_label(label)
                .outer_iter()
                .map(|point| (point[0], point[1]))
                .collect(),
            color: Color::category(label as usize),
            label: Some(format!("Class {label}")),
            radius: 2,
        })
        .collect();

    if let Some((predictor, resolution)) = boundary {
        let boundary_points = predictor.decision_boundary(&mins, &maxs, resolution);
        series.push(Series::Points {
            points: boundary_points
                .outer_iter()
                .map(|point| (point[0], point[1]))
                .collect(),
            color: Color::BOUNDARY,
            label: None,
            radius: 1,
        });
    }

    Ok(Panel {
        title: "Scatter Plot of Dataset Features".to_string(),
        x_range: (mins[0], maxs[0]),
        y_range: (mins[1], maxs[1]),
        x_label: Some("Feature 0".to_string()),
        y_label: Some("Feature 1".to_string()),
        show_legend,
        series,
    })
}

/// The non-empty line series, pairing each value sequence with the epoch axis.
fn line_series(epochs: &[usize], series: [(Vec<f32>, Color, &str); 3]) -> Vec<Series> {
    series
        .into_iter()
        .filter(|(data, _, _)| !data.is_empty())
        .map(|(data, color, label)| Series::Line {
            points: data
                .iter()
                .zip(epochs)
                .map(|(&value, &epoch)| (epoch as f32, value))
                .collect(),
            color,
            label: Some(label.to_string()),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::IDENTITY;
    use crate::data::Dataset;
    use crate::evaluation::{Evaluation, EvaluationSet};
    use crate::evaluation_history::{EpochEvaluation, EvaluationHistory};
    use crate::layers::Dense;
    use crate::model::NeuralNetwork;
    use crate::task::Task;
    use ndarray::array;

    /// A two-feature, two-class dataset.
    fn two_feature_dataset() -> Dataset {
        Dataset::tabular(
            array![[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]],
            array![0.0, 1.0, 0.0, 1.0],
            None,
        )
        .unwrap()
    }

    /// A one-feature, two-class dataset (too few features for a scatter plot).
    fn one_feature_dataset() -> Dataset {
        Dataset::tabular(
            array![[0.0], [1.0], [0.0], [1.0]],
            array![0.0, 1.0, 0.0, 1.0],
            None,
        )
        .unwrap()
    }

    /// A 2-input → 1-output binary predictor with an identity (logits) output.
    fn binary_predictor() -> Predictor {
        Predictor::new(
            NeuralNetwork::single(Dense::new(
                array![[1.0, 0.0]],
                array![0.0],
                IDENTITY.clone(),
            )),
            Task::Binary,
            None,
        )
    }

    fn checkpoint(epoch: usize) -> EpochEvaluation {
        EpochEvaluation {
            epoch,
            evaluation: EvaluationSet {
                train: Evaluation {
                    loss: 1.0,
                    accuracy: 50.0,
                },
                validation: Some(Evaluation {
                    loss: 1.1,
                    accuracy: 45.0,
                }),
                test: Evaluation {
                    loss: 1.2,
                    accuracy: 40.0,
                },
            },
        }
    }

    #[test]
    fn dataset_figure_has_one_panel_per_class_with_a_legend() {
        let figure = two_feature_dataset().figure().unwrap();
        assert_eq!(figure.panels.len(), 1);
        let panel = &figure.panels[0];
        assert!(panel.show_legend);
        // One scatter series per class, no boundary overlay.
        assert_eq!(panel.series.len(), 2);
        assert!(
            panel
                .series
                .iter()
                .all(|series| matches!(series, Series::Points { label: Some(_), .. }))
        );
    }

    #[test]
    fn scatter_panel_labels_axes_by_feature_index() {
        let panel = &two_feature_dataset().figure().unwrap().panels[0];
        assert_eq!(panel.x_label.as_deref(), Some("Feature 0"));
        assert_eq!(panel.y_label.as_deref(), Some("Feature 1"));
    }

    #[test]
    fn history_panels_label_epoch_against_their_own_metric() {
        let history = EvaluationHistory::new(vec![checkpoint(0), checkpoint(1)]);
        let figure = history.figure().unwrap();
        assert_eq!(figure.panels[0].x_label.as_deref(), Some("Epoch"));
        assert_eq!(figure.panels[0].y_label.as_deref(), Some("Loss"));
        assert_eq!(figure.panels[1].x_label.as_deref(), Some("Epoch"));
        assert_eq!(figure.panels[1].y_label.as_deref(), Some("Accuracy"));
    }

    #[test]
    fn spatial_figures_preserve_aspect_but_the_curves_chart_does_not() {
        // Scatter and boundary axes are both feature space (shared units), so the
        // aspect is preserved; loss/accuracy over epochs is an aspect-free chart.
        assert!(two_feature_dataset().figure().unwrap().preserve_aspect);
        assert!(
            binary_predictor()
                .boundary_figure(&two_feature_dataset(), 10)
                .unwrap()
                .preserve_aspect
        );
        let history = EvaluationHistory::new(vec![checkpoint(0), checkpoint(1)]);
        assert!(!history.figure().unwrap().preserve_aspect);
    }

    #[test]
    fn dataset_figure_rejects_non_two_feature_data() {
        let error = one_feature_dataset().figure().unwrap_err();
        assert!(error.to_string().contains("exactly two features"));
    }

    #[test]
    fn figure_with_padding_widens_the_axes() {
        let dataset = two_feature_dataset();
        let default = dataset.figure().unwrap();
        let wide = dataset.figure_with_padding(0.5).unwrap();
        // A larger factor pushes the lower x bound further out.
        assert!(wide.panels[0].x_range.0 < default.panels[0].x_range.0);
    }

    #[test]
    fn boundary_figure_adds_an_unlabeled_overlay_without_a_legend() {
        let dataset = two_feature_dataset();
        let figure = binary_predictor().boundary_figure(&dataset, 10).unwrap();
        let panel = &figure.panels[0];
        assert!(!panel.show_legend);
        // Two class series plus the boundary overlay.
        assert_eq!(panel.series.len(), 3);
        assert!(matches!(
            panel.series.last().unwrap(),
            Series::Points {
                label: None,
                color: Color::BOUNDARY,
                radius: 1,
                points,
            } if !points.is_empty()
        ));
    }

    #[test]
    fn boundary_figure_rejects_non_two_feature_data() {
        let dataset = one_feature_dataset();
        let error = binary_predictor()
            .boundary_figure(&dataset, 10)
            .unwrap_err();
        assert!(error.to_string().contains("exactly two features"));
    }

    #[test]
    fn history_figure_has_loss_and_accuracy_panels() {
        let history = EvaluationHistory::new(vec![checkpoint(0), checkpoint(1)]);
        let figure = history.figure().unwrap();
        assert_eq!(figure.panels.len(), 2);
        assert_eq!(figure.panels[0].title, "Training Loss Over Epochs");
        assert_eq!(
            figure.panels[1].title,
            "Training and Test Accuracy Over Epochs"
        );
        // Train, validation and test lines on each panel.
        assert_eq!(figure.panels[0].series.len(), 3);
        assert_eq!(figure.panels[1].series.len(), 3);
        // The x-axis spans one past the last epoch.
        assert_eq!(figure.panels[0].x_range, (0.0, 2.0));
    }

    #[test]
    fn history_figure_omits_empty_series() {
        // No validation split → the validation line is dropped.
        let without_validation = EpochEvaluation {
            epoch: 0,
            evaluation: EvaluationSet {
                train: Evaluation {
                    loss: 1.0,
                    accuracy: 50.0,
                },
                validation: None,
                test: Evaluation {
                    loss: 1.2,
                    accuracy: 40.0,
                },
            },
        };
        let history = EvaluationHistory::new(vec![without_validation]);
        let figure = history.figure().unwrap();
        assert_eq!(figure.panels[0].series.len(), 2);
    }

    #[test]
    fn history_figure_rejects_empty_history() {
        let error = EvaluationHistory::new(Vec::new()).figure().unwrap_err();
        assert!(error.to_string().contains("No data to plot"));
    }
}
