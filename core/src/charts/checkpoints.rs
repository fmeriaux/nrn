use crate::charts::{RenderConfig, add_padding, draw_chart, draw_with};
use crate::checkpoints::Checkpoints;
use plotters::coord::types::{RangedCoordf32, RangedCoordusize};
use plotters::element::Circle;
use plotters::prelude::full_palette::{GREEN_900, RED_900};
use plotters::prelude::*;
use plotters::style::full_palette::ORANGE_900;
use std::error::Error;

impl Checkpoints {
    /// Draws the training history (loss over epochs) as a line chart.
    /// Returns the plot as a vector of bytes in RGB format.
    pub fn draw(&self, cfg: &RenderConfig) -> Result<Vec<u8>, Box<dyn Error>> {
        draw_checkpoints(cfg, self)
    }
}

fn draw_serie<'a, 'b>(
    chart: &mut ChartContext<'a, BitMapBackend<'b>, Cartesian2d<RangedCoordusize, RangedCoordf32>>,
    data: &[f32],
    color: RGBColor,
    label: &str,
) -> Result<(), Box<dyn Error>> {
    if data.is_empty() {
        return Ok(());
    }

    chart
        .draw_series(LineSeries::new(
            data.iter()
                .enumerate()
                .map(|(i, &a)| (i, a))
                .collect::<Vec<(usize, f32)>>(),
            &color.to_rgba(),
        ))?
        .label(label)
        .legend(move |(x, y)| Circle::new((x, y), 2, color.to_rgba()));
    Ok(())
}

fn draw_checkpoints(
    cfg: &RenderConfig,
    checkpoints: &Checkpoints,
) -> Result<Vec<u8>, Box<dyn Error>> {
    draw_with(cfg, |root| {
        let (left, right) = root.split_vertically(50.percent());

        // Get the min and max values for loss and accuracy to set the y-axis range
        let (loss, acc) = checkpoints
            .loss_range()
            .zip(checkpoints.accuracy_range())
            .ok_or("No data to plot")?;

        // Add 5% padding to min and max values for better visualization
        let (mins, maxs) = add_padding(
            &vec![loss.0, acc.0],
            &vec![loss.1, acc.1],
            cfg.padding_factor,
        );

        draw_chart(
            &left,
            "Training Loss Over Epochs",
            0..checkpoints.len(),
            mins[0]..maxs[0],
            cfg,
            true,
            |loss_chart| {
                draw_serie(loss_chart, &checkpoints.train_losses(), RED_900, "Train")?;

                draw_serie(
                    loss_chart,
                    &checkpoints.validation_losses(),
                    ORANGE_900,
                    "Validation",
                )?;

                draw_serie(loss_chart, &checkpoints.test_losses(), GREEN_900, "Test")?;

                Ok(())
            },
        )?;

        draw_chart(
            &right,
            "Training and Test Accuracy Over Epochs",
            0..checkpoints.len(),
            mins[1]..maxs[1],
            cfg,
            true,
            |accuracy_chart| {
                draw_serie(
                    accuracy_chart,
                    &checkpoints.train_accuracies(),
                    RED_900,
                    "Train",
                )?;

                draw_serie(
                    accuracy_chart,
                    &checkpoints.validation_accuracies(),
                    ORANGE_900,
                    "Validation",
                )?;

                draw_serie(
                    accuracy_chart,
                    &checkpoints.test_accuracies(),
                    GREEN_900,
                    "Test",
                )?;

                Ok(())
            },
        )
    })
}
