use crate::charts::{RenderConfig, add_padding, draw_chart, draw_with};
use crate::training::History;
use plotters::element::Circle;
use plotters::prelude::full_palette::{BLUE_900, GREEN_900, RED_900};
use plotters::prelude::*;
use std::error::Error;

impl History {
    /// Draws the training history (loss over epochs) as a line chart.
    /// Returns the plot as a vector of bytes in RGB format.
    pub fn draw(&self, cfg: &RenderConfig) -> Result<Vec<u8>, Box<dyn Error>> {
        draw_history(cfg, self)
    }
}

fn draw_history(cfg: &RenderConfig, history: &History) -> Result<Vec<u8>, Box<dyn Error>> {
    draw_with(cfg, |root| {
        let (left, right) = root.split_vertically(50.percent());

        // Get the min and max values for loss and accuracy to set the y-axis range
        let (loss, acc) = history
            .loss_range()
            .zip(history.accuracy_range())
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
            0..history.loss.len(),
            mins[0]..maxs[0],
            cfg,
            false,
            |loss_chart| {
                loss_chart.draw_series(LineSeries::new(
                    history.loss.iter().enumerate().map(|(i, &l)| (i, l)),
                    &BLUE_900.to_rgba(),
                ))?;
                Ok(())
            },
        )?;

        draw_chart(
            &right,
            "Training and Test Accuracy Over Epochs",
            0..history.loss.len(),
            mins[1]..maxs[1],
            cfg,
            true,
            |accuracy_chart| {
                accuracy_chart
                    .draw_series(LineSeries::new(
                        history
                            .train_accuracy
                            .iter()
                            .enumerate()
                            .map(|(i, &a)| (i, a)),
                        &RED_900.to_rgba(),
                    ))?
                    .label("Train")
                    .legend(move |(x, y)| Circle::new((x, y), 2, RED_900.to_rgba()));

                accuracy_chart
                    .draw_series(LineSeries::new(
                        history
                            .test_accuracy
                            .iter()
                            .enumerate()
                            .map(|(i, &a)| (i, a)),
                        &GREEN.to_rgba(),
                    ))?
                    .label("Test")
                    .legend(move |(x, y)| Circle::new((x, y), 2, GREEN_900.to_rgba()));

                Ok(())
            },
        )
    })
}
