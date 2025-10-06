use crate::actions;
use crate::display::warning;
use crate::display::{ANIMATION_ICON, HISTORY_ICON, saved_at};
use crate::progression::Progression;
use clap::Args;
use nrn::charts::RenderConfig;
use nrn::data::SplitDataset;
use nrn::io::data::SplitDatasetExt;
use nrn::io::gif::save_gif_from_rgb;
use nrn::io::png::save_rgb;
use std::error::Error;

#[derive(Args, Debug)]
pub struct PlotArgs {
    /// Name of the training history file to plot
    history: String,

    /// Name of the dataset used for training for decision boundary visualization (only for 2D datasets)
    #[arg(short, long)]
    dataset: Option<String>,

    /// Specify the number of frames for the decision boundary animation
    #[arg(short, long, default_value_t = 20, requires = "dataset", value_parser = 2..201)]
    frames: u8,

    /// Specify the delay between frames in the decision boundary animation (in milliseconds)
    #[arg(long, default_value_t = 50, requires = "dataset", value_parser = 1..=1000)]
    delay: u16,

    /// Specify the width of the plot in pixels
    #[arg(long, default_value_t = 1200, value_parser = 100..=4096)]
    width: u16,

    /// Specify the height of the plot in pixels
    #[arg(long, default_value_t = 900, value_parser = 100..=4096)]
    height: u16,
}

impl PlotArgs {
    pub fn run(self) -> Result<(), Box<dyn Error>> {
        let history = actions::load_training_history(&self.history)?;
        let render_cfg = RenderConfig::new(self.width as u32, self.height as u32);

        let (width, height) = (self.width as u32, self.height as u32);

        let frame = history.draw(&render_cfg)?;

        saved_at(
            HISTORY_ICON,
            "TRAINING CURVES",
            save_rgb(frame, &self.history, width, height)?,
        );

        if let Some(dataset) = self.dataset {
            let dataset = SplitDataset::load(&dataset)?;

            if dataset.train.n_features() != 2 {
                warning(
                    "Decision boundary visualization is only available for datasets with exactly two features",
                );
                return Ok(());
            }

            let interval = history.model.len() / history.model.len().min(self.frames.into());

            let progression = Progression::new(
                history.model.len(),
                "Generating decision boundary animation",
            );

            let mut decision_frames = Vec::new();

            for step in progression.iter() {
                let step_number = step + 1;

                if step_number == 1
                    || step_number % interval == 0
                    || step_number == history.model.len()
                {
                    let model = &history.model[step];

                    let rgb_frame = model.draw_decision_boundary(&dataset.train, &render_cfg)?;

                    decision_frames.push(rgb_frame);
                }
            }

            saved_at(
                ANIMATION_ICON,
                "DECISION BOUNDARY ANIMATION",
                save_gif_from_rgb(
                    decision_frames,
                    self.width,
                    self.height,
                    self.delay,
                    &format!("{}", &self.history),
                )?,
            );
        }

        Ok(())
    }
}
