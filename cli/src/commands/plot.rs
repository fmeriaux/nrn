use crate::actions::chart::draw_data;
use crate::progression::Progression;
use crate::{actions, display_success, display_warning};
use clap::Args;
use colored::Colorize;
use nrn::data::SplitDataset;
use nrn::io::data::SplitDatasetExt;
use nrn::io::gif::save_gif_from_rgb;
use std::error::Error;

#[derive(Args, Debug)]
pub struct PlotArgs {
    /// Name of the training history file to plot
    history: String,

    /// Name of the dataset used for training for decision boundary visualization (only for 2D datasets)
    #[arg(short, long)]
    dataset: Option<String>,

    /// Specify the number of frames for the decision boundary animation
    #[arg(short, long, default_value_t = 20, requires = "dataset", value_parser = clap::value_parser!(u8).range(2..201))]
    frames: u8,

    /// Specify the width of the plot in pixels
    #[arg(long, default_value_t = 800, value_parser = clap::value_parser!(u16).range(100..=4096))]
    width: u16,

    /// Specify the height of the plot in pixels
    #[arg(long, default_value_t = 600, value_parser = clap::value_parser!(u16).range(100..=4096))]
    height: u16,
}

impl PlotArgs {
    pub fn run(self) -> Result<(), Box<dyn Error>> {
        let history = actions::load_training_history(&self.history)?;

        actions::chart::of_history(
            &format!("loss-{}", self.history),
            self.width as u32,
            self.height as u32,
            &[("Loss", &history.loss)],
        )?;

        actions::chart::of_history(
            &format!("accuracy-{}", self.history),
            self.width as u32,
            self.height as u32,
            &[
                ("Train Accuracy", &history.train_accuracy),
                ("Test Accuracy", &history.test_accuracy),
            ],
        )?;

        display_success!(
            "{} at {} {}",
            "Training history plots saved".bright_green(),
            self.history.bright_blue().italic(),
            "(PNG)".italic().dimmed()
        );

        if let Some(dataset) = self.dataset {
            let dataset = SplitDataset::load(&dataset)?;

            if dataset.train.n_features() != 2 {
                display_warning!(
                    "Decision boundary visualization is only available for datasets with exactly two features"
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

                    let rgb_frame = draw_data(
                        &dataset.train,
                        self.width as u32,
                        self.height as u32,
                        Some(&model),
                        false,
                    )?;

                    decision_frames.push(rgb_frame);
                }
            }

            save_gif_from_rgb(
                decision_frames,
                self.width,
                self.height,
                50,
                &format!("{}", self.history),
            )?;

            display_success!(
                "{} at {} {}",
                "Decision boundary animation saved".bright_green(),
                self.history.bright_blue().italic(),
                "(GIF)".italic().dimmed()
            );
        }

        Ok(())
    }
}
