use crate::actions::{load_dataset, load_history};
use crate::console::bar;
use crate::console::warning;
use crate::console::{ANIMATION_ICON, RUN_ICON, saved_at};
use clap::Args;
use indicatif::ProgressIterator;
use nrn::charts::RenderConfig;
use nrn::io::gif::save_gif_from_rgb;
use nrn::io::png::save_rgb;
use std::error::Error;

#[derive(Args, Debug)]
pub struct PlotArgs {
    /// Path to the training run directory
    run_dir: String,

    /// Name of the dataset used for training for decision boundary visualization (only for 2D datasets)
    #[arg(short, long)]
    dataset: Option<String>,

    /// Specify the number of frames for the decision boundary animation
    #[arg(short, long, default_value_t = 20, requires = "dataset", value_parser = clap::value_parser!(u8).range(2..=201))]
    frames: u8,

    /// Specify the delay between frames in the decision boundary animation (in milliseconds)
    #[arg(long, default_value_t = 50, requires = "dataset", value_parser = clap::value_parser!(u16).range(10..=1000))]
    delay: u16,

    /// Specify the width of the plot in pixels
    #[arg(long, default_value_t = 1200, value_parser = clap::value_parser!(u16).range(100..=4096))]
    width: u16,

    /// Specify the height of the plot in pixels
    #[arg(long, default_value_t = 900, value_parser = clap::value_parser!(u16).range(100..=4096))]
    height: u16,
}

impl PlotArgs {
    pub fn run(self) -> Result<(), Box<dyn Error>> {
        let archive = load_history(&self.run_dir)?;
        let history = archive.evaluation_history()?;
        let render_cfg = RenderConfig::new(self.width as u32, self.height as u32);

        let (width, height) = (self.width as u32, self.height as u32);

        let frame = history.draw(&render_cfg)?;

        saved_at(
            RUN_ICON,
            "TRAINING CURVES",
            save_rgb(frame, &self.run_dir, width, height)?,
        );

        if let Some(dataset) = self.dataset {
            let dataset = load_dataset(&dataset)?;

            if dataset.n_features() != 2 {
                warning(
                    "Decision boundary visualization is only available for datasets with exactly two features",
                );
                return Ok(());
            }

            let n = archive.len();
            let interval = n / n.min(self.frames.into());

            let progress = bar(n, "Generating decision boundary animation");

            let mut decision_frames = Vec::new();

            for step in (0..n).progress_with(progress) {
                let step_number = step + 1;

                if step_number == 1 || step_number % interval == 0 || step_number == n {
                    // Load one model at a time — no full checkpoint array in memory.
                    let model = archive.model_at(step)?;
                    let rgb_frame = model.draw_decision_boundary(&dataset, &render_cfg)?;
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
                    &self.run_dir,
                )?,
            );
        }

        Ok(())
    }
}
