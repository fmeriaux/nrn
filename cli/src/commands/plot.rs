use crate::display::{Artifacts, bar, loaded, saved, warning};
use clap::Args;
use indicatif::ProgressIterator;
use nrn::data::Dataset;
use nrn::data::scalers::ScalerMethod;
use nrn::io::run::TrainingRun;
use nrn::model::Predictor;
use nrn::plot::{ImageConfig, RasterAnimation};
use std::error::Error;
use std::path::Path;

#[derive(Args, Debug)]
pub struct PlotArgs {
    /// Path to the training run directory
    run_dir: String,

    /// Specify the number of frames for the decision boundary animation
    #[arg(short, long, default_value_t = 20, value_parser = clap::value_parser!(u8).range(2..=201))]
    frames: u8,

    /// Specify the delay between frames in the decision boundary animation (in milliseconds)
    #[arg(long, default_value_t = 50, value_parser = clap::value_parser!(u16).range(10..=1000))]
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
        let run = TrainingRun::open(&self.run_dir)?;
        let archive = run.archive()?;

        if archive.len() <= 2 {
            return Err("Training run must contain more than two checkpoints to plot.".into());
        }

        loaded(&archive);

        let history = archive.evaluation_history()?;
        let (width, height) = (self.width as u32, self.height as u32);
        let render_cfg = ImageConfig::new(width, height);

        let frame = history.figure()?.to_image(&render_cfg)?;

        let mut artifacts = Artifacts::from([("Training Curves", frame.save(&self.run_dir)?)]);

        // The run records its dataset (a sibling of the run directory) and scaler.
        let meta = run.meta();
        let dataset_path = Path::new(&self.run_dir)
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(&meta.dataset);
        let dataset = Dataset::load(&dataset_path)?;
        loaded(&dataset);

        if dataset.n_features() != 2 {
            warning!(
                "Decision boundary visualization is only available for datasets with exactly two features"
            );
        } else {
            let scaler: Option<ScalerMethod> = meta.scaler.clone().map(Into::into);

            let n = archive.len();
            let interval = n / n.min(self.frames.into());

            let progress = bar(n, "Generating decision boundary animation");

            let mut decision_frames = Vec::new();

            for step in (0..n).progress_with(progress) {
                let step_number = step + 1;

                if step_number == 1 || step_number % interval == 0 || step_number == n {
                    // Load one model at a time — no full checkpoint array in memory.
                    let model = archive.model_at(step)?;
                    let predictor = Predictor::new(model, scaler.clone());
                    // Resolution tracks the output width: one boundary grid line per pixel
                    // column is ample for a crisp curve at any figure size.
                    let figure = predictor.boundary_figure(&dataset, width as usize)?;
                    decision_frames.push(figure.to_image(&render_cfg)?);
                }
            }

            artifacts.add(
                "Decision Boundary Animation",
                RasterAnimation {
                    frames: decision_frames,
                    frame_delay: self.delay,
                }
                .save(&self.run_dir)?,
            );
        }

        saved(&artifacts);

        Ok(())
    }
}
