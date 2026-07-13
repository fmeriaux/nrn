use super::{Format, ImageSize, render};
use crate::display::{Artifacts, Frames, loaded, play_frames, saved, warning};
use crate::path::PathExt;
use clap::Args;
use nrn::data::Dataset;
use nrn::data::scalers::ScalerMethod;
use nrn::io::model::checkpoint::CheckpointArchive;
use nrn::io::model::run::TrainingRun;
use nrn::io::raster::gif::GifWriter;
use nrn::model::Predictor;
use nrn::plot::{ConsoleConfig, Figure};
use nrn::task::Task;
use std::error::Error;
use std::path::{Path, PathBuf};

#[derive(Args, Debug)]
pub struct RunArgs {
    /// Training run directory to plot
    run_dir: String,

    /// Output format
    #[arg(long, value_enum, default_value_t = Format::default())]
    format: Format,

    /// Animate the decision boundary across the run's checkpoints
    #[arg(long, default_value_t = false)]
    animate: bool,

    /// Number of frames in the decision boundary animation (capped at the number
    /// of recorded checkpoints)
    #[arg(short, long, default_value_t = 50, value_parser = clap::value_parser!(u8).range(2..=100))]
    frames: u8,

    /// Delay between animation frames (in milliseconds)
    #[arg(long, default_value_t = 100, value_parser = clap::value_parser!(u16).range(10..=1000))]
    delay: u16,

    #[command(flatten)]
    size: ImageSize,
}

impl RunArgs {
    pub fn run(self) -> Result<(), Box<dyn Error>> {
        let run = TrainingRun::open(&self.run_dir)?;
        let archive = run.archive()?;
        loaded(&archive);

        if archive.len() < 2 {
            return Err("a training run needs at least two checkpoints to plot".into());
        }

        // Load everything up front so the status lines stay grouped above the
        // rendered output rather than interrupting it (console format).
        let meta = run.meta();
        let dataset = Dataset::load(self.dataset_path(&meta.dataset))?;
        loaded(&dataset);

        let mut artifacts = Artifacts::empty();

        // Training curves are always available from the recorded evaluations.
        let curves = archive.evaluation_history()?.figure()?;
        if let Some(path) = render(&curves, self.format, self.size, self.artifact("curves"))? {
            artifacts.add("Training Curves", path);
        }

        if dataset.n_features() == 2 {
            let task = Task::from(meta.task.clone());
            let scaler: Option<ScalerMethod> = meta.scaler.clone().map(Into::into);
            if let Some(path) = self.boundary(&archive, &dataset, task, &scaler)? {
                artifacts.add("Decision Boundary", path);
            }
        } else if self.animate {
            warning!("The decision boundary animation needs a two-feature dataset");
        }

        if !artifacts.is_empty() {
            saved(&artifacts);
        }

        Ok(())
    }

    /// Renders the decision boundary: animated across checkpoints with `--animate`,
    /// or as a still of the final model otherwise. Returns a written path only for
    /// the image format.
    fn boundary(
        &self,
        archive: &CheckpointArchive,
        dataset: &Dataset,
        task: Task,
        scaler: &Option<ScalerMethod>,
    ) -> Result<Option<PathBuf>, Box<dyn Error>> {
        if self.animate {
            let indices = archive.sample(self.frames.into());
            return self.animation(archive, dataset, task, scaler, &indices);
        }

        let last = archive.len() - 1;
        let figure = boundary_at(archive, last, dataset, task, scaler, self.resolution())?;
        render(&figure, self.format, self.size, self.artifact("boundary"))
    }

    /// Renders the boundary at each checkpoint in `indices`: plays them inline for
    /// the console format, or streams them into a GIF for the image format.
    fn animation(
        &self,
        archive: &CheckpointArchive,
        dataset: &Dataset,
        task: Task,
        scaler: &Option<ScalerMethod>,
        indices: &[usize],
    ) -> Result<Option<PathBuf>, Box<dyn Error>> {
        let resolution = self.resolution();

        match self.format {
            Format::Console => {
                let progress = Frames::new(indices.len(), "Rendering frames");
                let mut figures = Vec::with_capacity(indices.len());
                for &index in indices {
                    figures.push(boundary_at(
                        archive, index, dataset, task, scaler, resolution,
                    )?);
                    progress.advance();
                }
                progress.finish();

                play_frames(&figures, self.delay);
                Ok(None)
            }
            Format::Image => {
                // One pass: render, encode and write each frame under a single
                // progress bar — no buffered animation, no silent encode phase.
                let cfg = self.size.config();
                let mut writer = GifWriter::create(
                    self.artifact("boundary"),
                    cfg.width,
                    cfg.height,
                    self.delay,
                )?;

                let progress = Frames::new(indices.len(), "Rendering GIF");
                for &index in indices {
                    let figure = boundary_at(archive, index, dataset, task, scaler, resolution)?;
                    writer.write_frame(&figure.to_image(&cfg)?)?;
                    progress.advance();
                }
                progress.finish();

                Ok(Some(writer.finish()))
            }
        }
    }

    /// The boundary grid resolution for the active format: one grid line per
    /// output column — the image's pixel width, or the text canvas width in dots
    /// for the console.
    fn resolution(&self) -> usize {
        match self.format {
            Format::Console => ConsoleConfig::default().width() as usize,
            Format::Image => self.size.config().width as usize,
        }
    }

    /// A path for an artifact named `{prefix}-{run}`, beside the run directory so
    /// the run's own files stay untouched (the extension is set by the writer).
    fn artifact(&self, prefix: &str) -> PathBuf {
        Path::new(&self.run_dir).sibling(prefix)
    }

    /// The recorded dataset path, a sibling of the run directory.
    fn dataset_path(&self, dataset: &str) -> PathBuf {
        Path::new(&self.run_dir)
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(dataset)
    }
}

/// Builds the decision-boundary figure for the checkpoint at `index`.
fn boundary_at(
    archive: &CheckpointArchive,
    index: usize,
    dataset: &Dataset,
    task: Task,
    scaler: &Option<ScalerMethod>,
    resolution: usize,
) -> Result<Figure, Box<dyn Error>> {
    let model = archive.model_at(index)?;
    let predictor = Predictor::new(model, task, scaler.clone());
    predictor.boundary_figure(dataset, resolution)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        args: RunArgs,
    }

    fn parse(extra: &[&str]) -> RunArgs {
        let mut argv = vec!["run", "runs/r1"];
        argv.extend_from_slice(extra);
        Cli::parse_from(argv).args
    }

    fn try_parse(extra: &[&str]) -> Result<RunArgs, clap::Error> {
        let mut argv = vec!["run", "runs/r1"];
        argv.extend_from_slice(extra);
        Cli::try_parse_from(argv).map(|cli| cli.args)
    }

    #[test]
    fn defaults_are_console_still_with_standard_animation_knobs() {
        let args = parse(&[]);
        assert_eq!(args.format, Format::Console);
        assert!(!args.animate);
        assert_eq!(args.frames, 50);
        assert_eq!(args.delay, 100);
    }

    #[test]
    fn animate_and_format_flags_are_parsed() {
        let args = parse(&[
            "--format",
            "image",
            "--animate",
            "--frames",
            "30",
            "--delay",
            "100",
        ]);
        assert_eq!(args.format, Format::Image);
        assert!(args.animate);
        assert_eq!(args.frames, 30);
        assert_eq!(args.delay, 100);
    }

    #[test]
    fn frames_below_the_minimum_are_rejected() {
        assert!(try_parse(&["--frames", "1"]).is_err());
    }

    #[test]
    fn delay_above_the_maximum_is_rejected() {
        assert!(try_parse(&["--delay", "2000"]).is_err());
    }

    #[test]
    fn resolution_tracks_the_output_width_per_format() {
        let console = parse(&[]);
        assert_eq!(
            console.resolution(),
            ConsoleConfig::default().width() as usize
        );

        let image = parse(&["--format", "image", "--width", "800"]);
        assert_eq!(image.resolution(), 800);
    }

    #[test]
    fn artifacts_land_beside_the_run_directory() {
        let args = parse(&[]);
        assert_eq!(args.artifact("curves"), Path::new("runs/curves-r1"));
        assert_eq!(args.artifact("boundary"), Path::new("runs/boundary-r1"));
    }

    #[test]
    fn dataset_path_is_a_sibling_of_the_run_directory() {
        let args = parse(&[]);
        assert_eq!(
            args.dataset_path("data.safetensors"),
            Path::new("runs/data.safetensors")
        );
    }
}
