use console::{Emoji, style};
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use nrn::data::scalers::{Scaler, ScalerMethod};
use nrn::data::{Dataset, DatasetOrigin};
use nrn::io::checkpoint::CheckpointArchive;
use nrn::model::NeuralNetwork;
use pathdiff::diff_paths;
use std::borrow::Cow;
use std::env;
use std::path::{Path, PathBuf};

pub(crate) const LOAD_ICON: Emoji = Emoji("📥", "[L]");
pub(crate) const GEN_ICON: Emoji = Emoji("🌱", "[G]");
pub(crate) const INIT_ICON: Emoji = Emoji("🚀", "[I]");
pub(crate) const MODEL_ICON: Emoji = Emoji("🧠", "[M]");
pub(crate) const SCALER_ICON: Emoji = Emoji("📏", "[S]");
pub(crate) const DATASET_ICON: Emoji = Emoji("📁", "[D]");
pub(crate) const PLOT_ICON: Emoji = Emoji("🧊", "[P]");
pub(crate) const RUN_ICON: Emoji = Emoji("📈", "[T]");
pub(crate) const ANIMATION_ICON: Emoji = Emoji("🎬", "[A]");
pub(crate) const WARN_ICON: Emoji = Emoji("⚠️", "[!]");
pub(crate) const ERROR_ICON: Emoji = Emoji("❌", "[X]");
pub(crate) const SUCCESS_ICON: Emoji = Emoji("✅", "[✓]");
pub(crate) const TRACE_ICON: Emoji = Emoji("🔍", "[*]");

pub(crate) trait Summary {
    fn summary(&self) -> String;
}

impl Summary for Dataset {
    fn summary(&self) -> String {
        let base = format!(
            "{} | Features: {} | Classes: {} | Samples: {}",
            style("DATASET").bold().blue(),
            style(self.n_features()).yellow(),
            style(self.n_classes()).yellow(),
            style(self.n_samples()).yellow()
        );
        match self.origin() {
            Some(DatasetOrigin::Synthetic { distribution, seed }) => format!(
                "{base} | Origin: synthetic {} (seed {})",
                style(distribution).yellow(),
                style(seed).yellow()
            ),
            Some(DatasetOrigin::Encoded { source }) => {
                format!("{base} | Origin: encoded from {}", style(source).yellow())
            }
            None => base,
        }
    }
}

impl Summary for ScalerMethod {
    fn summary(&self) -> String {
        format!(
            "{} | {}",
            style("SCALER").bold().blue(),
            style(self.name()).yellow()
        )
    }
}

impl Summary for NeuralNetwork {
    fn summary(&self) -> String {
        format!(
            "{} | {}",
            style("NEURAL NETWORK").bold().blue(),
            style(self.summary()).yellow(),
        )
    }
}

impl Summary for CheckpointArchive {
    fn summary(&self) -> String {
        format!(
            "{} | Checkpoints: {} | Epochs: {}..{}",
            style("TRAINING RUN").bold().blue(),
            style(self.len()).yellow(),
            style(self.epoch_at(0).unwrap_or(0)).yellow(),
            style(self.epoch_at(self.len().saturating_sub(1)).unwrap_or(0)).yellow(),
        )
    }
}

fn to_relative_path<P: AsRef<Path>>(path: P) -> PathBuf {
    env::current_dir()
        .ok()
        .and_then(|cwd| diff_paths(&path, cwd))
        .unwrap_or_else(|| path.as_ref().to_path_buf())
}

pub(crate) fn saved_at<P: AsRef<Path>>(icon: Emoji, name: &str, at: P) {
    let relative_path = to_relative_path(&at);
    println!(
        "{} Exported {} at {}",
        style(icon).bright().green(),
        style(name).bold().blue(),
        style(relative_path.display()).bright().magenta().italic()
    );
}

pub(crate) fn recording_at<P: AsRef<Path>>(icon: Emoji, name: &str, at: P) {
    let relative_path = to_relative_path(&at);
    println!(
        "{} Recording {} at {}",
        style(icon).bright().green(),
        style(name).bold().blue(),
        style(relative_path.display()).bright().magenta().italic()
    );
}

/// Builds a hidden progress bar with the project's standard style, drawn to stdout.
pub(crate) fn styled_bar() -> ProgressBar {
    let bar = ProgressBar::hidden();
    bar.set_style(
        ProgressStyle::with_template(
            "{msg} {spinner:.green} [{elapsed_precise}] {wide_bar} {pos}/{len} {percent}% ({eta})",
        )
        .unwrap(),
    );
    bar.set_draw_target(ProgressDrawTarget::stdout());
    bar
}

/// A standalone progress bar of known length, for use with [`indicatif::ProgressIterator`].
pub(crate) fn bar(len: usize, msg: impl Into<Cow<'static, str>>) -> ProgressBar {
    let bar = styled_bar();
    bar.set_length(len as u64);
    bar.set_message(msg);
    bar
}

pub(crate) fn loaded<S: Summary>(subject: &S) {
    println!(
        "{} Loaded {}",
        style(LOAD_ICON).bright().cyan(),
        subject.summary(),
    );
}

pub(crate) fn initialized<S: Summary>(subject: &S) {
    println!(
        "{} Initialized {}",
        style(INIT_ICON).bright().green(),
        subject.summary(),
    );
}

pub(crate) fn generated<S: Summary>(subject: &S) {
    println!(
        "{} Generated {}",
        style(GEN_ICON).bright().green(),
        subject.summary(),
    );
}

pub(crate) fn completed(message: &str) {
    println!("{} {}", style(SUCCESS_ICON).bright().green(), message);
}

pub(crate) fn trace(message: &str) {
    println!("{} {}", style(TRACE_ICON).bright().blue(), message);
}

pub(crate) fn warning(message: &str) {
    eprintln!(
        "{} {} {}",
        style(WARN_ICON).bright().yellow(),
        style("Warning:").bold().yellow(),
        style(message).yellow()
    );
}

pub(crate) fn error(message: &str) {
    eprintln!(
        "{} {} {}",
        style(ERROR_ICON).bright().red(),
        style("Error:").bold().red(),
        style(message).red()
    );
}
