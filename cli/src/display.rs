use console::{Emoji, style};
use nrn::data::SplitDataset;
use nrn::data::scalers::{Scaler, ScalerMethod};
use nrn::model::NeuralNetwork;
use nrn::training::History;
use pathdiff::diff_paths;
use std::env;
use std::path::{Path, PathBuf};

pub(crate) const LOAD_ICON: Emoji = Emoji("📥", "[L]");
pub(crate) const GEN_ICON: Emoji = Emoji("🌱", "[G]");
pub(crate) const INIT_ICON: Emoji = Emoji("🚀", "[I]");
pub(crate) const MODEL_ICON: Emoji = Emoji("🧠", "[M]");
pub(crate) const SCALER_ICON: Emoji = Emoji("📏", "[S]");
pub(crate) const DATASET_ICON: Emoji = Emoji("📁", "[D]");
pub(crate) const PLOT_ICON: Emoji = Emoji("🧊", "[P]");
pub(crate) const HISTORY_ICON: Emoji = Emoji("📈", "[T]");
pub(crate) const ANIMATION_ICON: Emoji = Emoji("🎬", "[A]");
pub(crate) const WARN_ICON: Emoji = Emoji("⚠️", "[!]");
pub(crate) const ERROR_ICON: Emoji = Emoji("❌", "[X]");
pub(crate) const SUCCESS_ICON: Emoji = Emoji("✅", "[✓]");
pub(crate) const TRACE_ICON: Emoji = Emoji("🔍", "[*]");

pub(crate) trait Summary {
    fn summary(&self) -> String;
}

impl Summary for SplitDataset {
    fn summary(&self) -> String {
        format!(
            "{} | Features: {} | Classes: {} | Split: Train={}, Test={}",
            style("DATASET").bold().blue(),
            style(self.train.n_features()).yellow(),
            style(self.train.n_classes()).yellow(),
            style(self.train.n_samples()).yellow(),
            style(self.test.n_samples()).yellow()
        )
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

impl Summary for Option<f32> {
    fn summary(&self) -> String {
        match self {
            Some(value) => format!("{}", value),
            None => "N/A".to_string(),
        }
    }
}

impl Summary for History {
    fn summary(&self) -> String {
        format!(
            "{} | Checkpoints: {} | Loss: {} | Accuracy: Train={}%, Test={}%",
            style("TRAINING HISTORY").bold().blue(),
            style(self.model.len()).yellow(),
            style(self.final_loss().summary()).yellow(),
            style(self.final_train_accuracy().summary()).yellow(),
            style(self.final_test_accuracy().summary()).yellow()
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
