//! Console rendering for the CLI.
//!
//! Every domain entity describes itself through [`Describe`], returning a
//! multi-line string: a styled title line followed by aligned, dotted-leader
//! rows. Entities build that string with the shared [`block`] helper, so the
//! layout — used by the entity verbs (`loaded`, `generated`, `initialized`) and
//! the training hyperparameter recap alike — lives in one place. One entity per
//! submodule; colors live in [`theme`].

mod checkpoint;
mod dataset;
mod evaluation;
pub(crate) mod hyperparameters;
mod icons;
mod model;
mod progress;
mod scaler;
mod theme;

pub(crate) use evaluation::{eval_set_summary, split_summary};
pub(crate) use icons::*;
pub(crate) use progress::{bar, styled_bar};

use console::{Emoji, style};
use pathdiff::diff_paths;
use std::env;
use std::path::{Path, PathBuf};

// ─── Entity descriptions ────────────────────────────────────────────────────

/// An entity that can describe itself for the console as a styled, multi-line
/// block. Implementors build their string with [`block`].
pub(crate) trait Describe {
    fn describe(&self) -> String;
}

/// Renders a labelled block: a styled `title` line followed by one dotted-leader
/// row per `(label, value)`. The returned string carries no leading icon — the
/// verb that prints it supplies the prefix.
fn block(title: &str, rows: &[(&str, String)]) -> String {
    let width = label_width(rows.iter().map(|(label, _)| *label));

    let mut out = theme::title(title);
    for (label, value) in rows {
        out.push('\n');
        out.push_str(&row(label, value, width, None));
    }
    out
}

/// The widest label in `labels`, used to align the dotted leaders of a block.
fn label_width<'a>(labels: impl Iterator<Item = &'a str>) -> usize {
    labels.map(str::len).max().unwrap_or(0)
}

/// One block row: `label`, a dotted leader padding it to `width`, then `value`.
/// A `previous` value (when it changed) is appended as a `▲ was …` annotation.
fn row(label: &str, value: &str, width: usize, previous: Option<&str>) -> String {
    let leader = ".".repeat(width + 3 - label.len());
    let mut line = format!(
        "   {} {} {}",
        theme::label(label),
        theme::leader(&leader),
        theme::value(value),
    );

    if let Some(previous) = previous {
        line.push_str(&format!("   {}", theme::diff(format!("▲ was {previous}"))));
    }

    line
}

// ─── Entity verbs ─────────────────────────────────────────────────────────────

pub(crate) fn loaded<D: Describe>(subject: &D) {
    println!("{} Loaded {}", theme::icon(LOAD_ICON), subject.describe(),);
}

pub(crate) fn initialized<D: Describe>(subject: &D) {
    println!(
        "{} Initialized {}",
        theme::icon(INIT_ICON),
        subject.describe(),
    );
}

pub(crate) fn generated<D: Describe>(subject: &D) {
    println!("{} Generated {}", theme::icon(GEN_ICON), subject.describe(),);
}

// ─── File event verbs ───────────────────────────────────────────────────────

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
        theme::icon(icon),
        style(name).bold().blue(),
        style(relative_path.display()).bright().magenta().italic()
    );
}

pub(crate) fn recording_at<P: AsRef<Path>>(icon: Emoji, name: &str, at: P) {
    let relative_path = to_relative_path(&at);
    println!(
        "{} Recording {} at {}",
        theme::icon(icon),
        style(name).bold().blue(),
        style(relative_path.display()).bright().magenta().italic()
    );
}

// ─── Message verbs ──────────────────────────────────────────────────────────

pub(crate) fn completed(message: &str) {
    println!("{} {}", theme::icon(SUCCESS_ICON), message);
}

pub(crate) fn trace(message: &str) {
    println!("{} {}", theme::icon(TRACE_ICON), message);
}

pub(crate) fn warning(message: &str) {
    eprintln!(
        "{} {} {}",
        theme::warn_icon(WARN_ICON),
        style("Warning:").bold().yellow(),
        style(message).yellow()
    );
}

pub(crate) fn error(message: &str) {
    eprintln!(
        "{} {} {}",
        theme::error_icon(ERROR_ICON),
        style("Error:").bold().red(),
        style(message).red()
    );
}
