//! Console rendering for the CLI.
//!
//! A value renders its detail through [`Describe`] as a styled string — a single
//! line, or aligned dotted-leader [`rows`]. A [`Named`] value pairs that detail
//! with a NAME header: the lifecycle verbs ([`loaded`]/[`generated`]/
//! [`initialized`]) prefix `NAME VERB` under an event icon, [`show`] and [`saved`]
//! prefix the NAME alone. One entity per submodule; colors live in [`theme`].

mod artifacts;
mod checkpoint;
mod dataset;
mod evaluation;
mod hyperparameters;
mod icons;
mod model;
mod progress;
mod scaler;
mod theme;

pub(crate) use artifacts::Artifacts;
pub(crate) use hyperparameters::HyperParametersView;
pub(crate) use icons::*;
pub(crate) use progress::{bar, styled_bar};

use crate::path::PathExt;
use console::{Emoji, style};
use std::fmt::Display;
use std::path::Path;

// ─── Entity descriptions ────────────────────────────────────────────────────

/// A value that can describe itself for the console as a styled string — a
/// single line, or the multi-line detail of [`rows`].
pub(crate) trait Describe {
    /// The styled text describing this value.
    fn describe(&self) -> String;
}

/// An entity with a NAME.
pub(crate) trait Named {
    /// This entity's display name.
    const NAME: &'static str;
}

/// One dotted-leader row per `(label, value)`, aligned to the widest label.
fn rows(entries: &[(&str, String)]) -> String {
    let width = label_width(entries.iter().map(|(label, _)| *label));

    entries
        .iter()
        .map(|(label, value)| row(label, value, width, None))
        .collect::<Vec<_>>()
        .join("\n")
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

// ─── Action & entity verbs ────────────────────────────────────────────────────

/// Traces an action as a single status line: a styled icon then `message`,
/// followed by a blank line so consecutive blocks stay visually separated.
pub(crate) fn action(icon: Emoji, message: impl Display) {
    println!("{} {}\n", theme::icon(icon), message);
}

/// The event `icon`, a header pairing the entity's NAME with `verb`, then its
/// detail rows.
fn reported<E: Named + Describe>(icon: Emoji, verb: &str, entity: &E) {
    action(
        icon,
        format!(
            "{}\n{}",
            theme::title(&format!("{} {verb}", E::NAME)),
            entity.describe()
        ),
    );
}

pub(crate) fn loaded<E: Named + Describe>(entity: &E) {
    reported(LOAD_ICON, "LOADED", entity);
}

pub(crate) fn generated<E: Named + Describe>(entity: &E) {
    reported(GEN_ICON, "GENERATED", entity);
}

pub(crate) fn initialized<E: Named + Describe>(entity: &E) {
    reported(INIT_ICON, "INITIALIZED", entity);
}

/// The trace icon, the entity's NAME, then its detail rows.
pub(crate) fn show<E: Named + Describe>(entity: &E) {
    action(
        TRACE_ICON,
        format!("{}\n{}", theme::title(E::NAME), entity.describe()),
    );
}

/// The save icon, the ARTIFACTS header, then the written files.
pub(crate) fn saved(artifacts: &Artifacts) {
    action(
        SAVE_ICON,
        format!(
            "{}\n{}",
            theme::title(Artifacts::NAME),
            artifacts.describe()
        ),
    );
}

// ─── File event verbs ───────────────────────────────────────────────────────

/// A record event: the record icon, the entity `name`, and the directory path.
pub(crate) fn recording_at<P: AsRef<Path>>(name: &str, at: P) {
    action(
        RECORD_ICON,
        format!(
            "Recording {} at {}",
            theme::title(name),
            theme::value(at.as_ref().to_relative().display())
        ),
    );
}

// ─── Message verbs ──────────────────────────────────────────────────────────

pub(crate) fn completed(message: &str) {
    action(SUCCESS_ICON, message);
}

pub(crate) fn trace(message: &str) {
    action(TRACE_ICON, message);
}

pub(crate) fn warning(message: &str) {
    eprintln!(
        "{} {} {}\n",
        theme::warn_icon(WARN_ICON),
        style("Warning:").bold().yellow(),
        style(message).yellow()
    );
}

pub(crate) fn error(message: &str) {
    eprintln!(
        "{} {} {}\n",
        theme::error_icon(ERROR_ICON),
        style("Error:").bold().red(),
        style(message).red()
    );
}
