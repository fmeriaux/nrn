//! Console rendering for the CLI.
//!
//! A value renders its detail through [`Describe`] as a styled string — a single
//! line, or aligned dotted-leader [`rows`]. A [`Named`] value pairs that detail
//! with a NAME header via [`titled`]: a single-line description collapses onto
//! the header (`NAME · detail`), a multi-line one follows below it. The
//! lifecycle verbs ([`loaded`]/[`generated`]/[`initialized`]) head the NAME with
//! `VERB` under an event icon, [`show`] and [`saved`] head the NAME alone. One
//! entity per submodule; colors live in [`theme`].

mod artifacts;
mod checkpoint;
mod classes;
mod classification;
mod dataset;
mod evaluation;
mod hyperparameters;
mod icons;
mod instance;
mod model;
mod predictor;
mod progress;
mod terminal;
mod theme;

pub(crate) use artifacts::Artifacts;
pub(crate) use hyperparameters::HyperParametersView;
pub(crate) use icons::*;
pub(crate) use instance::ReadInstance;
pub(crate) use progress::{Encoding, Epochs, Frames, Spinner};
pub(crate) use terminal::{play_frames, preview};

use console::Emoji;
use std::fmt::Display;
use std::io::{Write, stdout};

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
    let width = column_width(entries.iter().map(|(label, _)| *label));

    entries
        .iter()
        .map(|(label, value)| row(label, value, width, None))
        .collect::<Vec<_>>()
        .join("\n")
}

/// The widest of `texts`, used to align a column to it.
fn column_width<'a>(texts: impl Iterator<Item = &'a str>) -> usize {
    texts.map(str::len).max().unwrap_or(0)
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

/// Places an already-styled `title` header with an entity's `description`: a
/// single-line description collapses onto the title as `title · description`; a
/// multi-line description follows on its own lines.
fn titled(title: String, description: String) -> String {
    if description.contains('\n') {
        format!("{title}\n{description}")
    } else {
        format!("{title} · {description}")
    }
}

/// The event `icon`, a header pairing the entity's NAME with `verb`, then its
/// description.
fn reported<E: Named + Describe>(icon: Emoji, verb: &str, entity: &E) {
    action(
        icon,
        titled(
            theme::title(format!("{} {verb}", E::NAME)),
            entity.describe(),
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

/// An `icon` line heading the entity's NAME with its description.
fn headed<E: Named + Describe>(icon: Emoji, entity: &E) {
    action(icon, titled(theme::title(E::NAME), entity.describe()));
}

/// The trace icon, the entity's NAME, then its description.
pub(crate) fn show<E: Named + Describe>(entity: &E) {
    headed(TRACE_ICON, entity);
}

/// The results icon, the entity's NAME, then its description.
pub(crate) fn evaluated<E: Named + Describe>(entity: &E) {
    headed(EVAL_ICON, entity);
}

/// The save icon, the ARTIFACTS header, then the written files.
pub(crate) fn saved(artifacts: &Artifacts) {
    headed(SAVE_ICON, artifacts);
}

/// The record icon, then an active `RECORDING NAME` header and its
/// description.
pub(crate) fn recording<E: Named + Describe>(entity: &E) {
    action(
        RECORD_ICON,
        titled(
            theme::active(format!("RECORDING {}", E::NAME)),
            entity.describe(),
        ),
    );
}

// ─── Message verbs ──────────────────────────────────────────────────────────

/// The inline prompt for the `index`-th feature value, leaving the cursor on the
/// same line so the typed value follows it.
pub(crate) fn prompt(index: usize) {
    print!(
        "{} {} {} ",
        theme::title("Feature"),
        theme::value(index),
        theme::leader("▸")
    );
    let _ = stdout().flush();
}

/// Backing implementation for the [`completed!`] macro: a success-icon status
/// line, followed by a blank line. Prefer the macro at call sites.
#[doc(hidden)]
pub(crate) fn emit_completed(message: &str) {
    action(SUCCESS_ICON, theme::success(message));
}

/// Backing implementation for the [`completed_with!`] macro: a success status
/// line with a dim `· caption` set apart from the success headline — a secondary
/// fact about the same event. Prefer the macro at call sites.
#[doc(hidden)]
pub(crate) fn emit_completed_with(message: &str, caption: Option<&str>) {
    let mut line = theme::success(message);
    if let Some(caption) = caption {
        line.push_str(&format!(" {}", theme::caption(format!("· {caption}"))));
    }
    action(SUCCESS_ICON, line);
}

/// Backing implementation for the [`warning!`] macro: a styled `Warning:` line
/// on stderr, followed by a blank line. Prefer the macro at call sites.
#[doc(hidden)]
pub(crate) fn emit_warning(message: &str) {
    eprintln!(
        "{} {} {}\n",
        theme::warn_icon(WARN_ICON),
        theme::warn_label("Warning:"),
        theme::warn_text(message)
    );
}

/// Backing implementation for the [`error!`] macro: a styled `Error:` line on
/// stderr, followed by a blank line. Prefer the macro at call sites.
#[doc(hidden)]
pub(crate) fn emit_error(message: &str) {
    eprintln!(
        "{} {} {}\n",
        theme::error_icon(ERROR_ICON),
        theme::error_label("Error:"),
        theme::error_text(message)
    );
}

/// Emit a success status line, taking `format!`-style arguments.
macro_rules! completed {
    ($($arg:tt)*) => { $crate::display::emit_completed(&::std::format!($($arg)*)) };
}

/// Emit a success status line with a dim `· caption` detail set apart from the
/// success headline. The `caption` (an `Option<&str>`) comes first; the remaining
/// `format!`-style arguments build the headline.
macro_rules! completed_with {
    ($caption:expr, $($arg:tt)*) => {
        $crate::display::emit_completed_with(&::std::format!($($arg)*), $caption)
    };
}

/// Emit a styled `Warning:` line to stderr, taking `format!`-style arguments.
macro_rules! warning {
    ($($arg:tt)*) => { $crate::display::emit_warning(&::std::format!($($arg)*)) };
}

/// Emit a styled `Error:` line to stderr, taking `format!`-style arguments.
macro_rules! error {
    ($($arg:tt)*) => { $crate::display::emit_error(&::std::format!($($arg)*)) };
}

pub(crate) use {completed, completed_with, error, warning};
