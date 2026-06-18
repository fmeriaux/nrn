//! Centralized console palette. Every color used by the entity blocks and the
//! message verbs lives here, so the look is changed in one place and the
//! [`Description`](super::Description) values stay plain (ANSI-free, testable).

use console::{Emoji, style};
use std::fmt::Display;

/// A leading line icon in the standard accent — the single place ordinary icons
/// get styled, so verbs pass only the glyph. Severity icons use [`warn_icon`] /
/// [`error_icon`] instead, where the conventional color carries meaning.
pub(crate) fn icon(glyph: Emoji) -> String {
    style(glyph).bright().green().to_string()
}

/// A warning icon, in the conventional warning color.
pub(crate) fn warn_icon(glyph: Emoji) -> String {
    style(glyph).bright().yellow().to_string()
}

/// An error icon, in the conventional error color.
pub(crate) fn error_icon(glyph: Emoji) -> String {
    style(glyph).bright().red().to_string()
}

/// The bold blue entity title (e.g. `DATASET`, `TRAINING HYPERPARAMETERS`).
pub(crate) fn title(text: &str) -> String {
    style(text).bold().blue().to_string()
}

/// A field label (the left column of a block row).
pub(crate) fn label(text: &str) -> String {
    style(text).cyan().to_string()
}

/// The dotted leader joining a label to its value.
pub(crate) fn leader(dots: &str) -> String {
    style(dots).dim().to_string()
}

/// A field value (the right column of a block row, and inline figures).
pub(crate) fn value(text: impl Display) -> String {
    style(text).yellow().to_string()
}

/// The `▲ was …` annotation shown when a value differs from a previous run.
pub(crate) fn diff(text: impl Display) -> String {
    style(text).yellow().to_string()
}
