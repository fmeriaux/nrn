//! Centralized console palette: every color used by the entity blocks and the
//! message verbs. [`Describe`](super::Describe) values stay ANSI-free.

use console::{Emoji, style};
use std::fmt::Display;

/// The CLI's 256-color palette. Every colored span resolves to one of these
/// indices, shared between the `style`-wrapping helpers below and the
/// `indicatif` progress templates (which take color codes as bare strings, so
/// they can't go through the helpers). Severity icons use named terminal
/// colors instead, where the convention itself carries the meaning.
pub(crate) const TITLE: u8 = 33; // entity titles and progress prefixes
pub(crate) const LABEL: u8 = 37; // field labels
pub(crate) const VALUE: u8 = 172; // field values, inline figures, and diffs
pub(crate) const ACCENT: u8 = 35; // success lines and the progress fill/spinner
pub(crate) const ACTIVE: u8 = 92; // in-progress action verbs
pub(crate) const WARN: u8 = 166; // warning lead-ins and bodies
pub(crate) const ERROR: u8 = 160; // error lead-ins and bodies

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
pub(crate) fn title(text: impl Display) -> String {
    style(text).bold().color256(TITLE).to_string()
}

/// A field label (the left column of a block row).
pub(crate) fn label(text: impl Display) -> String {
    style(text).color256(LABEL).to_string()
}

/// A success status line, in bold green — the `completed!` counterpart to the
/// bold `warn_label`/`error_label` lead-ins.
pub(crate) fn success(text: impl Display) -> String {
    style(text).bold().color256(ACCENT).to_string()
}

/// The dotted leader joining a label to its value.
pub(crate) fn leader(dots: impl Display) -> String {
    style(dots).dim().to_string()
}

/// A dim trailing caption — e.g. an artifact's role shown beside its path.
pub(crate) fn caption(text: impl Display) -> String {
    style(text).dim().to_string()
}

/// A field value (the right column of a block row, and inline figures).
pub(crate) fn value(text: impl Display) -> String {
    style(text).color256(VALUE).to_string()
}

/// An in-progress action verb (e.g. `RECORDING`), in bold violet.
pub(crate) fn active(text: impl Display) -> String {
    style(text).bold().color256(ACTIVE).to_string()
}

/// The `▲ was …` annotation shown when a value differs from a previous run.
pub(crate) fn diff(text: impl Display) -> String {
    style(text).color256(VALUE).to_string()
}

/// The bold `Warning:` lead-in, in the conventional warning color.
pub(crate) fn warn_label(text: impl Display) -> String {
    style(text).bold().color256(WARN).to_string()
}

/// A warning message body.
pub(crate) fn warn_text(text: impl Display) -> String {
    style(text).color256(WARN).to_string()
}

/// The bold `Error:` lead-in, in the conventional error color.
pub(crate) fn error_label(text: impl Display) -> String {
    style(text).bold().color256(ERROR).to_string()
}

/// An error message body.
pub(crate) fn error_text(text: impl Display) -> String {
    style(text).color256(ERROR).to_string()
}
