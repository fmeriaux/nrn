//! Rendering a [`Figure`] to the current terminal.

use crate::display::warning;
use console::{Term, measure_text_width};
use nrn::plot::{ConsoleConfig, Figure};
use std::thread::sleep;
use std::time::Duration;

/// The widest a preview grows, regardless of terminal width.
const MAX_COLUMNS: u16 = 60;
/// The narrowest canvas, matching the dot floor `textplots` enforces.
const MIN_COLUMNS: u16 = 16;
const MIN_ROWS: u16 = 8;
/// Width reserved to the right of the canvas for the axis labels.
const AXIS_LABEL_COLUMNS: u16 = 8;
/// The narrowest terminal a preview renders into, below which it warns and skips.
const MIN_PREVIEW_COLUMNS: u16 = MIN_COLUMNS + AXIS_LABEL_COLUMNS;
/// The least width each panel needs to be worth placing side by side.
const SIDE_BY_SIDE_MIN_COLUMNS: u16 = 40;
/// Blank columns between side-by-side panels, mirroring the renderer's gutter.
const GUTTER_COLUMNS: u16 = 3;

/// Prints `figure` to stdout as an inline preview sized to the current terminal,
/// or warns when the terminal is too narrow to render into.
pub(crate) fn preview(figure: &Figure) {
    let Some(config) = preview_config(figure) else {
        warn_too_narrow();
        return;
    };
    println!("{}", figure.to_console(&config));
}

/// Plays `frames` as an inline animation, redrawing each in place with `delay`
/// milliseconds between them and leaving the final frame on screen. Frames are
/// rendered to a console canvas sized to the current terminal, or skipped with a
/// warning when the terminal is too narrow.
pub(crate) fn play_frames(figures: &[Figure], delay: u16) {
    let Some(first) = figures.first() else {
        return;
    };

    // One canvas size for every frame, so each redraw lands on the same lines.
    let Some(config) = preview_config(first) else {
        warn_too_narrow();
        return;
    };
    let term = Term::stdout();
    let width = term.size_checked().map(|(_, columns)| columns);
    let mut previous_lines = 0;

    for figure in figures {
        if previous_lines > 0 {
            let _ = term.clear_last_lines(previous_lines);
        }
        let frame = figure.to_console(&config);
        print!("{frame}");
        let _ = term.flush();
        previous_lines = visual_lines(&frame, width);
        sleep(Duration::from_millis(delay.into()));
    }

    println!();
}

/// The number of terminal rows `frame` occupies, counting a line that overflows
/// `width` as the several rows it wraps onto so an in-place redraw clears it all.
fn visual_lines(frame: &str, width: Option<u16>) -> usize {
    frame.lines().map(|line| wrapped_rows(line, width)).sum()
}

/// The rows a single `line` fills once wrapped at `width`, ignoring ANSI escapes.
fn wrapped_rows(line: &str, width: Option<u16>) -> usize {
    let Some(width) = width.map(usize::from).filter(|&w| w > 0) else {
        return 1;
    };
    measure_text_width(line).div_ceil(width).max(1)
}

/// Warns that the terminal can't fit the smallest legible canvas.
fn warn_too_narrow() {
    warning!(
        "Terminal too narrow to render the plot (needs at least {MIN_PREVIEW_COLUMNS} columns)"
    );
}

/// A console canvas for an inline preview, or `None` when the terminal is known
/// to be narrower than the smallest canvas it could render. A figure falls back
/// to a default when the terminal size can't be determined (e.g. piped output).
fn preview_config(figure: &Figure) -> Option<ConsoleConfig> {
    let panels = figure.panels.len().max(1) as u16;
    let size = Term::stdout().size_checked();
    renders_into(size).then(|| config_for(size, figure.data_aspect(), panels))
}

/// Whether `size` can fit the smallest legible canvas: an unknown size renders
/// at the default, a known one only when it clears the canvas floor.
fn renders_into(size: Option<(u16, u16)>) -> bool {
    match size {
        Some((_, columns)) => columns >= MIN_PREVIEW_COLUMNS,
        None => true,
    }
}

/// Sizes a preview within half the terminal height and a readable width: a
/// spatial figure keeps its data `aspect`, while an aspect-free chart fills the
/// budget. When the terminal is wide enough the `panels` sit side by side at
/// full height; otherwise they stack and split the height between them.
fn config_for(size: Option<(u16, u16)>, aspect: Option<f32>, panels: u16) -> ConsoleConfig {
    let Some((rows, columns)) = size else {
        return ConsoleConfig::default();
    };

    let panels = panels.max(1);
    let usable = columns.saturating_sub(AXIS_LABEL_COLUMNS);

    let (width_budget, row_panels, layout) = match side_by_side_width(usable, panels) {
        Some(per_panel) => (per_panel, 1, panels),
        None => (usable, panels, 1),
    };
    let max_columns = width_budget.clamp(MIN_COLUMNS, MAX_COLUMNS);
    let max_rows = (rows / 2 / row_panels).max(MIN_ROWS);

    let config = match aspect {
        Some(aspect) => ConsoleConfig::fitted(aspect, max_columns, max_rows),
        None => ConsoleConfig::filling(max_columns, max_rows),
    };
    config.with_columns(layout as usize)
}

/// The per-panel width when `panels` fit side by side within `usable` columns —
/// each at least [`SIDE_BY_SIDE_MIN_COLUMNS`] wide after the gutters between
/// them — or `None` when they should stack and use the full width instead.
fn side_by_side_width(usable: u16, panels: u16) -> Option<u16> {
    if panels <= 1 {
        return Some(usable);
    }
    let gutters = (panels - 1) * GUTTER_COLUMNS;
    let per_panel = usable.saturating_sub(gutters) / panels;
    (per_panel >= SIDE_BY_SIDE_MIN_COLUMNS).then_some(per_panel)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_square_domain_stays_square_and_capped_at_half_the_height() {
        // 24 rows → at most 12 rows for a single panel → 48 dots tall; a square
        // aspect makes it 48 dots wide too, well within the width budget.
        let cfg = config_for(Some((24, 80)), Some(1.0), 1);
        assert_eq!((cfg.width(), cfg.height()), (48, 48));
    }

    #[test]
    fn a_wide_domain_is_capped_by_the_width_budget() {
        // A 4:1 domain would want 4× the height in width; the column cap holds
        // it to MAX_COLUMNS (60) → 120 dots, with height following the aspect.
        let cfg = config_for(Some((24, 200)), Some(4.0), 1);
        assert_eq!(cfg.width(), 120);
        assert_eq!(cfg.height(), 30);
    }

    #[test]
    fn wide_terminals_place_panels_side_by_side_at_full_height() {
        // 120 columns, 2 panels: (112 usable − 3 gutter) / 2 = 54 cols → 108
        // dots each. They share the row, so each keeps the whole half-height
        // budget: 40 rows → 20 → 80 dots.
        let cfg = config_for(Some((40, 120)), None, 2);
        assert_eq!(cfg.columns(), 2);
        assert_eq!((cfg.width(), cfg.height()), (108, 80));
    }

    #[test]
    fn standard_width_terminals_stack_panels_and_split_the_height() {
        // 80 columns, 2 panels: (72 usable − 3 gutter) / 2 = 34 cols, below the
        // side-by-side minimum, so they stack and the height halves between them.
        let cfg = config_for(Some((40, 80)), None, 2);
        assert_eq!(cfg.columns(), 1);
        // Stacked: 72 usable cols capped at MAX_COLUMNS (60) → 120 dots; 40 rows
        // → 20 → 10 per panel → 40 dots.
        assert_eq!((cfg.width(), cfg.height()), (120, 40));
    }

    #[test]
    fn a_chart_without_an_aspect_fills_the_width_budget() {
        // No aspect to honour: the canvas fills the column budget (capped at
        // MAX_COLUMNS 60 → 120 dots) instead of falling back to a default.
        let cfg = config_for(Some((24, 200)), None, 1);
        assert_eq!(cfg.width(), 120);
    }

    #[test]
    fn preview_floors_tiny_terminals() {
        let cfg = config_for(Some((1, 1)), Some(1.0), 1);
        assert!(cfg.width() >= 32 && cfg.height() >= 3);
    }

    #[test]
    fn a_terminal_below_the_canvas_floor_does_not_render() {
        // Narrower than MIN_PREVIEW_COLUMNS: even the smallest canvas plus its
        // axis labels would overflow, so skip it.
        assert!(!renders_into(Some((24, MIN_PREVIEW_COLUMNS - 1))));
        // At or above the threshold it renders; an unknown size falls back and renders.
        assert!(renders_into(Some((24, MIN_PREVIEW_COLUMNS))));
        assert!(renders_into(None));
    }

    #[test]
    fn visual_lines_counts_wrapped_rows() {
        // On a 30-column terminal a 32-char line wraps onto two rows; a short
        // line and a blank line each take one.
        let frame = format!("{}\nshort\n\n", "x".repeat(32));
        assert_eq!(visual_lines(&frame, Some(30)), 2 + 1 + 1);
        // An unknown width can't wrap, so every line counts as a single row.
        assert_eq!(visual_lines(&frame, None), 3);
    }

    #[test]
    fn preview_defaults_without_a_known_size() {
        let cfg = config_for(None, Some(1.0), 1);
        let default = ConsoleConfig::default();
        assert_eq!(
            (cfg.width(), cfg.height()),
            (default.width(), default.height())
        );
    }
}
