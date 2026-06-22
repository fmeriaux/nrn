//! Rendering a [`Figure`] to the current terminal.

use console::Term;
use nrn::plot::{ConsoleConfig, Figure};
use std::thread::sleep;
use std::time::Duration;

/// Prints `figure` to stdout as an inline preview sized to the current terminal.
pub(crate) fn preview(figure: &Figure) {
    println!("{}", figure.to_console(&preview_config(figure)));
}

/// Plays `frames` as an inline animation, redrawing each in place with `delay`
/// milliseconds between them and leaving the final frame on screen. Frames are
/// rendered to a console canvas sized to the current terminal.
pub(crate) fn play_frames(figures: &[Figure], delay: u16) {
    let Some(first) = figures.first() else {
        return;
    };

    // One canvas size for every frame, so each redraw lands on the same lines.
    let config = preview_config(first);
    let term = Term::stdout();
    let mut previous_lines = 0;

    for figure in figures {
        if previous_lines > 0 {
            let _ = term.clear_last_lines(previous_lines);
        }
        let frame = figure.to_console(&config);
        print!("{frame}");
        let _ = term.flush();
        previous_lines = frame.lines().count();
        sleep(Duration::from_millis(delay.into()));
    }

    println!();
}

/// A console canvas for an inline preview, falling back to a default when the
/// terminal size can't be determined (e.g. piped output).
fn preview_config(figure: &Figure) -> ConsoleConfig {
    let panels = figure.panels.len().max(1) as u16;
    config_for(Term::stdout().size_checked(), figure.data_aspect(), panels)
}

/// Sizes a preview within half the terminal height, split evenly across its
/// `panels`, and a readable width: a spatial figure keeps its data `aspect`,
/// while an aspect-free chart fills the budget.
fn config_for(size: Option<(u16, u16)>, aspect: Option<f32>, panels: u16) -> ConsoleConfig {
    let Some((rows, columns)) = size else {
        return ConsoleConfig::default();
    };

    let max_rows = (rows / 2 / panels.max(1)).max(MIN_ROWS);
    let max_columns = columns.saturating_sub(2).clamp(MIN_COLUMNS, MAX_COLUMNS);

    match aspect {
        Some(aspect) => ConsoleConfig::fitted(aspect, max_columns, max_rows),
        None => ConsoleConfig::filling(max_columns, max_rows),
    }
}

/// The widest a preview grows, regardless of terminal width.
const MAX_COLUMNS: u16 = 60;
/// Floors keeping the preview legible on a tiny terminal.
const MIN_COLUMNS: u16 = 16;
const MIN_ROWS: u16 = 8;

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
    fn the_height_budget_is_split_across_panels() {
        // 40 rows → 20-row half budget, shared by two panels → 10 rows → 40 dots
        // each, so the stacked figure still fits within half the terminal.
        let cfg = config_for(Some((40, 80)), None, 2);
        assert_eq!(cfg.height(), 40);
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
    fn preview_defaults_without_a_known_size() {
        let cfg = config_for(None, Some(1.0), 1);
        let default = ConsoleConfig::default();
        assert_eq!(
            (cfg.width(), cfg.height()),
            (default.width(), default.height())
        );
    }
}
