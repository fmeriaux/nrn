//! Rendering a [`Figure`] to the current terminal.

use console::Term;
use nrn::plot::{ConsoleConfig, Figure};

/// Prints `figure` to stdout as an inline preview sized to the current terminal.
pub(crate) fn preview(figure: &Figure) {
    println!("{}", figure.to_console(&preview_config(figure)));
}

/// A console canvas for an inline preview, falling back to a default when the
/// terminal size or the figure's aspect can't be determined (e.g. piped output).
fn preview_config(figure: &Figure) -> ConsoleConfig {
    config_for(Term::stdout().size_checked(), figure.data_aspect())
}

/// Fits a preview to the figure's data aspect within half the terminal height
/// and a readable width, so it stays glanceable instead of filling the screen.
fn config_for(size: Option<(u16, u16)>, aspect: Option<f32>) -> ConsoleConfig {
    match (size, aspect) {
        (Some((rows, columns)), Some(aspect)) => {
            let max_rows = (rows / 2).max(MIN_ROWS);
            let max_columns = columns.saturating_sub(2).clamp(MIN_COLUMNS, MAX_COLUMNS);
            ConsoleConfig::fitted(aspect, max_columns, max_rows)
        }
        _ => ConsoleConfig::default(),
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
        // 24 rows → at most 12 rows → 48 dots tall; a square aspect makes it
        // 48 dots wide too, well within the width budget.
        let cfg = config_for(Some((24, 80)), Some(1.0));
        assert_eq!((cfg.width(), cfg.height()), (48, 48));
    }

    #[test]
    fn a_wide_domain_is_capped_by_the_width_budget() {
        // A 4:1 domain would want 4× the height in width; the column cap holds
        // it to MAX_COLUMNS (60) → 120 dots, with height following the aspect.
        let cfg = config_for(Some((24, 200)), Some(4.0));
        assert_eq!(cfg.width(), 120);
        assert_eq!(cfg.height(), 30);
    }

    #[test]
    fn preview_floors_tiny_terminals() {
        let cfg = config_for(Some((1, 1)), Some(1.0));
        assert!(cfg.width() >= 32 && cfg.height() >= 3);
    }

    #[test]
    fn preview_defaults_without_a_known_size() {
        let cfg = config_for(None, Some(1.0));
        let default = ConsoleConfig::default();
        assert_eq!(
            (cfg.width(), cfg.height()),
            (default.width(), default.height())
        );
    }

    #[test]
    fn preview_defaults_without_an_aspect() {
        let cfg = config_for(Some((24, 80)), None);
        let default = ConsoleConfig::default();
        assert_eq!(
            (cfg.width(), cfg.height()),
            (default.width(), default.height())
        );
    }
}
