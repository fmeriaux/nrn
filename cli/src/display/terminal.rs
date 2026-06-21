//! Rendering a [`Figure`] to the current terminal.

use console::Term;
use nrn::plot::{ConsoleConfig, Figure};
use std::error::Error;

/// Prints `figure` to stdout, rendered to fit the current terminal.
pub(crate) fn preview(figure: &Figure) -> Result<(), Box<dyn Error>> {
    println!("{}", figure.to_console(&terminal_console_config())?);
    Ok(())
}

/// A console canvas sized to the current terminal, falling back to a default
/// when the size can't be determined (e.g. piped output).
fn terminal_console_config() -> ConsoleConfig {
    console_config_for(Term::stdout().size_checked())
}

/// Maps a terminal `(rows, columns)` size to a console canvas, reserving room
/// for axis labels and surrounding output; defaults when the size is unknown.
fn console_config_for(size: Option<(u16, u16)>) -> ConsoleConfig {
    match size {
        // Braille cells span two dots across and four down.
        Some((rows, columns)) => ConsoleConfig::new(
            (columns.saturating_sub(6) as u32).max(16) * 2,
            (rows.saturating_sub(10) as u32).max(8) * 4,
        ),
        None => ConsoleConfig::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn console_config_tracks_the_terminal_size() {
        // 80 columns → (80-6)*2 dots wide; 24 rows → (24-10)*4 dots tall.
        let cfg = console_config_for(Some((24, 80)));
        assert_eq!((cfg.width, cfg.height), (148, 56));
    }

    #[test]
    fn console_config_floors_tiny_terminals() {
        let cfg = console_config_for(Some((1, 1)));
        assert_eq!((cfg.width, cfg.height), (32, 32));
    }

    #[test]
    fn console_config_defaults_without_a_known_size() {
        let cfg = console_config_for(None);
        let default = ConsoleConfig::default();
        assert_eq!((cfg.width, cfg.height), (default.width, default.height));
    }
}
