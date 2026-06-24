//! The CLI's progress bars: pretty, task-specific, and the only place the
//! `indicatif` backend is touched. Each wrapper owns a [`ProgressBar`], styles
//! itself through [`theme`](super::theme), and exposes domain methods so call
//! sites never see — or restyle — the bar.

use super::theme;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use nrn::evaluation::{Evaluation, EvaluationSet};
use std::fmt::Display;
use std::time::Duration;

/// The fill / head / empty glyphs of every bar — a thin, single-weight rule.
const PROGRESS_CHARS: &str = "━╸─";

/// A hidden→stdout bar in the project style: a spinner, a bold action prefix,
/// the fill bar, then the caller's `suffix` of contextual tokens.
fn styled(suffix: &str) -> ProgressBar {
    let template = format!(
        "{{spinner:.{accent}}} {{prefix:.bold.{active}}} {{wide_bar:.{accent}}} {suffix}",
        accent = theme::ACCENT,
        active = theme::ACTIVE,
    );
    let bar = ProgressBar::hidden();
    bar.set_style(
        ProgressStyle::with_template(&template)
            .unwrap()
            .progress_chars(PROGRESS_CHARS),
    );
    bar.set_draw_target(ProgressDrawTarget::stdout());
    bar
}

/// An indeterminate spinner for a single blocking step with no measurable
/// progress (e.g. splitting a large dataset). It ticks on a background thread,
/// so it keeps animating while the calling thread is busy.
pub(crate) struct Spinner(ProgressBar);

impl Spinner {
    /// Starts a spinner labelled `message`, animating until [`finish`](Self::finish).
    pub(crate) fn start(message: &'static str) -> Self {
        let bar = ProgressBar::new_spinner();
        bar.set_style(
            ProgressStyle::with_template(&format!(
                "{{spinner:.{accent}}} {{msg:.bold.{active}}}",
                accent = theme::ACCENT,
                active = theme::ACTIVE,
            ))
            .unwrap(),
        );
        bar.set_draw_target(ProgressDrawTarget::stdout());
        bar.set_message(message);
        bar.enable_steady_tick(Duration::from_millis(80));
        Self(bar)
    }

    /// Stops and clears the spinner.
    pub(crate) fn finish(&self) {
        self.0.finish_and_clear();
    }
}

/// Tracks image encoding across every category under a single bar: the prefix
/// names the category being read, the bar fills over the run's total images.
pub(crate) struct Encoding(ProgressBar);

impl Encoding {
    /// A bar spanning `total` images across the whole input directory.
    pub(crate) fn new(total: usize) -> Self {
        let bar = styled("{pos}/{len} images · {eta:.dim}");
        bar.set_length(total as u64);
        Self(bar)
    }

    /// Announces the category now being read — the `index`-th of `count`.
    pub(crate) fn category(&self, index: usize, count: usize, name: impl Display) {
        self.0.set_prefix(format!("[{}/{count}] {name}", index + 1));
    }

    /// Records one processed image.
    pub(crate) fn advance(&self) {
        self.0.inc(1);
    }

    /// Clears the bar once every category is encoded.
    pub(crate) fn finish(&self) {
        self.0.finish_and_clear();
    }
}

/// Tracks the training loop epoch by epoch, surfacing the latest evaluated
/// loss and accuracy — each with a trend chevron against the previous
/// checkpoint — beside the bar.
pub(crate) struct Epochs {
    bar: ProgressBar,
    /// The previous checkpoint's evaluation, for the trend chevrons.
    previous: Option<Evaluation>,
}

impl Epochs {
    /// A bar awaiting [`start`](Self::start) — held by the monitor before the
    /// loop's total epoch count is known.
    pub(crate) fn new() -> Self {
        Self {
            bar: styled("{pos}/{len} epochs · {msg}{eta:.dim}"),
            previous: None,
        }
    }

    /// Reveals the bar, spanning `epochs` total epochs.
    pub(crate) fn start(&self, epochs: usize) {
        self.bar.set_prefix("Training");
        self.bar.set_length(epochs as u64);
    }

    /// Records one finished epoch.
    pub(crate) fn advance(&self) {
        self.bar.inc(1);
    }

    /// Surfaces the latest evaluation — validation when present, else train —
    /// each metric trailed by a [`trend`] chevron against the previous checkpoint.
    pub(crate) fn evaluated(&mut self, eval: &EvaluationSet) {
        let metrics = eval.validation.unwrap_or(eval.train);

        let (loss_trend, acc_trend) = match self.previous {
            Some(previous) => (
                trend(previous.loss, metrics.loss, false),
                trend(previous.accuracy, metrics.accuracy, true),
            ),
            None => (String::new(), String::new()),
        };

        self.bar.set_message(format!(
            "loss {:.4}{loss_trend} · acc {:.1}%{acc_trend} · ",
            metrics.loss, metrics.accuracy
        ));

        self.previous = Some(metrics);
    }

    /// Runs `body` with the bar suspended, so printed lines aren't garbled by
    /// its redraws.
    pub(crate) fn quiet<R>(&self, body: impl FnOnce() -> R) -> R {
        self.bar.suspend(body)
    }

    /// Clears the bar at the end of the run.
    pub(crate) fn finish(&self) {
        self.bar.finish_and_clear();
    }
}

/// A trend chevron for a metric moving from `previous` to `current`: the arrow
/// points the way it moved, styled [`improving`](theme::improving) when that's a
/// gain (loss falling, accuracy rising) and [`regressing`](theme::regressing)
/// otherwise, with a leading space. Empty when unchanged or not comparable
/// (e.g. a diverged `NaN`).
fn trend(previous: f32, current: f32, higher_is_better: bool) -> String {
    let delta = current - previous;
    if !delta.is_normal() {
        return String::new();
    }
    let rising = delta > 0.0;
    let chevron = if rising { '▲' } else { '▼' };
    let styled = if rising == higher_is_better {
        theme::improving(chevron)
    } else {
        theme::regressing(chevron)
    };
    format!(" {styled}")
}

/// Tracks decision-boundary frame rendering for an animation.
pub(crate) struct Frames(ProgressBar);

impl Frames {
    /// A bar over `count` frames, prefixed with what's being rendered
    /// (e.g. `"Rendering GIF"`).
    pub(crate) fn new(count: usize, what: &'static str) -> Self {
        let bar = styled("{pos}/{len} frames · {eta:.dim}");
        bar.set_length(count as u64);
        bar.set_prefix(what);
        Self(bar)
    }

    /// Records one rendered frame.
    pub(crate) fn advance(&self) {
        self.0.inc(1);
    }

    /// Clears the bar once every frame is rendered.
    pub(crate) fn finish(&self) {
        self.0.finish_and_clear();
    }
}

#[cfg(test)]
mod tests {
    use super::{theme, trend};

    // Comparing against the theme helpers keeps the assertions agnostic to
    // whether colors are enabled (both sides share that state).
    #[test]
    fn falling_loss_improves_with_a_down_chevron() {
        assert_eq!(
            trend(1.0, 0.5, false),
            format!(" {}", theme::improving('▼'))
        );
    }

    #[test]
    fn rising_loss_regresses_with_an_up_chevron() {
        assert_eq!(
            trend(0.5, 1.0, false),
            format!(" {}", theme::regressing('▲'))
        );
    }

    #[test]
    fn rising_accuracy_improves_with_an_up_chevron() {
        assert_eq!(
            trend(90.0, 95.0, true),
            format!(" {}", theme::improving('▲'))
        );
    }

    #[test]
    fn falling_accuracy_regresses_with_a_down_chevron() {
        assert_eq!(
            trend(95.0, 90.0, true),
            format!(" {}", theme::regressing('▼'))
        );
    }

    #[test]
    fn unchanged_or_nan_metrics_show_no_chevron() {
        assert!(trend(1.0, 1.0, false).is_empty());
        assert!(trend(1.0, f32::NAN, false).is_empty());
        assert!(trend(f32::NAN, 1.0, true).is_empty());
    }
}
