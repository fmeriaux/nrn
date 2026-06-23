//! The CLI's progress bars: pretty, task-specific, and the only place the
//! `indicatif` backend is touched. Each wrapper owns a [`ProgressBar`], styles
//! itself through [`theme`](super::theme), and exposes domain methods so call
//! sites never see — or restyle — the bar.

use super::theme;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use nrn::evaluation::EvaluationSet;
use std::fmt::Display;
use std::time::Duration;

/// The fill / head / empty glyphs of every bar — a thin, single-weight rule.
const PROGRESS_CHARS: &str = "━╸─";

/// A hidden→stdout bar in the project style: a green spinner, a bold blue
/// prefix label, the accent fill bar, then the caller's `suffix` of contextual
/// tokens.
fn styled(suffix: &str) -> ProgressBar {
    let template = format!(
        "{{spinner:.{accent}}} {{prefix:.bold.{title}}} {{wide_bar:.{accent}}} {suffix}",
        accent = theme::ACCENT,
        title = theme::TITLE,
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
                "{{spinner:.{accent}}} {{msg:.bold.{title}}}",
                accent = theme::ACCENT,
                title = theme::TITLE,
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
/// loss and accuracy beside the bar.
pub(crate) struct Epochs(ProgressBar);

impl Epochs {
    /// A bar awaiting [`start`](Self::start) — held by the monitor before the
    /// loop's total epoch count is known.
    pub(crate) fn new() -> Self {
        Self(styled("{pos}/{len} epochs · {msg}{eta:.dim}"))
    }

    /// Reveals the bar, spanning `epochs` total epochs.
    pub(crate) fn start(&self, epochs: usize) {
        self.0.set_prefix("Training");
        self.0.set_length(epochs as u64);
    }

    /// Records one finished epoch.
    pub(crate) fn advance(&self) {
        self.0.inc(1);
    }

    /// Surfaces the latest evaluation — validation when present, else train.
    pub(crate) fn evaluated(&self, eval: &EvaluationSet) {
        let metrics = eval.validation.unwrap_or(eval.train);
        self.0.set_message(format!(
            "loss {:.4} · acc {:.1}% · ",
            metrics.loss, metrics.accuracy
        ));
    }

    /// Runs `body` with the bar suspended, so printed lines aren't garbled by
    /// its redraws.
    pub(crate) fn quiet<R>(&self, body: impl FnOnce() -> R) -> R {
        self.0.suspend(body)
    }

    /// Clears the bar at the end of the run.
    pub(crate) fn finish(&self) {
        self.0.finish_and_clear();
    }
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
