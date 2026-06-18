//! The project's standard progress bar, drawn to stdout.

use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use std::borrow::Cow;

/// Builds a hidden progress bar with the project's standard style, drawn to stdout.
pub(crate) fn styled_bar() -> ProgressBar {
    let bar = ProgressBar::hidden();
    bar.set_style(
        ProgressStyle::with_template(
            "{msg} {spinner:.green} [{elapsed_precise}] {wide_bar} {pos}/{len} {percent}% ({eta})",
        )
        .unwrap(),
    );
    bar.set_draw_target(ProgressDrawTarget::stdout());
    bar
}

/// A standalone progress bar of known length, for use with [`indicatif::ProgressIterator`].
pub(crate) fn bar(len: usize, msg: impl Into<Cow<'static, str>>) -> ProgressBar {
    let bar = styled_bar();
    bar.set_length(len as u64);
    bar.set_message(msg);
    bar
}
