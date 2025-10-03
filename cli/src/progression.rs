use indicatif::{ProgressBar, ProgressBarIter, ProgressDrawTarget, ProgressStyle};
use std::borrow::Cow;
use std::ops::Range;

pub struct Progression {
    length: usize,
    bar: ProgressBar,
}

impl Progression {
    pub fn new(len: usize, msg: impl Into<Cow<'static, str>>) -> Progression {
        let bar = ProgressBar::hidden();
        bar.set_length(len as u64);
        bar.set_style(
            ProgressStyle::with_template(
                "{msg} {spinner:.green} [{elapsed_precise}] {wide_bar} {pos}/{len} {percent}% ({eta})",
            )
            .unwrap(),
        );

        bar.set_message(msg);
        bar.set_draw_target(ProgressDrawTarget::stdout());

        Progression { length: len, bar }
    }

    pub fn inc(&self) {
        self.bar.inc(1);
    }

    pub fn done(&self) {
        self.bar.finish_and_clear();
    }

    pub fn iter(&self) -> ProgressBarIter<Range<usize>> {
        self.bar.wrap_iter(0..self.length)
    }
}
