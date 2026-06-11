use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use nrn::callbacks::{TrainingCallback, TrainingOutcome};
use nrn::evaluation::EvaluationSet;
use nrn::training::TrainingConfig;
use std::borrow::Cow;
use std::io::Result;

/// Builds a hidden progress bar with the project's standard style, drawn to stdout.
fn styled_bar() -> ProgressBar {
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
pub fn bar(len: usize, msg: impl Into<Cow<'static, str>>) -> ProgressBar {
    let bar = styled_bar();
    bar.set_length(len as u64);
    bar.set_message(msg);
    bar
}

pub struct Progression {
    msg: Cow<'static, str>,
    bar: ProgressBar,
}

impl Progression {
    pub fn new(msg: impl Into<Cow<'static, str>>) -> Progression {
        Progression {
            msg: msg.into(),
            bar: styled_bar(),
        }
    }
}

impl TrainingCallback for Progression {
    fn on_train_start(&mut self, config: &TrainingConfig) -> Result<()> {
        self.bar.set_length(config.epochs as u64);
        self.bar.set_message(self.msg.clone());
        Ok(())
    }

    fn on_epoch_end(&mut self, _epoch: usize) -> Result<()> {
        self.bar.inc(1);
        Ok(())
    }

    fn on_train_end(
        &mut self,
        _outcome: TrainingOutcome,
        _eval: Option<&EvaluationSet>,
        _epoch: usize,
    ) -> Result<()> {
        self.bar.finish_and_clear();
        Ok(())
    }
}
