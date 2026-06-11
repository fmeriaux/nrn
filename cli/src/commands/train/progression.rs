use crate::console::styled_bar;
use indicatif::ProgressBar;
use nrn::evaluation::EvaluationSet;
use nrn::model::NeuralNetwork;
use nrn::training::{TrainingCallback, TrainingConfig, TrainingOutcome};
use std::borrow::Cow;
use std::io::Result;

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
        _model: Option<&NeuralNetwork>,
        _eval: Option<&EvaluationSet>,
        _epoch: usize,
    ) -> Result<()> {
        self.bar.finish_and_clear();
        Ok(())
    }
}
