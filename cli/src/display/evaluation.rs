//! Inline evaluation/split descriptions: single-line [`Describe`] strings
//! appended to a message verb (e.g. `Training completed · Train(..) · …`).

use super::{Describe, theme};
use nrn::data::ModelSplit;
use nrn::evaluation::{Evaluation, EvaluationSet};

/// The realized per-split sample counts.
impl Describe for ModelSplit {
    fn describe(&self) -> String {
        format!(
            "Samples · Train={} · Val={} · Test={}",
            theme::value(self.train_size()),
            theme::value(self.validation_size()),
            theme::value(self.test_size()),
        )
    }
}

/// `Train(..) · Val(..) · Test(..)`, each split's loss/accuracy via
/// [`evaluation_summary`] and `N/A` for an absent validation split.
impl Describe for EvaluationSet {
    fn describe(&self) -> String {
        format!(
            "Train({}) · Val({}) · Test({})",
            evaluation_summary(&self.train),
            self.validation
                .as_ref()
                .map_or_else(|| "N/A".to_string(), evaluation_summary),
            evaluation_summary(&self.test),
        )
    }
}

/// Formats a single [`Evaluation`] as `L=<loss> · A=<accuracy>%`.
fn evaluation_summary(eval: &Evaluation) -> String {
    format!(
        "L={} · A={}{}",
        theme::value(format!("{:.4}", eval.loss)),
        theme::value(format!("{:.1}", eval.accuracy)),
        theme::value("%"),
    )
}
