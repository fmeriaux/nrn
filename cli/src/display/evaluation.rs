//! Inline evaluation/split fragments. Unlike the entity blocks, these are
//! single-line strings appended to a message verb (e.g. `Training completed ·
//! Train(..) · …`), so they stay one line rather than becoming a block.

use super::theme;
use nrn::data::ModelSplit;
use nrn::evaluation::{Evaluation, EvaluationSet};

/// The realized per-split sample counts, complementing the recap's `Split` ratios.
pub(crate) fn split_summary(split: &ModelSplit) -> String {
    format!(
        "Samples · Train={} · Val={} · Test={}",
        theme::value(split.train_size()),
        theme::value(split.validation_size()),
        theme::value(split.test_size()),
    )
}

/// Formats an [`EvaluationSet`] as `Train(..) · Val(..) · Test(..)`, each split's
/// loss/accuracy via [`evaluation_summary`] and `N/A` for an absent validation split.
pub(crate) fn eval_set_summary(eval: &EvaluationSet) -> String {
    format!(
        "Train({}) · Val({}) · Test({})",
        evaluation_summary(&eval.train),
        eval.validation
            .as_ref()
            .map_or_else(|| "N/A".to_string(), evaluation_summary),
        evaluation_summary(&eval.test),
    )
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
