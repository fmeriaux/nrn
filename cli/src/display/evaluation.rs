//! The split-sample line shown at train start, and the per-split EVALUATION
//! table — loss and accuracy for Train / Val / Test — shown at train end.

use super::{Describe, Named, column_width, theme};
use nrn::data::ModelSplit;
use nrn::evaluation::{Evaluation, EvaluationSet};

const LOSS_HEADER: &str = "loss";
const ACC_HEADER: &str = "accuracy";

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

impl Named for EvaluationSet {
    const NAME: &'static str = "EVALUATION";
}

/// One row per split — its label, loss, and accuracy — under a `loss`/`accuracy`
/// header, the columns aligned. An absent validation split reads `n/a`.
impl Describe for EvaluationSet {
    fn describe(&self) -> String {
        let cells: Vec<(&str, String, String)> = [
            ("Train", Some(&self.train)),
            ("Val", self.validation.as_ref()),
            ("Test", Some(&self.test)),
        ]
        .into_iter()
        .map(|(name, eval)| match eval {
            Some(Evaluation { loss, accuracy }) => {
                (name, format!("{loss:.4}"), format!("{accuracy:.1}%"))
            }
            None => (name, "n/a".to_string(), "n/a".to_string()),
        })
        .collect();

        let name_w = column_width(cells.iter().map(|(name, ..)| *name));
        let loss_w = column_width(
            cells
                .iter()
                .map(|(_, loss, _)| loss.as_str())
                .chain([LOSS_HEADER]),
        );
        let acc_w = column_width(
            cells
                .iter()
                .map(|(.., acc)| acc.as_str())
                .chain([ACC_HEADER]),
        );

        let header = format!(
            "   {}   {}   {}",
            " ".repeat(name_w),
            theme::caption(format!("{LOSS_HEADER:>loss_w$}")),
            theme::caption(format!("{ACC_HEADER:>acc_w$}")),
        );

        let body = cells.iter().map(|(name, loss, acc)| {
            format!(
                "   {}   {}   {}",
                theme::label(format!("{name:<name_w$}")),
                theme::value(format!("{loss:>loss_w$}")),
                theme::value(format!("{acc:>acc_w$}")),
            )
        });

        std::iter::once(header)
            .chain(body)
            .collect::<Vec<_>>()
            .join("\n")
    }
}
