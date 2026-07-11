use super::{Describe, Named, theme};
use nrn::objectives::Objective;

impl Named for Objective {
    const NAME: &'static str = "OBJECTIVE";
}

impl Describe for Objective {
    fn describe(&self) -> String {
        let detail = match self {
            Objective::Binary => "classification (binary)".to_string(),
            Objective::MultiClass { n_classes } => format!("classification ({n_classes} classes)"),
            Objective::MultiLabel { n_labels } => format!("classification ({n_labels} labels)"),
            Objective::Regression { n_outputs } => format!("regression ({n_outputs} outputs)"),
        };
        theme::value(detail)
    }
}

#[cfg(test)]
mod tests {
    use crate::display::Describe;
    use nrn::objectives::Objective;

    #[test]
    fn describes_binary_and_multi_class() {
        assert!(
            Objective::Binary
                .describe()
                .contains("classification (binary)")
        );
        assert!(
            Objective::MultiClass { n_classes: 5 }
                .describe()
                .contains("classification (5 classes)")
        );
    }
}
