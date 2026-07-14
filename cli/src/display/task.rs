use super::{Describe, Named, theme};
use nrn::task::Task;

impl Named for Task {
    const NAME: &'static str = "TASK";
}

impl Describe for Task {
    fn describe(&self) -> String {
        let detail = match self {
            Task::Binary => "classification (binary)".to_string(),
            Task::MultiClass { n_classes } => format!("classification ({n_classes} classes)"),
            Task::MultiLabel { n_labels } => format!("classification ({n_labels} labels)"),
            Task::Regression { shape } => format!("regression (shape {shape:?})"),
        };
        theme::value(detail)
    }
}

#[cfg(test)]
mod tests {
    use crate::display::Describe;
    use nrn::task::Task;

    #[test]
    fn describes_binary_and_multi_class() {
        assert!(Task::Binary.describe().contains("classification (binary)"));
        assert!(
            Task::MultiClass { n_classes: 5 }
                .describe()
                .contains("classification (5 classes)")
        );
    }
}
