use super::{Describe, Named, theme};
use nrn::task::Task;

impl Named for Task {
    const NAME: &'static str = "TASK";
}

impl Describe for Task {
    fn describe(&self) -> String {
        let detail = match self {
            Task::Binary => "classification (binary)".to_string(),
            Task::MultiClass { class_count } => format!("classification ({class_count} classes)"),
            Task::MultiLabel { label_count } => format!("classification ({label_count} labels)"),
            Task::Regression { target_shape } => format!("regression (shape {target_shape:?})"),
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
            Task::MultiClass { class_count: 5 }
                .describe()
                .contains("classification (5 classes)")
        );
    }
}
