use super::{Describe, Named, rows};
use ndarray::Ix2;
use nrn::model::{Inference, Labels};

impl Named for Inference {
    const NAME: &'static str = "PREDICTION";
}

impl Describe for Inference {
    fn describe(&self) -> String {
        match self {
            Inference::Classification {
                ranking, labels, ..
            } => describe_classification(ranking, labels.as_ref()),
            Inference::Values(activations) => {
                let output = activations
                    .output()
                    .into_dimensionality::<Ix2>()
                    .expect("infer_instance yields a flat output vector, not a spatial one");

                let entries: Vec<(String, String)> = output
                    .column(0)
                    .iter()
                    .enumerate()
                    .map(|(index, value)| (format!("Output {index}"), format!("{value:.4}")))
                    .collect();

                let borrowed: Vec<(&str, String)> = entries
                    .iter()
                    .map(|(label, value)| (label.as_str(), value.clone()))
                    .collect();

                rows(&borrowed)
            }
        }
    }
}

/// Ranked `(class, probability)` rows, named by `labels` when known, with the winning class
/// (rank 0) marked by an arrow.
fn describe_classification(ranking: &[(usize, f32)], labels: Option<&Labels>) -> String {
    let entries: Vec<(String, String)> = ranking
        .iter()
        .enumerate()
        .map(|(rank, (class, probability))| {
            let percentage = format!("{:.2}%", probability * 100.0);
            let value = if rank == 0 {
                format!("{percentage}  \u{25c0}")
            } else {
                percentage
            };
            let name = labels
                .and_then(|labels| labels.names().get(*class))
                .cloned()
                .unwrap_or_else(|| format!("Class {class}"));
            (name, value)
        })
        .collect();

    let borrowed: Vec<(&str, String)> = entries
        .iter()
        .map(|(label, value)| (label.as_str(), value.clone()))
        .collect();

    rows(&borrowed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn describe_marks_only_the_winning_class_with_an_arrow() {
        let text = describe_classification(&[(1, 0.7), (2, 0.2), (0, 0.1)], None);

        let winner = text.lines().find(|line| line.contains("Class 1")).unwrap();
        assert!(winner.contains('\u{25c0}'));
        let runner_up = text.lines().find(|line| line.contains("Class 2")).unwrap();
        assert!(!runner_up.contains('\u{25c0}'));
    }

    #[test]
    fn describe_names_classes_from_labels_when_known() {
        let labels = Labels::new(vec![
            "cat".to_string(),
            "dog".to_string(),
            "bird".to_string(),
        ]);
        let text = describe_classification(&[(1, 0.7), (0, 0.2), (2, 0.1)], Some(&labels));

        assert!(text.contains("dog"));
        assert!(text.contains("cat"));
        assert!(text.contains("bird"));
        assert!(!text.contains("Class"));
    }

    #[test]
    fn describe_falls_back_to_class_n_without_labels() {
        let text = describe_classification(&[(0, 0.9), (1, 0.1)], None);
        assert!(text.contains("Class 0"));
        assert!(text.contains("Class 1"));
    }
}
