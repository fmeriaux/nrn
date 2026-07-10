use super::{Describe, Named, rows};
use nrn::classification::Classification;

impl Named for Classification {
    const NAME: &'static str = "CLASSIFICATION";
}

/// One dotted-leader row per class, ranked most likely first, each showing its
/// probability as a percentage, with the winning class marked by an arrow.
impl Describe for Classification {
    fn describe(&self) -> String {
        let entries: Vec<(String, String)> = self
            .ranking()
            .iter()
            .enumerate()
            .map(|(rank, (class, probability))| {
                let percentage = format!("{:.2}%", probability * 100.0);
                let value = if rank == 0 {
                    format!("{percentage}  \u{25c0}")
                } else {
                    percentage
                };
                (format!("Class {class}"), value)
            })
            .collect();

        let borrowed: Vec<(&str, String)> = entries
            .iter()
            .map(|(label, value)| (label.as_str(), value.clone()))
            .collect();

        rows(&borrowed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use nrn::classification::Classification;

    #[test]
    fn describe_marks_only_the_winning_class_with_an_arrow() {
        let text = Classification::from_probabilities(array![0.1, 0.7, 0.2].view()).describe();

        // Class 1 wins, so the arrow sits on its row and nowhere else.
        let winner = text.lines().find(|line| line.contains("Class 1")).unwrap();
        assert!(winner.contains('\u{25c0}'));
        let runner_up = text.lines().find(|line| line.contains("Class 2")).unwrap();
        assert!(!runner_up.contains('\u{25c0}'));
    }
}
