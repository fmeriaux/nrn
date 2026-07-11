//! The [`Objective`]: the learning task a run optimizes for, reified as a type.

use crate::data::Dataset;

/// The learning task a training run optimizes for. Each variant names the task and carries the
/// output width it implies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Objective {
    /// Two mutually exclusive classes, one output.
    Binary,
    /// `n_classes` mutually exclusive classes, `n_classes` outputs.
    MultiClass { n_classes: usize },
    /// `n_labels` independent yes/no labels, `n_labels` outputs.
    MultiLabel { n_labels: usize },
    /// `n_outputs` real-valued targets, `n_outputs` outputs.
    Regression { n_outputs: usize },
}

impl Objective {
    /// Infers the objective from a dataset's targets, as single-label classification on its
    /// class count.
    pub fn from_dataset(dataset: &Dataset) -> Self {
        Self::classification(dataset.n_classes())
    }

    /// Builds a single-label classification objective from a class count: `2` is
    /// [`Binary`](Objective::Binary), `3` or more is [`MultiClass`](Objective::MultiClass).
    /// # Panics
    /// When `n_classes` is less than 2.
    pub fn classification(n_classes: usize) -> Self {
        assert!(n_classes >= 2, "Number of classes must be at least 2.");
        if n_classes == 2 {
            Objective::Binary
        } else {
            Objective::MultiClass { n_classes }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    /// A two-feature dataset with `n_classes` contiguous labels spread across ten samples.
    fn dataset_with_classes(n_classes: usize) -> Dataset {
        let features = Array2::from_shape_fn((10, 2), |(i, _)| i as f32);
        let labels = Array1::from_shape_fn(10, |i| (i % n_classes) as f32);
        Dataset::new(features, labels, None).unwrap()
    }

    #[test]
    fn classification_selects_binary_for_two_classes() {
        assert_eq!(Objective::classification(2), Objective::Binary);
    }

    #[test]
    fn classification_selects_multi_class_for_three_or_more() {
        assert_eq!(
            Objective::classification(5),
            Objective::MultiClass { n_classes: 5 }
        );
    }

    #[test]
    #[should_panic(expected = "Number of classes must be at least 2.")]
    fn classification_rejects_fewer_than_two_classes() {
        Objective::classification(1);
    }

    #[test]
    fn from_dataset_reads_the_class_count() {
        assert_eq!(
            Objective::from_dataset(&dataset_with_classes(2)),
            Objective::Binary
        );
        assert_eq!(
            Objective::from_dataset(&dataset_with_classes(4)),
            Objective::MultiClass { n_classes: 4 }
        );
    }
}
