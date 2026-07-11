//! The [`Objective`]: the learning task a run optimizes for, reified as a type.

use crate::data::Dataset;
use ndarray::{ArrayD, Axis};
use std::collections::HashSet;
use std::error::Error;

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
    /// Infers the objective from the shape and dtype of a dataset's targets:
    /// - rank-1 non-negative integers over two distinct values →
    ///   [`Binary`](Objective::Binary), over three or more → [`MultiClass`](Objective::MultiClass);
    /// - rank-1 real values → [`Regression`](Objective::Regression) with one output;
    /// - rank-2 `(samples, k)` in `{0, 1}` → [`MultiLabel`](Objective::MultiLabel);
    /// - rank-2 `(samples, k)` real values → [`Regression`](Objective::Regression) with
    ///   `k` outputs.
    pub fn from_dataset(dataset: &Dataset) -> Self {
        let targets = dataset.targets();
        if targets.ndim() == 1 {
            let class_ids = targets.iter().all(|&v| v >= 0.0 && v.fract() == 0.0);
            match dataset.n_classes() {
                2 if class_ids => Objective::Binary,
                n if class_ids && n > 2 => Objective::MultiClass { n_classes: n },
                _ => Objective::Regression { n_outputs: 1 },
            }
        } else {
            let width = targets.len_of(Axis(1));
            if targets.iter().all(|&v| v == 0.0 || v == 1.0) {
                Objective::MultiLabel { n_labels: width }
            } else {
                Objective::Regression { n_outputs: width }
            }
        }
    }

    /// Checks that a dataset's targets can support this objective.
    ///
    /// # Errors
    /// An [`ObjectiveDataError`] describing the first mismatch between the targets'
    /// shape or values and what this objective requires.
    pub fn validate_dataset(&self, dataset: &Dataset) -> Result<(), ObjectiveDataError> {
        let targets = dataset.targets();
        match self {
            Objective::Binary => validate_classification(targets, 2),
            Objective::MultiClass { n_classes } => validate_classification(targets, *n_classes),
            Objective::MultiLabel { n_labels } => validate_multilabel(targets, *n_labels),
            Objective::Regression { n_outputs } => validate_regression(targets, *n_outputs),
        }
    }
}

/// Checks that `targets` are rank-1 contiguous 0-indexed class ids over exactly
/// `expected` classes.
fn validate_classification(
    targets: &ArrayD<f32>,
    expected: usize,
) -> Result<(), ObjectiveDataError> {
    if targets.ndim() != 1 {
        return Err(ObjectiveDataError::WrongTargetRank {
            expected: 1,
            found: targets.ndim(),
        });
    }
    if !targets.iter().all(|&v| v >= 0.0 && v.fract() == 0.0) {
        return Err(ObjectiveDataError::NonContiguousClassIds);
    }
    let n_classes = targets
        .iter()
        .map(|&v| v.to_bits())
        .collect::<HashSet<u32>>()
        .len();
    if n_classes < 2 {
        return Err(ObjectiveDataError::TooFewClasses(n_classes));
    }
    let max_id = targets.iter().fold(0.0_f32, |m, &v| m.max(v)) as usize;
    if max_id + 1 != n_classes {
        return Err(ObjectiveDataError::NonContiguousClassIds);
    }
    if n_classes != expected {
        return Err(ObjectiveDataError::ClassCountMismatch {
            expected,
            found: n_classes,
        });
    }
    Ok(())
}

/// Checks that `targets` are a rank-2 `{0, 1}` indicator of width `n_labels`.
fn validate_multilabel(targets: &ArrayD<f32>, n_labels: usize) -> Result<(), ObjectiveDataError> {
    if targets.ndim() != 2 {
        return Err(ObjectiveDataError::WrongTargetRank {
            expected: 2,
            found: targets.ndim(),
        });
    }
    if !targets.iter().all(|&v| v == 0.0 || v == 1.0) {
        return Err(ObjectiveDataError::LabelsNotBinary);
    }
    let width = targets.len_of(Axis(1));
    if width != n_labels {
        return Err(ObjectiveDataError::WidthMismatch {
            expected: n_labels,
            found: width,
        });
    }
    Ok(())
}

/// Checks that `targets` are finite and `n_outputs` wide.
fn validate_regression(targets: &ArrayD<f32>, n_outputs: usize) -> Result<(), ObjectiveDataError> {
    if !targets.iter().all(|&v| v.is_finite()) {
        return Err(ObjectiveDataError::NonFiniteTargets);
    }
    let width = if targets.ndim() == 1 {
        1
    } else {
        targets.len_of(Axis(1))
    };
    if width != n_outputs {
        return Err(ObjectiveDataError::WidthMismatch {
            expected: n_outputs,
            found: width,
        });
    }
    Ok(())
}

/// Errors returned when a dataset's targets cannot support an [`Objective`].
#[derive(Debug, PartialEq, Eq)]
pub enum ObjectiveDataError {
    /// The targets have the wrong rank for the objective.
    WrongTargetRank { expected: usize, found: usize },
    /// The class ids are not contiguous 0-indexed integers.
    NonContiguousClassIds,
    /// Fewer than two distinct classes are present (carries the count found).
    TooFewClasses(usize),
    /// The number of classes differs from the objective's declared count.
    ClassCountMismatch { expected: usize, found: usize },
    /// The multi-label targets are not all `0` or `1`.
    LabelsNotBinary,
    /// The regression targets contain a non-finite value.
    NonFiniteTargets,
    /// The target width differs from the objective's declared width.
    WidthMismatch { expected: usize, found: usize },
}

impl std::fmt::Display for ObjectiveDataError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ObjectiveDataError::WrongTargetRank { expected, found } => {
                write!(f, "targets must be rank-{expected}, but found rank-{found}")
            }
            ObjectiveDataError::NonContiguousClassIds => {
                write!(f, "class ids must be contiguous 0-indexed integers")
            }
            ObjectiveDataError::TooFewClasses(n) => {
                write!(f, "a classifier needs at least 2 classes, but found {n}")
            }
            ObjectiveDataError::ClassCountMismatch { expected, found } => write!(
                f,
                "objective declares {expected} classes, but the targets carry {found}"
            ),
            ObjectiveDataError::LabelsNotBinary => {
                write!(f, "multi-label targets must all be 0 or 1")
            }
            ObjectiveDataError::NonFiniteTargets => {
                write!(f, "regression targets must all be finite")
            }
            ObjectiveDataError::WidthMismatch { expected, found } => write!(
                f,
                "objective declares {expected} outputs, but the targets carry {found}"
            ),
        }
    }
}

impl Error for ObjectiveDataError {}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2, array};

    /// A two-feature dataset carrying `labels` as its rank-1 targets.
    fn tabular_dataset(labels: Array1<f32>) -> Dataset {
        let inputs = Array2::from_shape_fn((labels.len(), 2), |(i, _)| i as f32);
        Dataset::tabular(inputs, labels, None).unwrap()
    }

    /// A two-feature dataset carrying a rank-2 `targets` indicator or output block.
    fn rank2_target_dataset(targets: Array2<f32>) -> Dataset {
        let inputs = Array2::from_shape_fn((targets.nrows(), 2), |(i, _)| i as f32).into_dyn();
        Dataset::new(inputs, targets.into_dyn(), None).unwrap()
    }

    /// A dataset with `n_classes` contiguous labels spread across ten samples.
    fn dataset_with_classes(n_classes: usize) -> Dataset {
        tabular_dataset(Array1::from_shape_fn(10, |i| (i % n_classes) as f32))
    }

    #[test]
    fn from_dataset_infers_binary_and_multi_class() {
        assert_eq!(
            Objective::from_dataset(&dataset_with_classes(2)),
            Objective::Binary
        );
        assert_eq!(
            Objective::from_dataset(&dataset_with_classes(4)),
            Objective::MultiClass { n_classes: 4 }
        );
    }

    #[test]
    fn from_dataset_infers_regression_from_real_rank1_targets() {
        let dataset = tabular_dataset(array![0.5f32, 1.5, 2.25, 0.1]);
        assert_eq!(
            Objective::from_dataset(&dataset),
            Objective::Regression { n_outputs: 1 }
        );
    }

    #[test]
    fn from_dataset_infers_regression_from_a_single_class() {
        // One distinct integer label is not classifiable; inference falls back to regression.
        let dataset = tabular_dataset(array![1.0f32, 1.0, 1.0]);
        assert_eq!(
            Objective::from_dataset(&dataset),
            Objective::Regression { n_outputs: 1 }
        );
    }

    #[test]
    fn from_dataset_infers_multi_label_from_a_binary_indicator() {
        let targets = array![[1.0f32, 0.0, 1.0], [0.0, 1.0, 1.0]];
        assert_eq!(
            Objective::from_dataset(&rank2_target_dataset(targets)),
            Objective::MultiLabel { n_labels: 3 }
        );
    }

    #[test]
    fn from_dataset_infers_regression_from_real_rank2_targets() {
        let targets = array![[0.5f32, 1.0], [2.0, 3.5]];
        assert_eq!(
            Objective::from_dataset(&rank2_target_dataset(targets)),
            Objective::Regression { n_outputs: 2 }
        );
    }

    #[test]
    fn validate_dataset_accepts_matching_shapes() {
        assert!(
            Objective::Binary
                .validate_dataset(&dataset_with_classes(2))
                .is_ok()
        );
        assert!(
            Objective::MultiClass { n_classes: 4 }
                .validate_dataset(&dataset_with_classes(4))
                .is_ok()
        );
        let multilabel = rank2_target_dataset(array![[1.0f32, 0.0], [0.0, 1.0]]);
        assert!(
            Objective::MultiLabel { n_labels: 2 }
                .validate_dataset(&multilabel)
                .is_ok()
        );
        let regression = rank2_target_dataset(array![[0.5f32, 1.0], [2.0, 3.5]]);
        assert!(
            Objective::Regression { n_outputs: 2 }
                .validate_dataset(&regression)
                .is_ok()
        );
    }

    #[test]
    fn validate_rejects_non_contiguous_class_ids() {
        // labels = {0, 2}: two classes, but not the contiguous {0, 1} set.
        let dataset = tabular_dataset(array![0.0f32, 2.0]);
        assert_eq!(
            Objective::Binary.validate_dataset(&dataset),
            Err(ObjectiveDataError::NonContiguousClassIds)
        );
    }

    #[test]
    fn validate_rejects_out_of_range_multiclass_label() {
        // labels = {0, 1, 2, 5}: four classes, but id 5 breaks contiguity.
        let dataset = tabular_dataset(array![0.0f32, 1.0, 2.0, 5.0]);
        assert_eq!(
            Objective::MultiClass { n_classes: 4 }.validate_dataset(&dataset),
            Err(ObjectiveDataError::NonContiguousClassIds)
        );
    }

    #[test]
    fn validate_rejects_a_single_class() {
        let dataset = tabular_dataset(array![1.0f32, 1.0, 1.0]);
        assert_eq!(
            Objective::Binary.validate_dataset(&dataset),
            Err(ObjectiveDataError::TooFewClasses(1))
        );
    }

    #[test]
    fn validate_rejects_a_class_count_mismatch() {
        // Three contiguous classes cannot satisfy a Binary objective.
        assert_eq!(
            Objective::Binary.validate_dataset(&dataset_with_classes(3)),
            Err(ObjectiveDataError::ClassCountMismatch {
                expected: 2,
                found: 3
            })
        );
    }

    #[test]
    fn validate_rejects_a_wrong_target_rank() {
        // A rank-2 target cannot satisfy a single-label classification objective.
        let dataset = rank2_target_dataset(array![[1.0f32, 0.0], [0.0, 1.0]]);
        assert_eq!(
            Objective::Binary.validate_dataset(&dataset),
            Err(ObjectiveDataError::WrongTargetRank {
                expected: 1,
                found: 2
            })
        );
    }

    #[test]
    fn validate_rejects_a_non_binary_multi_label() {
        let dataset = rank2_target_dataset(array![[0.5f32, 0.0], [1.0, 1.0]]);
        assert_eq!(
            Objective::MultiLabel { n_labels: 2 }.validate_dataset(&dataset),
            Err(ObjectiveDataError::LabelsNotBinary)
        );
    }

    #[test]
    fn validate_rejects_a_multi_label_width_mismatch() {
        let dataset = rank2_target_dataset(array![[1.0f32, 0.0], [0.0, 1.0]]);
        assert_eq!(
            Objective::MultiLabel { n_labels: 3 }.validate_dataset(&dataset),
            Err(ObjectiveDataError::WidthMismatch {
                expected: 3,
                found: 2
            })
        );
    }

    #[test]
    fn validate_rejects_non_finite_regression_targets() {
        let dataset = rank2_target_dataset(array![[0.5f32, f32::NAN], [1.0, 2.0]]);
        assert_eq!(
            Objective::Regression { n_outputs: 2 }.validate_dataset(&dataset),
            Err(ObjectiveDataError::NonFiniteTargets)
        );
    }

    #[test]
    fn validate_rejects_a_regression_width_mismatch() {
        let dataset = rank2_target_dataset(array![[0.5f32, 1.0], [2.0, 3.5]]);
        assert_eq!(
            Objective::Regression { n_outputs: 1 }.validate_dataset(&dataset),
            Err(ObjectiveDataError::WidthMismatch {
                expected: 1,
                found: 2
            })
        );
    }

    #[test]
    fn objective_data_error_messages_are_descriptive() {
        assert_eq!(
            ObjectiveDataError::WrongTargetRank {
                expected: 1,
                found: 2
            }
            .to_string(),
            "targets must be rank-1, but found rank-2"
        );
        assert_eq!(
            ObjectiveDataError::ClassCountMismatch {
                expected: 2,
                found: 3
            }
            .to_string(),
            "objective declares 2 classes, but the targets carry 3"
        );
        assert_eq!(
            ObjectiveDataError::WidthMismatch {
                expected: 1,
                found: 2
            }
            .to_string(),
            "objective declares 1 outputs, but the targets carry 2"
        );
    }
}
