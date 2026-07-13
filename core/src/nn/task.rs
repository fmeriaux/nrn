//! The [`Task`]: the learning task a run optimizes for, reified as a type.

use crate::data::Dataset;
use ndarray::{ArrayD, Axis};
use std::collections::HashSet;
use std::error::Error;

/// The learning task a training run optimizes for. Each variant names the task and carries the
/// output width it implies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Task {
    /// Two mutually exclusive classes, one output.
    Binary,
    /// `n_classes` mutually exclusive classes, `n_classes` outputs.
    MultiClass { n_classes: usize },
    /// `n_labels` independent yes/no labels, `n_labels` outputs.
    MultiLabel { n_labels: usize },
    /// `n_outputs` real-valued targets, `n_outputs` outputs.
    Regression { n_outputs: usize },
}

impl Task {
    /// Infers the task from the shape and dtype of a dataset's targets:
    /// - rank-1 non-negative integers over two distinct values →
    ///   [`Binary`](Task::Binary), over three or more → [`MultiClass`](Task::MultiClass);
    /// - rank-1 real values → [`Regression`](Task::Regression) with one output;
    /// - rank-2 `(samples, k)` in `{0, 1}` → [`MultiLabel`](Task::MultiLabel);
    /// - rank-2 `(samples, k)` real values → [`Regression`](Task::Regression) with
    ///   `k` outputs.
    pub fn from_dataset(dataset: &Dataset) -> Self {
        let targets = dataset.targets();
        if targets.ndim() == 1 {
            let class_ids = targets.iter().all(|&v| v >= 0.0 && v.fract() == 0.0);
            match dataset.n_classes() {
                2 if class_ids => Task::Binary,
                n if class_ids && n > 2 => Task::MultiClass { n_classes: n },
                _ => Task::Regression { n_outputs: 1 },
            }
        } else {
            let width = targets.len_of(Axis(1));
            if targets.iter().all(|&v| v == 0.0 || v == 1.0) {
                Task::MultiLabel { n_labels: width }
            } else {
                Task::Regression { n_outputs: width }
            }
        }
    }

    /// Checks that a dataset's targets can support this task.
    ///
    /// # Errors
    /// An [`TaskDataError`] describing the first mismatch between the targets'
    /// shape or values and what this task requires.
    pub fn validate_dataset(&self, dataset: &Dataset) -> Result<(), TaskDataError> {
        let targets = dataset.targets();
        match self {
            Task::Binary => validate_classification(targets, 2),
            Task::MultiClass { n_classes } => validate_classification(targets, *n_classes),
            Task::MultiLabel { n_labels } => validate_multilabel(targets, *n_labels),
            Task::Regression { n_outputs } => validate_regression(targets, *n_outputs),
        }
    }

    /// The number of output units this task implies: one logit for
    /// [`Binary`](Task::Binary), and the class/label/output count for the others.
    pub fn output_size(&self) -> usize {
        match self {
            Task::Binary => 1,
            Task::MultiClass { n_classes } => *n_classes,
            Task::MultiLabel { n_labels } => *n_labels,
            Task::Regression { n_outputs } => *n_outputs,
        }
    }
}

/// Checks that `targets` are rank-1 non-negative integer class ids spanning exactly
/// `expected` classes.
fn validate_classification(targets: &ArrayD<f32>, expected: usize) -> Result<(), TaskDataError> {
    if targets.ndim() != 1 {
        return Err(TaskDataError::WrongTargetRank {
            expected: 1,
            found: targets.ndim(),
        });
    }
    if !targets.iter().all(|&v| v >= 0.0 && v.fract() == 0.0) {
        return Err(TaskDataError::InvalidClassIds);
    }
    let n_classes = targets
        .iter()
        .map(|&v| v.to_bits())
        .collect::<HashSet<u32>>()
        .len();
    if n_classes < 2 {
        return Err(TaskDataError::TooFewClasses(n_classes));
    }
    if n_classes != expected {
        return Err(TaskDataError::ClassCountMismatch {
            expected,
            found: n_classes,
        });
    }
    Ok(())
}

/// Checks that `targets` are a rank-2 `{0, 1}` indicator of width `n_labels`.
fn validate_multilabel(targets: &ArrayD<f32>, n_labels: usize) -> Result<(), TaskDataError> {
    if targets.ndim() != 2 {
        return Err(TaskDataError::WrongTargetRank {
            expected: 2,
            found: targets.ndim(),
        });
    }
    if !targets.iter().all(|&v| v == 0.0 || v == 1.0) {
        return Err(TaskDataError::LabelsNotBinary);
    }
    let width = targets.len_of(Axis(1));
    if width != n_labels {
        return Err(TaskDataError::WidthMismatch {
            expected: n_labels,
            found: width,
        });
    }
    Ok(())
}

/// Checks that `targets` are finite and `n_outputs` wide.
fn validate_regression(targets: &ArrayD<f32>, n_outputs: usize) -> Result<(), TaskDataError> {
    if !targets.iter().all(|&v| v.is_finite()) {
        return Err(TaskDataError::NonFiniteTargets);
    }
    let width = if targets.ndim() == 1 {
        1
    } else {
        targets.len_of(Axis(1))
    };
    if width != n_outputs {
        return Err(TaskDataError::WidthMismatch {
            expected: n_outputs,
            found: width,
        });
    }
    Ok(())
}

/// Errors returned when a dataset's targets cannot support an [`Task`].
#[derive(Debug, PartialEq, Eq)]
pub enum TaskDataError {
    /// The targets have the wrong rank for the task.
    WrongTargetRank { expected: usize, found: usize },
    /// The class ids are not non-negative integers.
    InvalidClassIds,
    /// Fewer than two distinct classes are present (carries the count found).
    TooFewClasses(usize),
    /// The number of classes differs from the task's declared count.
    ClassCountMismatch { expected: usize, found: usize },
    /// The multi-label targets are not all `0` or `1`.
    LabelsNotBinary,
    /// The regression targets contain a non-finite value.
    NonFiniteTargets,
    /// The target width differs from the task's declared width.
    WidthMismatch { expected: usize, found: usize },
}

impl std::fmt::Display for TaskDataError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TaskDataError::WrongTargetRank { expected, found } => {
                write!(f, "targets must be rank-{expected}, but found rank-{found}")
            }
            TaskDataError::InvalidClassIds => {
                write!(f, "class ids must be non-negative integers")
            }
            TaskDataError::TooFewClasses(n) => {
                write!(f, "a classifier needs at least 2 classes, but found {n}")
            }
            TaskDataError::ClassCountMismatch { expected, found } => write!(
                f,
                "task declares {expected} classes, but the targets carry {found}"
            ),
            TaskDataError::LabelsNotBinary => {
                write!(f, "multi-label targets must all be 0 or 1")
            }
            TaskDataError::NonFiniteTargets => {
                write!(f, "regression targets must all be finite")
            }
            TaskDataError::WidthMismatch { expected, found } => write!(
                f,
                "task declares {expected} outputs, but the targets carry {found}"
            ),
        }
    }
}

impl Error for TaskDataError {}

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
        assert_eq!(Task::from_dataset(&dataset_with_classes(2)), Task::Binary);
        assert_eq!(
            Task::from_dataset(&dataset_with_classes(4)),
            Task::MultiClass { n_classes: 4 }
        );
    }

    #[test]
    fn from_dataset_infers_regression_from_real_rank1_targets() {
        let dataset = tabular_dataset(array![0.5f32, 1.5, 2.25, 0.1]);
        assert_eq!(
            Task::from_dataset(&dataset),
            Task::Regression { n_outputs: 1 }
        );
    }

    #[test]
    fn from_dataset_infers_regression_from_a_single_class() {
        // One distinct integer label is not classifiable; inference falls back to regression.
        let dataset = tabular_dataset(array![1.0f32, 1.0, 1.0]);
        assert_eq!(
            Task::from_dataset(&dataset),
            Task::Regression { n_outputs: 1 }
        );
    }

    #[test]
    fn from_dataset_infers_multi_label_from_a_binary_indicator() {
        let targets = array![[1.0f32, 0.0, 1.0], [0.0, 1.0, 1.0]];
        assert_eq!(
            Task::from_dataset(&rank2_target_dataset(targets)),
            Task::MultiLabel { n_labels: 3 }
        );
    }

    #[test]
    fn from_dataset_infers_regression_from_real_rank2_targets() {
        let targets = array![[0.5f32, 1.0], [2.0, 3.5]];
        assert_eq!(
            Task::from_dataset(&rank2_target_dataset(targets)),
            Task::Regression { n_outputs: 2 }
        );
    }

    #[test]
    fn output_size_reflects_the_task_width() {
        assert_eq!(Task::Binary.output_size(), 1);
        assert_eq!(Task::MultiClass { n_classes: 4 }.output_size(), 4);
        assert_eq!(Task::MultiLabel { n_labels: 3 }.output_size(), 3);
        assert_eq!(Task::Regression { n_outputs: 2 }.output_size(), 2);
    }

    #[test]
    fn validate_dataset_accepts_matching_shapes() {
        assert!(
            Task::Binary
                .validate_dataset(&dataset_with_classes(2))
                .is_ok()
        );
        assert!(
            Task::MultiClass { n_classes: 4 }
                .validate_dataset(&dataset_with_classes(4))
                .is_ok()
        );
        let multilabel = rank2_target_dataset(array![[1.0f32, 0.0], [0.0, 1.0]]);
        assert!(
            Task::MultiLabel { n_labels: 2 }
                .validate_dataset(&multilabel)
                .is_ok()
        );
        let regression = rank2_target_dataset(array![[0.5f32, 1.0], [2.0, 3.5]]);
        assert!(
            Task::Regression { n_outputs: 2 }
                .validate_dataset(&regression)
                .is_ok()
        );
    }

    #[test]
    fn validate_accepts_non_contiguous_class_ids() {
        // labels = {0, 2}: two classes, gaps in the id range are no longer rejected.
        let dataset = tabular_dataset(array![0.0f32, 2.0]);
        assert!(Task::Binary.validate_dataset(&dataset).is_ok());
    }

    #[test]
    fn validate_accepts_out_of_range_multiclass_label() {
        // labels = {0, 1, 2, 5}: four classes, id 5 no longer breaks validation.
        let dataset = tabular_dataset(array![0.0f32, 1.0, 2.0, 5.0]);
        assert!(
            Task::MultiClass { n_classes: 4 }
                .validate_dataset(&dataset)
                .is_ok()
        );
    }

    #[test]
    fn validate_rejects_invalid_class_ids() {
        // A negative and a fractional value are never valid class ids.
        let dataset = tabular_dataset(array![0.0f32, -1.0]);
        assert_eq!(
            Task::Binary.validate_dataset(&dataset),
            Err(TaskDataError::InvalidClassIds)
        );
        let dataset = tabular_dataset(array![0.0f32, 1.5]);
        assert_eq!(
            Task::Binary.validate_dataset(&dataset),
            Err(TaskDataError::InvalidClassIds)
        );
    }

    #[test]
    fn validate_rejects_a_single_class() {
        let dataset = tabular_dataset(array![1.0f32, 1.0, 1.0]);
        assert_eq!(
            Task::Binary.validate_dataset(&dataset),
            Err(TaskDataError::TooFewClasses(1))
        );
    }

    #[test]
    fn validate_rejects_a_class_count_mismatch() {
        // Three contiguous classes cannot satisfy a Binary task.
        assert_eq!(
            Task::Binary.validate_dataset(&dataset_with_classes(3)),
            Err(TaskDataError::ClassCountMismatch {
                expected: 2,
                found: 3
            })
        );
    }

    #[test]
    fn validate_rejects_a_wrong_target_rank() {
        // A rank-2 target cannot satisfy a single-label classification task.
        let dataset = rank2_target_dataset(array![[1.0f32, 0.0], [0.0, 1.0]]);
        assert_eq!(
            Task::Binary.validate_dataset(&dataset),
            Err(TaskDataError::WrongTargetRank {
                expected: 1,
                found: 2
            })
        );
    }

    #[test]
    fn validate_rejects_a_non_binary_multi_label() {
        let dataset = rank2_target_dataset(array![[0.5f32, 0.0], [1.0, 1.0]]);
        assert_eq!(
            Task::MultiLabel { n_labels: 2 }.validate_dataset(&dataset),
            Err(TaskDataError::LabelsNotBinary)
        );
    }

    #[test]
    fn validate_rejects_a_multi_label_width_mismatch() {
        let dataset = rank2_target_dataset(array![[1.0f32, 0.0], [0.0, 1.0]]);
        assert_eq!(
            Task::MultiLabel { n_labels: 3 }.validate_dataset(&dataset),
            Err(TaskDataError::WidthMismatch {
                expected: 3,
                found: 2
            })
        );
    }

    #[test]
    fn validate_rejects_non_finite_regression_targets() {
        let dataset = rank2_target_dataset(array![[0.5f32, f32::NAN], [1.0, 2.0]]);
        assert_eq!(
            Task::Regression { n_outputs: 2 }.validate_dataset(&dataset),
            Err(TaskDataError::NonFiniteTargets)
        );
    }

    #[test]
    fn validate_rejects_a_regression_width_mismatch() {
        let dataset = rank2_target_dataset(array![[0.5f32, 1.0], [2.0, 3.5]]);
        assert_eq!(
            Task::Regression { n_outputs: 1 }.validate_dataset(&dataset),
            Err(TaskDataError::WidthMismatch {
                expected: 1,
                found: 2
            })
        );
    }

    #[test]
    fn task_data_error_messages_are_descriptive() {
        assert_eq!(
            TaskDataError::WrongTargetRank {
                expected: 1,
                found: 2
            }
            .to_string(),
            "targets must be rank-1, but found rank-2"
        );
        assert_eq!(
            TaskDataError::ClassCountMismatch {
                expected: 2,
                found: 3
            }
            .to_string(),
            "task declares 2 classes, but the targets carry 3"
        );
        assert_eq!(
            TaskDataError::WidthMismatch {
                expected: 1,
                found: 2
            }
            .to_string(),
            "task declares 1 outputs, but the targets carry 2"
        );
    }
}
