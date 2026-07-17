//! The [`Task`]: the learning task a run optimizes for, reified as a type.

use crate::data::{Dataset, TargetKind, Targets};
use ndarray::Axis;
use std::error::Error;

/// The learning task a training run optimizes for. Each variant names the task and carries the
/// output shape it implies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Task {
    /// Two mutually exclusive classes, one output.
    Binary,
    /// `class_count` mutually exclusive classes, `class_count` outputs.
    MultiClass { class_count: usize },
    /// `label_count` independent labels in `[0, 1]`, `label_count` outputs.
    MultiLabel { label_count: usize },
    /// Real-valued targets, shaped `target_shape` per sample.
    Regression { target_shape: Vec<usize> },
}

impl Task {
    /// Infers the task from a dataset's targets:
    /// - `ClassLabel` over two classes → [`Binary`](Task::Binary), over three or more →
    ///   [`MultiClass`](Task::MultiClass);
    /// - rank-2 `(samples, k)` `Value` targets in `{0, 1}` → [`MultiLabel`](Task::MultiLabel);
    /// - any other `Value` targets → [`Regression`](Task::Regression), shaped by the dataset's
    ///   per-sample (trailing) axes.
    pub fn from_dataset(dataset: &Dataset) -> Self {
        match dataset.targets() {
            Targets::ClassLabel(label) => {
                let class_count = label.class_count();
                if class_count == 2 {
                    Task::Binary
                } else {
                    Task::MultiClass { class_count }
                }
            }
            Targets::Value(values)
                if values.as_array().ndim() == 2
                    && values.as_array().iter().all(|&v| v == 0.0 || v == 1.0) =>
            {
                Task::MultiLabel {
                    label_count: values.as_array().len_of(Axis(1)),
                }
            }
            Targets::Value(values) => Task::Regression {
                target_shape: values.as_array().shape()[1..].to_vec(),
            },
        }
    }

    /// Checks that a dataset's targets can support this task.
    ///
    /// # Errors
    /// An [`TaskDataError`] describing the first mismatch between the targets'
    /// dtype, shape, or values and what this task requires.
    pub fn validate_dataset(&self, dataset: &Dataset) -> Result<(), TaskDataError> {
        let targets = dataset.targets();
        match self {
            Task::Binary => validate_multiclass(targets, 2),
            Task::MultiClass { class_count } => validate_multiclass(targets, *class_count),
            Task::MultiLabel { label_count } => validate_multilabel(targets, *label_count),
            Task::Regression { target_shape } => validate_regression(targets, target_shape),
        }
    }

    /// The per-sample output shape this task implies: a single-element shape carrying the
    /// class/label count for every task but [`Regression`](Task::Regression), which carries
    /// its own declared target shape (a scalar target's empty shape becomes the single output
    /// neuron that produces it).
    pub fn output_shape(&self) -> Vec<usize> {
        match self {
            Task::Binary => vec![1],
            Task::MultiClass { class_count } => vec![*class_count],
            Task::MultiLabel { label_count } => vec![*label_count],
            Task::Regression { target_shape } if target_shape.is_empty() => vec![1],
            Task::Regression { target_shape } => target_shape.clone(),
        }
    }

    /// The flattened output width this task implies: the element count of
    /// [`output_shape`](Self::output_shape).
    pub fn output_size(&self) -> usize {
        self.output_shape().iter().product()
    }
}

/// Checks that `targets` are `ClassLabel` spanning exactly `expected` classes.
fn validate_multiclass(targets: &Targets, expected: usize) -> Result<(), TaskDataError> {
    let Targets::ClassLabel(label) = targets else {
        return Err(TaskDataError::WrongTargetKind {
            expected: TargetKind::ClassLabel,
            found: targets.kind(),
        });
    };
    let class_count = label.class_count();
    if class_count != expected {
        return Err(TaskDataError::ClassCountMismatch {
            expected,
            found: class_count,
        });
    }
    Ok(())
}

/// Checks that `targets` are `Value`, shaped `expected` per sample.
fn validate_regression(targets: &Targets, expected: &[usize]) -> Result<(), TaskDataError> {
    if targets.kind() != TargetKind::Value {
        return Err(TaskDataError::WrongTargetKind {
            expected: TargetKind::Value,
            found: targets.kind(),
        });
    }
    let found = targets.shape();
    if found != expected {
        return Err(TaskDataError::ShapeMismatch {
            expected: expected.to_vec(),
            found: found.to_vec(),
        });
    }
    Ok(())
}

/// Checks that `targets` are `Value`, shaped `[label_count]` per sample, with every value in
/// `[0, 1]` — the range a multi-label task's loss (soft or hard indicators) expects.
fn validate_multilabel(targets: &Targets, label_count: usize) -> Result<(), TaskDataError> {
    let Targets::Value(values) = targets else {
        return Err(TaskDataError::WrongTargetKind {
            expected: TargetKind::Value,
            found: targets.kind(),
        });
    };
    let expected = vec![label_count];
    let found = targets.shape().to_vec();
    if found != expected {
        return Err(TaskDataError::ShapeMismatch { expected, found });
    }
    if values.as_array().iter().any(|&v| !(0.0..=1.0).contains(&v)) {
        return Err(TaskDataError::LabelsOutOfRange);
    }
    Ok(())
}

/// Errors returned when a dataset's targets cannot support an [`Task`].
#[derive(Debug, PartialEq, Eq)]
pub enum TaskDataError {
    /// The targets are not the dtype this task requires.
    WrongTargetKind {
        expected: TargetKind,
        found: TargetKind,
    },
    /// The number of classes differs from the task's declared count.
    ClassCountMismatch { expected: usize, found: usize },
    /// The targets' per-sample shape differs from the task's declared shape.
    ShapeMismatch {
        expected: Vec<usize>,
        found: Vec<usize>,
    },
    /// A multi-label target falls outside `[0, 1]`.
    LabelsOutOfRange,
}

impl std::fmt::Display for TaskDataError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TaskDataError::WrongTargetKind { expected, found } => {
                write!(f, "task requires {expected} targets, but found {found}")
            }
            TaskDataError::ClassCountMismatch { expected, found } => write!(
                f,
                "task declares {expected} classes, but the targets carry {found}"
            ),
            TaskDataError::ShapeMismatch { expected, found } => write!(
                f,
                "task declares shape {expected:?}, but the targets carry shape {found:?}"
            ),
            TaskDataError::LabelsOutOfRange => {
                write!(f, "multi-label targets must all lie within [0, 1]")
            }
        }
    }
}

impl Error for TaskDataError {}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2, array};

    /// A two-feature dataset carrying `ids` as contiguous `ClassLabel` targets.
    fn classification_dataset(ids: Vec<u32>) -> Dataset {
        let inputs = Array2::from_shape_fn((ids.len(), 2), |(i, _)| i as f32);
        let targets = Targets::class_label(Array1::from(ids), None).unwrap();
        Dataset::new(inputs.into_dyn(), targets, None).unwrap()
    }

    /// A two-feature dataset carrying `values` as `Value` targets.
    fn value_dataset(values: Array2<f32>) -> Dataset {
        let inputs = Array2::from_shape_fn((values.nrows(), 2), |(i, _)| i as f32).into_dyn();
        Dataset::new(inputs, Targets::value(values.into_dyn()).unwrap(), None).unwrap()
    }

    /// A rank-3 `(samples, height, width)` `Value`-target dataset.
    fn rank3_value_dataset(values: ndarray::Array3<f32>) -> Dataset {
        let inputs =
            Array2::from_shape_fn((values.len_of(Axis(0)), 2), |(i, _)| i as f32).into_dyn();
        Dataset::new(inputs, Targets::value(values.into_dyn()).unwrap(), None).unwrap()
    }

    /// A dataset with `n_classes` contiguous class ids spread across ten samples.
    fn dataset_with_classes(n_classes: usize) -> Dataset {
        classification_dataset((0..10).map(|i| (i % n_classes) as u32).collect())
    }

    #[test]
    fn from_dataset_infers_binary_and_multi_class() {
        assert_eq!(Task::from_dataset(&dataset_with_classes(2)), Task::Binary);
        assert_eq!(
            Task::from_dataset(&dataset_with_classes(4)),
            Task::MultiClass { class_count: 4 }
        );
    }

    #[test]
    fn from_dataset_infers_multi_label_from_a_binary_indicator() {
        let targets = array![[1.0f32, 0.0, 1.0], [0.0, 1.0, 1.0]];
        assert_eq!(
            Task::from_dataset(&value_dataset(targets)),
            Task::MultiLabel { label_count: 3 }
        );
    }

    #[test]
    fn from_dataset_infers_regression_from_real_rank2_targets() {
        let targets = array![[0.5f32, 1.0], [2.0, 3.5]];
        assert_eq!(
            Task::from_dataset(&value_dataset(targets)),
            Task::Regression {
                target_shape: vec![2]
            }
        );
    }

    #[test]
    fn from_dataset_infers_regression_shape_from_real_rank3_targets() {
        // A (samples, height, width) target block: the per-sample shape is the trailing axes.
        let targets = ndarray::Array3::from_shape_fn((2, 3, 4), |(s, h, w)| (s + h + w) as f32);
        assert_eq!(
            Task::from_dataset(&rank3_value_dataset(targets)),
            Task::Regression {
                target_shape: vec![3, 4]
            }
        );
    }

    #[test]
    fn output_shape_reflects_the_task_shape() {
        assert_eq!(Task::Binary.output_shape(), vec![1]);
        assert_eq!(Task::MultiClass { class_count: 4 }.output_shape(), vec![4]);
        assert_eq!(Task::MultiLabel { label_count: 3 }.output_shape(), vec![3]);
        assert_eq!(
            Task::Regression {
                target_shape: vec![3, 4, 4]
            }
            .output_shape(),
            vec![3, 4, 4]
        );
    }

    #[test]
    fn output_shape_normalizes_a_scalar_regression_target_to_one_neuron() {
        // A rank-1 `(samples,)` target has an empty per-sample shape, but a Dense layer
        // always has at least one output neuron.
        assert_eq!(
            Task::Regression {
                target_shape: vec![]
            }
            .output_shape(),
            vec![1]
        );
    }

    #[test]
    fn output_size_flattens_the_task_shape() {
        assert_eq!(Task::Binary.output_size(), 1);
        assert_eq!(Task::MultiClass { class_count: 4 }.output_size(), 4);
        assert_eq!(Task::MultiLabel { label_count: 3 }.output_size(), 3);
        assert_eq!(
            Task::Regression {
                target_shape: vec![2]
            }
            .output_size(),
            2
        );
        assert_eq!(
            Task::Regression {
                target_shape: vec![3, 4, 4]
            }
            .output_size(),
            48
        );
    }

    #[test]
    fn validate_dataset_accepts_matching_shapes() {
        assert!(
            Task::Binary
                .validate_dataset(&dataset_with_classes(2))
                .is_ok()
        );
        assert!(
            Task::MultiClass { class_count: 4 }
                .validate_dataset(&dataset_with_classes(4))
                .is_ok()
        );
        let multilabel = value_dataset(array![[1.0f32, 0.0], [0.0, 1.0]]);
        assert!(
            Task::MultiLabel { label_count: 2 }
                .validate_dataset(&multilabel)
                .is_ok()
        );
        let regression = value_dataset(array![[0.5f32, 1.0], [2.0, 3.5]]);
        assert!(
            Task::Regression {
                target_shape: vec![2]
            }
            .validate_dataset(&regression)
            .is_ok()
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
    fn validate_rejects_class_label_targets_for_a_value_task() {
        let dataset = dataset_with_classes(2);
        assert_eq!(
            Task::MultiLabel { label_count: 2 }.validate_dataset(&dataset),
            Err(TaskDataError::WrongTargetKind {
                expected: TargetKind::Value,
                found: TargetKind::ClassLabel
            })
        );
    }

    #[test]
    fn validate_rejects_value_targets_for_a_classification_task() {
        let dataset = value_dataset(array![[1.0f32, 0.0], [0.0, 1.0]]);
        assert_eq!(
            Task::Binary.validate_dataset(&dataset),
            Err(TaskDataError::WrongTargetKind {
                expected: TargetKind::ClassLabel,
                found: TargetKind::Value
            })
        );
    }

    #[test]
    fn validate_rejects_a_shape_mismatch() {
        let dataset = value_dataset(array![[1.0f32, 0.0], [0.0, 1.0]]);
        assert_eq!(
            Task::MultiLabel { label_count: 3 }.validate_dataset(&dataset),
            Err(TaskDataError::ShapeMismatch {
                expected: vec![3],
                found: vec![2]
            })
        );
    }

    #[test]
    fn validate_rejects_out_of_range_multi_label_targets() {
        let dataset = value_dataset(array![[1.0f32, -0.1], [0.0, 1.2]]);
        assert_eq!(
            Task::MultiLabel { label_count: 2 }.validate_dataset(&dataset),
            Err(TaskDataError::LabelsOutOfRange)
        );
    }

    #[test]
    fn validate_accepts_soft_multi_label_targets_within_range() {
        let dataset = value_dataset(array![[0.2f32, 0.8], [0.9, 0.1]]);
        assert!(
            Task::MultiLabel { label_count: 2 }
                .validate_dataset(&dataset)
                .is_ok()
        );
    }

    #[test]
    fn validate_rejects_a_regression_rank_mismatch_with_a_matching_element_count() {
        // A (3, 4) target block flattens to 12 elements, same as a declared [12] shape, but
        // the rank differs — a flattened count match is not enough.
        let targets = ndarray::Array3::from_shape_fn((2, 3, 4), |(s, h, w)| (s + h + w) as f32);
        let dataset = rank3_value_dataset(targets);
        assert_eq!(
            Task::Regression {
                target_shape: vec![12]
            }
            .validate_dataset(&dataset),
            Err(TaskDataError::ShapeMismatch {
                expected: vec![12],
                found: vec![3, 4]
            })
        );
    }

    #[test]
    fn task_data_error_messages_are_descriptive() {
        assert_eq!(
            TaskDataError::WrongTargetKind {
                expected: TargetKind::ClassLabel,
                found: TargetKind::Value
            }
            .to_string(),
            "task requires class-label targets, but found numeric"
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
            TaskDataError::ShapeMismatch {
                expected: vec![1],
                found: vec![3, 4]
            }
            .to_string(),
            "task declares shape [1], but the targets carry shape [3, 4]"
        );
        assert_eq!(
            TaskDataError::LabelsOutOfRange.to_string(),
            "multi-label targets must all lie within [0, 1]"
        );
    }
}
