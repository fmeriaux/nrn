//! Layer and network configuration: the declarative description of a network's architecture.
//! [`NetworkConfig`] bundles the per-sample input shape with an ordered stack of
//! [`LayerConfig`]s, one per layer kind. [`ModelConfig`] pairs a [`Task`] with the [`Labels`]
//! naming its classes or multi-label positions, when known.

mod labels;
mod layer;
mod network;

pub use labels::*;
pub use layer::*;
pub use network::*;

use crate::task::Task;
use std::fmt;

/// A [`Task`] paired with the name vocabulary for its classes or multi-label positions, when known.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelConfig {
    task: Task,
    labels: Option<Labels>,
}

impl ModelConfig {
    /// Pairs `task` with `labels`, validating that `labels`, when given, name exactly `task`'s
    /// classes or labels.
    ///
    /// # Errors
    /// - [`ModelConfigError::UnclassifiedTask`] when `labels` is given but `task` carries no
    ///   discrete outputs to name (regression).
    /// - [`ModelConfigError::LabelCountMismatch`] when `labels`' count differs from the number
    ///   of classes or labels `task` declares.
    pub fn new(task: Task, labels: Option<Labels>) -> Result<Self, ModelConfigError> {
        if let Some(labels) = &labels {
            let expected = match &task {
                Task::Binary => 2,
                Task::MultiClass { class_count } => *class_count,
                Task::MultiLabel { label_count } => *label_count,
                Task::Regression { .. } => {
                    return Err(ModelConfigError::UnclassifiedTask);
                }
            };
            if labels.len() != expected {
                return Err(ModelConfigError::LabelCountMismatch {
                    expected,
                    found: labels.len(),
                });
            }
        }
        Ok(Self { task, labels })
    }

    /// The learning task.
    pub fn task(&self) -> &Task {
        &self.task
    }

    /// The task's name vocabulary, when known.
    pub fn labels(&self) -> Option<&Labels> {
        self.labels.as_ref()
    }
}

/// Errors returned when a [`Task`] and [`Labels`] cannot form a valid [`ModelConfig`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelConfigError {
    /// Labels were given for a task with no discrete outputs to name (regression).
    UnclassifiedTask,
    /// The number of labels differs from the number of classes or labels the task declares.
    LabelCountMismatch {
        /// The number of classes or labels `task` declares.
        expected: usize,
        /// The number of labels given.
        found: usize,
    },
}

impl fmt::Display for ModelConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelConfigError::UnclassifiedTask => {
                write!(
                    f,
                    "labels require discrete outputs, but the task is a regression"
                )
            }
            ModelConfigError::LabelCountMismatch { expected, found } => {
                write!(f, "task declares {expected} labels, but {found} were given")
            }
        }
    }
}

impl std::error::Error for ModelConfigError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn labels(names: &[&str]) -> Labels {
        Labels::new(names.iter().map(|s| s.to_string()).collect())
    }

    #[test]
    fn new_accepts_no_labels_for_any_task() {
        assert!(ModelConfig::new(Task::Binary, None).is_ok());
        assert!(
            ModelConfig::new(
                Task::Regression {
                    target_shape: vec![2]
                },
                None
            )
            .is_ok()
        );
    }

    #[test]
    fn new_accepts_labels_matching_the_declared_count() {
        assert!(ModelConfig::new(Task::Binary, Some(labels(&["cat", "dog"]))).is_ok());
        assert!(
            ModelConfig::new(
                Task::MultiClass { class_count: 3 },
                Some(labels(&["cat", "dog", "bird"]))
            )
            .is_ok()
        );
        assert!(
            ModelConfig::new(
                Task::MultiLabel { label_count: 2 },
                Some(labels(&["has_cat", "outdoor"]))
            )
            .is_ok()
        );
    }

    #[test]
    fn new_rejects_a_label_count_mismatch() {
        assert_eq!(
            ModelConfig::new(Task::Binary, Some(labels(&["cat", "dog", "bird"]))),
            Err(ModelConfigError::LabelCountMismatch {
                expected: 2,
                found: 3
            })
        );
    }

    #[test]
    fn new_rejects_labels_on_a_regression_task() {
        assert_eq!(
            ModelConfig::new(
                Task::Regression {
                    target_shape: vec![2]
                },
                Some(labels(&["cat", "dog"]))
            ),
            Err(ModelConfigError::UnclassifiedTask)
        );
    }

    #[test]
    fn model_config_error_messages_are_descriptive() {
        assert_eq!(
            ModelConfigError::UnclassifiedTask.to_string(),
            "labels require discrete outputs, but the task is a regression"
        );
        assert_eq!(
            ModelConfigError::LabelCountMismatch {
                expected: 2,
                found: 3
            }
            .to_string(),
            "task declares 2 labels, but 3 were given"
        );
    }
}
