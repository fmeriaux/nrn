use crate::task::Task;
use serde::{Deserialize, Serialize};

/// Serializable mirror of [`Task`], one JSON object tagged by task with the width or shape it
/// carries.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TaskRecord {
    /// Mirror of [`Task::Binary`].
    Binary,
    /// Mirror of [`Task::MultiClass`].
    MultiClass { n_classes: usize },
    /// Mirror of [`Task::MultiLabel`].
    MultiLabel { n_labels: usize },
    /// Mirror of [`Task::Regression`].
    Regression { shape: Vec<usize> },
}

impl From<Task> for TaskRecord {
    fn from(task: Task) -> Self {
        match task {
            Task::Binary => TaskRecord::Binary,
            Task::MultiClass { n_classes } => TaskRecord::MultiClass { n_classes },
            Task::MultiLabel { n_labels } => TaskRecord::MultiLabel { n_labels },
            Task::Regression { shape } => TaskRecord::Regression { shape },
        }
    }
}

impl From<TaskRecord> for Task {
    fn from(record: TaskRecord) -> Self {
        match record {
            TaskRecord::Binary => Task::Binary,
            TaskRecord::MultiClass { n_classes } => Task::MultiClass { n_classes },
            TaskRecord::MultiLabel { n_labels } => Task::MultiLabel { n_labels },
            TaskRecord::Regression { shape } => Task::Regression { shape },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_round_trips_every_task() {
        for task in [
            Task::Binary,
            Task::MultiClass { n_classes: 4 },
            Task::MultiLabel { n_labels: 3 },
            Task::Regression { shape: vec![2] },
            Task::Regression {
                shape: vec![3, 4, 4],
            },
        ] {
            assert_eq!(Task::from(TaskRecord::from(task.clone())), task);
        }
    }

    #[test]
    fn record_serializes_tagged_by_task() {
        let json = serde_json::to_string(&TaskRecord::MultiClass { n_classes: 4 }).unwrap();
        assert_eq!(json, r#"{"type":"multi_class","n_classes":4}"#);
    }
}
