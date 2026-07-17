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
    MultiClass { class_count: usize },
    /// Mirror of [`Task::MultiLabel`].
    MultiLabel { label_count: usize },
    /// Mirror of [`Task::Regression`].
    Regression { target_shape: Vec<usize> },
}

impl From<Task> for TaskRecord {
    fn from(task: Task) -> Self {
        match task {
            Task::Binary => TaskRecord::Binary,
            Task::MultiClass { class_count } => TaskRecord::MultiClass { class_count },
            Task::MultiLabel { label_count } => TaskRecord::MultiLabel { label_count },
            Task::Regression { target_shape } => TaskRecord::Regression { target_shape },
        }
    }
}

impl From<TaskRecord> for Task {
    fn from(record: TaskRecord) -> Self {
        match record {
            TaskRecord::Binary => Task::Binary,
            TaskRecord::MultiClass { class_count } => Task::MultiClass { class_count },
            TaskRecord::MultiLabel { label_count } => Task::MultiLabel { label_count },
            TaskRecord::Regression { target_shape } => Task::Regression { target_shape },
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
            Task::MultiClass { class_count: 4 },
            Task::MultiLabel { label_count: 3 },
            Task::Regression {
                target_shape: vec![2],
            },
            Task::Regression {
                target_shape: vec![3, 4, 4],
            },
        ] {
            assert_eq!(Task::from(TaskRecord::from(task.clone())), task);
        }
    }

    #[test]
    fn record_serializes_tagged_by_task() {
        let json = serde_json::to_string(&TaskRecord::MultiClass { class_count: 4 }).unwrap();
        assert_eq!(json, r#"{"type":"multi_class","class_count":4}"#);
    }
}
