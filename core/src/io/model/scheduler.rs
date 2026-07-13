use crate::io::json;
use crate::schedulers::SchedulerState;
use serde::{Deserialize, Serialize};
use std::io::Result;
use std::path::{Path, PathBuf};

/// JSON-serializable mirror of [`SchedulerState`], kept here so the core type
/// stays free of serde (see its doc comment).
#[derive(Serialize, Deserialize)]
struct SchedulerStateRecord {
    current_step: usize,
}

impl From<&SchedulerState> for SchedulerStateRecord {
    fn from(state: &SchedulerState) -> Self {
        SchedulerStateRecord {
            current_step: state.current_step,
        }
    }
}

impl From<SchedulerStateRecord> for SchedulerState {
    fn from(record: SchedulerStateRecord) -> Self {
        SchedulerState {
            current_step: record.current_step,
        }
    }
}

/// Saves a [`SchedulerState`] to a `.json` file.
pub fn save<P: AsRef<Path>>(state: &SchedulerState, path: P) -> Result<PathBuf> {
    json::save(&SchedulerStateRecord::from(state), path)
}

/// Loads a [`SchedulerState`] previously written by [`save`].
pub fn load<P: AsRef<Path>>(path: P) -> Result<SchedulerState> {
    json::load::<SchedulerStateRecord, _>(path).map(SchedulerState::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn temp_path(tag: &str) -> PathBuf {
        PathBuf::from(format!(
            "target/nrn_test_scheduler_{tag}_{}",
            std::process::id()
        ))
    }

    fn cleanup(path: &Path) {
        let _ = std::fs::remove_file(path.with_extension("json"));
    }

    #[test]
    fn save_load_roundtrip_preserves_current_step() {
        let state = SchedulerState { current_step: 7 };

        let path = temp_path("roundtrip");
        save(&state, &path).unwrap();
        let loaded = load(&path).unwrap();
        cleanup(&path);

        assert_eq!(loaded.current_step, 7);
    }
}
