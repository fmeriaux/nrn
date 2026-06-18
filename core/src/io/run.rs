use crate::io::checkpoint::{CheckpointArchive, CheckpointRecorder, scan_checkpoints};
use crate::io::hyperparams::HyperParametersRecord;
use crate::io::json;
use crate::io::path::PathExt;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::ErrorKind::AlreadyExists;
use std::io::{Error, Result};
use std::path::{Path, PathBuf};

/// Top-level metadata for a training run directory.
/// Written once by [`TrainingRun::create`] into `meta.json`.
#[derive(Serialize, Deserialize, Clone)]
pub struct TrainingMeta {
    pub dataset: String,
    pub hyperparams: HyperParametersRecord,
}

/// A handle to a training run directory: its location and run-level metadata.
/// Produces [`CheckpointRecorder`]s (the pure runtime callback) and
/// [`CheckpointArchive`]s (read-only access to recorded checkpoints).
pub struct TrainingRun {
    dir: PathBuf,
    meta: TrainingMeta,
}

impl TrainingRun {
    /// Creates a fresh run directory at `path`, writing `meta.json`.
    ///
    /// Returns an error if `checkpoint-*` subdirectories already exist and
    /// `overwrite` is `false`; otherwise they are removed.
    pub fn create<P: AsRef<Path>>(path: P, meta: &TrainingMeta, overwrite: bool) -> Result<Self> {
        let dir = Path::combine_safe_with_cwd(path)?;
        fs::create_dir_all(&dir)?;

        let existing = scan_checkpoints(&dir)?;
        if !existing.is_empty() {
            if !overwrite {
                return Err(Error::new(
                    AlreadyExists,
                    format!("training history already exists at {}", dir.display()),
                ));
            }
            for checkpoint in existing {
                fs::remove_dir_all(&checkpoint.dir)?;
            }
        }

        json::save(meta, dir.join("meta"))?;

        Ok(TrainingRun {
            dir,
            meta: meta.clone(),
        })
    }

    /// Opens an existing run directory, loading `meta.json`.
    ///
    /// Returns a `NotFound` error if the directory or its `meta.json` doesn't exist.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let dir = Path::combine_safe_with_cwd(path)?;
        let meta: TrainingMeta = json::load(dir.join("meta"))?;
        Ok(TrainingRun { dir, meta })
    }

    /// Returns this run's metadata.
    pub fn meta(&self) -> &TrainingMeta {
        &self.meta
    }

    /// Returns a recorder for writing checkpoints into this run.
    pub fn recorder(&self) -> CheckpointRecorder {
        CheckpointRecorder::new(self.dir.clone())
    }

    /// Removes checkpoints whose epoch is greater than `from_epoch`, rewinding
    /// the trajectory. Returns the number of checkpoints removed.
    pub fn trim_after(&self, from_epoch: usize) -> Result<usize> {
        let to_remove: Vec<_> = scan_checkpoints(&self.dir)?
            .into_iter()
            .filter(|checkpoint| checkpoint.epoch > from_epoch)
            .collect();

        let trimmed = to_remove.len();
        for checkpoint in to_remove {
            fs::remove_dir_all(&checkpoint.dir)?;
        }

        Ok(trimmed)
    }

    /// Returns a read-only archive over this run's checkpoints.
    pub fn archive(&self) -> Result<CheckpointArchive> {
        CheckpointArchive::load(&self.dir)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = PathBuf::from(format!("target/nrn_run_{}_{}", tag, std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn cleanup(dir: &Path) {
        let _ = fs::remove_dir_all(dir);
    }

    fn meta(dataset: &str) -> TrainingMeta {
        TrainingMeta {
            dataset: dataset.to_string(),
            hyperparams: HyperParametersRecord::sample(),
        }
    }

    /// Creates an empty `checkpoint-{epoch:06}/` directory. The run lifecycle
    /// (create/trim) only cares about which checkpoint directories exist, not
    /// their contents — that I/O is exercised in [`crate::io::checkpoint`].
    fn make_checkpoint(dir: &Path, epoch: usize) {
        fs::create_dir_all(dir.join(format!("checkpoint-{epoch:06}"))).unwrap();
    }

    #[test]
    fn meta_json_written_by_create() {
        let dir = temp_dir("meta");
        TrainingRun::create(&dir, &meta("my_dataset"), false).unwrap();

        let loaded = TrainingRun::open(&dir).unwrap();
        cleanup(&dir);

        assert_eq!(loaded.meta().dataset, "my_dataset");
    }

    #[test]
    fn create_errors_if_checkpoints_exist_without_overwrite() {
        let dir = temp_dir("no_overwrite");
        TrainingRun::create(&dir, &meta("ds"), false).unwrap();
        make_checkpoint(&dir, 0);

        let result = TrainingRun::create(&dir, &meta("ds"), false);
        cleanup(&dir);

        let err = result.err().unwrap();
        assert_eq!(err.kind(), AlreadyExists);
        assert!(err.to_string().contains("already exists"));
    }

    #[test]
    fn create_with_overwrite_purges_previous_checkpoints() {
        let dir = temp_dir("overwrite");
        TrainingRun::create(&dir, &meta("ds"), false).unwrap();
        for i in 0..3 {
            make_checkpoint(&dir, i * 10);
        }

        TrainingRun::create(&dir, &meta("ds"), true).unwrap();
        make_checkpoint(&dir, 0);

        let archive = CheckpointArchive::load(&dir).unwrap();
        cleanup(&dir);

        assert_eq!(archive.len(), 1, "stale checkpoints were not purged");
    }

    #[test]
    fn trim_after_removes_checkpoints_past_from_epoch() {
        let dir = temp_dir("resume_trim");
        TrainingRun::create(&dir, &meta("ds"), false).unwrap();
        for i in 0..5 {
            make_checkpoint(&dir, i * 10);
        }

        let trimmed = TrainingRun::open(&dir).unwrap().trim_after(20).unwrap();

        let archive = CheckpointArchive::load(&dir).unwrap();
        cleanup(&dir);

        assert_eq!(archive.len(), 3); // epochs 0, 10, 20 kept; 30, 40 removed
        assert_eq!(trimmed, 2);
    }

    #[test]
    fn trim_after_keeps_all_when_from_epoch_is_last() {
        let dir = temp_dir("resume_keep_all");
        TrainingRun::create(&dir, &meta("ds"), false).unwrap();
        for i in 0..3 {
            make_checkpoint(&dir, i * 10);
        }

        let trimmed = TrainingRun::open(&dir).unwrap().trim_after(20).unwrap();

        let archive = CheckpointArchive::load(&dir).unwrap();
        cleanup(&dir);

        assert_eq!(archive.len(), 3);
        assert_eq!(trimmed, 0);
    }

    #[test]
    fn trim_after_lets_a_checkpoint_be_repopulated() {
        let dir = temp_dir("resume_write");
        TrainingRun::create(&dir, &meta("ds"), false).unwrap();
        for i in 0..3 {
            make_checkpoint(&dir, i * 10); // 0, 10, 20
        }

        let run = TrainingRun::open(&dir).unwrap();
        run.trim_after(10).unwrap(); // drops 20
        make_checkpoint(&dir, 20); // rewritten

        let archive = CheckpointArchive::load(&dir).unwrap();
        cleanup(&dir);

        assert_eq!(archive.len(), 3); // 0, 10, 20
    }

    #[test]
    fn create_rejects_path_traversal() {
        let result = TrainingRun::create("../../nrn_traversal_test", &meta("x"), false);
        assert!(result.is_err());
    }

    #[test]
    fn open_rejects_path_traversal() {
        let result = TrainingRun::open("../../nrn_traversal_test");
        assert!(result.is_err());
    }

    #[test]
    fn open_errors_when_meta_missing() {
        let dir = temp_dir("open_missing");
        let result = TrainingRun::open(&dir);
        cleanup(&dir);

        assert!(result.is_err());
    }
}
