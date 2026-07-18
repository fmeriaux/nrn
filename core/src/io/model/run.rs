use crate::io::json;
use crate::io::model::checkpoint::{CheckpointArchive, CheckpointRecorder};
use crate::io::model::config::{CONFIG_STEM, ModelConfigRecord, PREPROCESSOR_STEM};
use crate::io::model::hyperparams::HyperParametersRecord;
use crate::io::model::scalers::ScalerRecord;
use crate::io::path::PathExt;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::ErrorKind::AlreadyExists;
use std::io::{Error, Result};
use std::path::{Path, PathBuf};

/// Training-lifecycle metadata for a training run directory. Written once by
/// [`TrainingRun::create`] into `meta.json`; the model blueprint and scaler live
/// alongside it in `config.json` / `preprocessor.json`.
#[derive(Serialize, Deserialize, Clone)]
pub struct TrainingMeta {
    /// Bare file name (stem) of the dataset the run trained on.
    pub dataset: String,
    /// Bare file name (stem) of the run's final-model artifact.
    pub model: String,
    /// Hyperparameters the run was configured with.
    pub hyperparams: HyperParametersRecord,
}

/// A handle to a training run directory: its location and run-level metadata.
/// Produces [`CheckpointRecorder`]s (the pure runtime callback) and
/// [`CheckpointArchive`]s (read-only access to recorded checkpoints).
pub struct TrainingRun {
    dir: PathBuf,
    meta: TrainingMeta,
    config: ModelConfigRecord,
    scaler: Option<ScalerRecord>,
}

impl TrainingRun {
    /// Creates a fresh run directory at `path`, writing `meta.json`, `config.json`, and,
    /// when `scaler` is present, `preprocessor.json`.
    ///
    /// Returns an error if `checkpoint-*` subdirectories already exist and
    /// `overwrite` is `false`; otherwise they are removed.
    pub fn create<P: AsRef<Path>>(
        path: P,
        meta: &TrainingMeta,
        config: &ModelConfigRecord,
        scaler: Option<&ScalerRecord>,
        overwrite: bool,
    ) -> Result<Self> {
        let dir = Path::combine_safe_with_cwd(path)?;
        fs::create_dir_all(&dir)?;

        let existing = CheckpointArchive::load(&dir, config.network.clone())?;
        if !existing.is_empty() {
            if !overwrite {
                return Err(Error::new(
                    AlreadyExists,
                    format!("training history already exists at {}", dir.display()),
                ));
            }
            for checkpoint in existing.entries() {
                fs::remove_dir_all(&checkpoint.dir)?;
            }
        }

        json::save(meta, dir.join("meta"))?;
        config.save(dir.join(CONFIG_STEM))?;
        if let Some(scaler) = scaler {
            scaler.save(dir.join(PREPROCESSOR_STEM))?;
        }

        Ok(TrainingRun {
            dir,
            meta: meta.clone(),
            config: config.clone(),
            scaler: scaler.cloned(),
        })
    }

    /// Opens an existing run directory, loading `meta.json` and `config.json`. The scaler is
    /// read only when a `preprocessor.json` sidecar exists.
    ///
    /// Returns a `NotFound` error if the directory or its `meta.json`/`config.json` doesn't exist.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let dir = Path::combine_safe_with_cwd(path)?;
        let meta: TrainingMeta = json::load(dir.join("meta"))?;
        let config = ModelConfigRecord::load(dir.join(CONFIG_STEM))?;
        let scaler = dir
            .join(PREPROCESSOR_STEM)
            .optional_sidecar("json")
            .map(ScalerRecord::load)
            .transpose()?;

        Ok(TrainingRun {
            dir,
            meta,
            config,
            scaler,
        })
    }

    /// Returns this run's training-lifecycle metadata.
    pub fn meta(&self) -> &TrainingMeta {
        &self.meta
    }

    /// Returns this run's model blueprint: the architecture its checkpoint weights belong to,
    /// and the task it was trained for.
    pub fn config(&self) -> &ModelConfigRecord {
        &self.config
    }

    /// Returns this run's scaler, fitted on the train split and immutable across the run,
    /// or `None` when the run applies no scaling.
    pub fn scaler(&self) -> Option<&ScalerRecord> {
        self.scaler.as_ref()
    }

    /// Returns a recorder for writing checkpoints into this run.
    pub fn recorder(&self) -> CheckpointRecorder {
        CheckpointRecorder::new(self.dir.clone())
    }

    /// Removes checkpoints whose epoch is greater than `from_epoch`, rewinding
    /// the trajectory. Returns the number of checkpoints removed.
    pub fn trim_after(&self, from_epoch: usize) -> Result<usize> {
        let archive = CheckpointArchive::load(&self.dir, self.config.network.clone())?;
        let to_remove: Vec<&Path> = archive
            .entries()
            .iter()
            .filter(|checkpoint| checkpoint.epoch > from_epoch)
            .map(|checkpoint| checkpoint.dir.as_path())
            .collect();

        let trimmed = to_remove.len();
        for dir in to_remove {
            fs::remove_dir_all(dir)?;
        }

        Ok(trimmed)
    }

    /// Returns a read-only archive over this run's checkpoints.
    pub fn archive(&self) -> Result<CheckpointArchive> {
        CheckpointArchive::load(&self.dir, self.config.network.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::model::network::NetworkConfigRecord;
    use crate::io::model::task::TaskRecord;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = PathBuf::from(format!("target/nrn_run_{}_{}", tag, std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn cleanup(dir: &Path) {
        let _ = fs::remove_dir_all(dir);
    }

    fn sample_network() -> NetworkConfigRecord {
        use crate::activations::{IDENTITY, RELU};
        use crate::model::{NetworkConfig, NeuralNetwork};

        let config = NetworkConfig::builder(vec![2])
            .dense(3, &RELU)
            .dense(1, &IDENTITY)
            .build();
        NetworkConfigRecord::from(&NeuralNetwork::from_config(config, 0).unwrap())
    }

    fn sample_config() -> ModelConfigRecord {
        ModelConfigRecord {
            network: sample_network(),
            task: TaskRecord::Binary,
            labels: None,
        }
    }

    fn meta(dataset: &str) -> TrainingMeta {
        TrainingMeta {
            dataset: dataset.to_string(),
            model: format!("model-{dataset}"),
            hyperparams: HyperParametersRecord::sample(),
        }
    }

    /// Creates an empty `checkpoint-{epoch:06}/` directory. The run lifecycle
    /// (create/trim) only cares about which checkpoint directories exist, not
    /// their contents — that I/O is exercised in [`crate::io::model::checkpoint`].
    fn make_checkpoint(dir: &Path, epoch: usize) {
        fs::create_dir_all(dir.join(format!("checkpoint-{epoch:06}"))).unwrap();
    }

    #[test]
    fn meta_json_written_by_create() {
        let dir = temp_dir("meta");
        TrainingRun::create(&dir, &meta("my_dataset"), &sample_config(), None, false).unwrap();

        let loaded = TrainingRun::open(&dir).unwrap();
        cleanup(&dir);

        assert_eq!(loaded.meta().dataset, "my_dataset");
    }

    #[test]
    fn config_json_roundtrips_the_blueprint() {
        let dir = temp_dir("config");
        let config = sample_config();
        TrainingRun::create(&dir, &meta("ds"), &config, None, false).unwrap();

        let loaded = TrainingRun::open(&dir).unwrap();
        cleanup(&dir);

        assert_eq!(loaded.config(), &config);
    }

    #[test]
    fn preprocessor_json_roundtrips_the_scaler() {
        use crate::data::scalers::{MinMaxScaler, ScalerMethod};
        use ndarray::array;

        let dir = temp_dir("meta_scaler");
        let scaler = ScalerMethod::MinMax(
            MinMaxScaler::default().fit(array![[0.0, 0.0], [1.0, 1.0]].view()),
        );
        let scaler: ScalerRecord = scaler.into();
        TrainingRun::create(&dir, &meta("ds"), &sample_config(), Some(&scaler), false).unwrap();

        let loaded = TrainingRun::open(&dir).unwrap();
        cleanup(&dir);

        assert_eq!(loaded.scaler(), Some(&scaler));
    }

    #[test]
    fn no_preprocessor_file_when_scaler_is_absent() {
        let dir = temp_dir("no_scaler");
        TrainingRun::create(&dir, &meta("ds"), &sample_config(), None, false).unwrap();
        let exists = dir.join("preprocessor.json").exists();
        cleanup(&dir);

        assert!(!exists);
    }

    #[test]
    fn create_errors_if_checkpoints_exist_without_overwrite() {
        let dir = temp_dir("no_overwrite");
        TrainingRun::create(&dir, &meta("ds"), &sample_config(), None, false).unwrap();
        make_checkpoint(&dir, 0);

        let result = TrainingRun::create(&dir, &meta("ds"), &sample_config(), None, false);
        cleanup(&dir);

        let err = result.err().unwrap();
        assert_eq!(err.kind(), AlreadyExists);
        assert!(err.to_string().contains("already exists"));
    }

    #[test]
    fn create_with_overwrite_purges_previous_checkpoints() {
        let dir = temp_dir("overwrite");
        TrainingRun::create(&dir, &meta("ds"), &sample_config(), None, false).unwrap();
        for i in 0..3 {
            make_checkpoint(&dir, i * 10);
        }

        TrainingRun::create(&dir, &meta("ds"), &sample_config(), None, true).unwrap();
        make_checkpoint(&dir, 0);

        let archive = CheckpointArchive::load(&dir, sample_network()).unwrap();
        cleanup(&dir);

        assert_eq!(archive.len(), 1, "stale checkpoints were not purged");
    }

    #[test]
    fn trim_after_removes_checkpoints_past_from_epoch() {
        let dir = temp_dir("resume_trim");
        TrainingRun::create(&dir, &meta("ds"), &sample_config(), None, false).unwrap();
        for i in 0..5 {
            make_checkpoint(&dir, i * 10);
        }

        let trimmed = TrainingRun::open(&dir).unwrap().trim_after(20).unwrap();

        let archive = CheckpointArchive::load(&dir, sample_network()).unwrap();
        cleanup(&dir);

        assert_eq!(archive.len(), 3); // epochs 0, 10, 20 kept; 30, 40 removed
        assert_eq!(trimmed, 2);
    }

    #[test]
    fn trim_after_keeps_all_when_from_epoch_is_last() {
        let dir = temp_dir("resume_keep_all");
        TrainingRun::create(&dir, &meta("ds"), &sample_config(), None, false).unwrap();
        for i in 0..3 {
            make_checkpoint(&dir, i * 10);
        }

        let trimmed = TrainingRun::open(&dir).unwrap().trim_after(20).unwrap();

        let archive = CheckpointArchive::load(&dir, sample_network()).unwrap();
        cleanup(&dir);

        assert_eq!(archive.len(), 3);
        assert_eq!(trimmed, 0);
    }

    #[test]
    fn trim_after_lets_a_checkpoint_be_repopulated() {
        let dir = temp_dir("resume_write");
        TrainingRun::create(&dir, &meta("ds"), &sample_config(), None, false).unwrap();
        for i in 0..3 {
            make_checkpoint(&dir, i * 10); // 0, 10, 20
        }

        let run = TrainingRun::open(&dir).unwrap();
        run.trim_after(10).unwrap(); // drops 20
        make_checkpoint(&dir, 20); // rewritten

        let archive = CheckpointArchive::load(&dir, sample_network()).unwrap();
        cleanup(&dir);

        assert_eq!(archive.len(), 3); // 0, 10, 20
    }

    #[test]
    fn create_rejects_path_traversal() {
        let result = TrainingRun::create(
            "../../nrn_traversal_test",
            &meta("x"),
            &sample_config(),
            None,
            false,
        );
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
