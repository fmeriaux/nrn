use crate::callbacks::TrainingCallback;
use crate::evaluation::{Evaluation, EvaluationSet};
use crate::io::bytes::secure_read;
use crate::io::json;
use crate::io::path::PathExt;
use crate::io::tensors;
use crate::model::NeuralNetwork;
use crate::training_history::TrainingHistory;
use safetensors::SafeTensors;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::ErrorKind::InvalidData;
use std::io::{Error, Result};
use std::path::{Path, PathBuf};

/// Top-level metadata for a training history directory.
/// Written once by [`SnapshotRecorder::create`] into `meta.json`.
#[derive(Serialize, Deserialize)]
pub struct TrainingMeta {
    pub dataset: String,
}

impl TrainingMeta {
    /// Loads the metadata from `meta.json` inside `dir`.
    pub fn load<P: AsRef<Path>>(dir: P) -> Result<Self> {
        let dir = Path::combine_safe_with_cwd(dir)?;
        json::load(dir.join("meta"))
    }

    fn save<P: AsRef<Path>>(&self, dir: P) -> Result<PathBuf> {
        let dir = Path::combine_safe_with_cwd(dir)?;
        json::save(self, dir.join("meta"))
    }
}

#[derive(Serialize, Deserialize)]
struct SnapshotEvals {
    train: MetricPair,
    #[serde(skip_serializing_if = "Option::is_none")]
    validation: Option<MetricPair>,
    test: MetricPair,
}

#[derive(Serialize, Deserialize)]
struct MetricPair {
    loss: f32,
    accuracy: f32,
}

impl From<Evaluation> for MetricPair {
    fn from(eval: Evaluation) -> Self {
        MetricPair {
            loss: eval.loss,
            accuracy: eval.accuracy,
        }
    }
}

impl From<MetricPair> for Evaluation {
    fn from(pair: MetricPair) -> Self {
        Evaluation {
            loss: pair.loss,
            accuracy: pair.accuracy,
        }
    }
}

impl From<&EvaluationSet> for SnapshotEvals {
    fn from(eval: &EvaluationSet) -> Self {
        SnapshotEvals {
            train: eval.train.into(),
            validation: eval.validation.map(Into::into),
            test: eval.test.into(),
        }
    }
}

impl From<SnapshotEvals> for EvaluationSet {
    fn from(evals: SnapshotEvals) -> Self {
        EvaluationSet {
            train: evals.train.into(),
            validation: evals.validation.map(Into::into),
            test: evals.test.into(),
        }
    }
}

/// A reference to a snapshot subdirectory, named `snapshot-{epoch:06}`.
struct SnapshotRef {
    epoch: usize,
    dir: PathBuf,
}

/// Scans `dir` for `snapshot-*` subdirectories, sorted by their numeric epoch
/// (not lexically, so 10+ snapshots sort correctly). Reads no other files.
fn scan_snapshots(dir: &Path) -> Result<Vec<SnapshotRef>> {
    let mut snapshots: Vec<SnapshotRef> = fs::read_dir(dir)?
        .filter_map(std::result::Result::ok)
        .filter_map(|entry| {
            let path = entry.path();
            let name = path.file_name()?.to_str()?;
            if path.is_dir() {
                let epoch = name.strip_prefix("snapshot-")?.parse::<usize>().ok()?;
                Some(SnapshotRef { epoch, dir: path })
            } else {
                None
            }
        })
        .collect();

    snapshots.sort_by_key(|s| s.epoch);
    Ok(snapshots)
}

/// Writes model snapshots to a directory.
///
/// Each call to [`write`](SnapshotRecorder::write) creates a subdirectory
/// `snapshot-{epoch:06}/` containing `model.safetensors` and `evaluations.json`.
/// Implements [`TrainingCallback`]: [`on_evaluate`](TrainingCallback::on_evaluate)
/// writes a snapshot.
#[derive(Debug)]
pub struct SnapshotRecorder {
    dir: PathBuf,
}

impl SnapshotRecorder {
    /// Creates a fresh recorder at `path`, writing `meta.json` with the dataset name.
    ///
    /// Returns an error if `snapshot-*` subdirectories already exist and
    /// `overwrite` is `false`; otherwise they are removed.
    pub fn create<P: AsRef<Path>>(path: P, dataset: &str, overwrite: bool) -> Result<Self> {
        let dir = Path::combine_safe_with_cwd(path)?;
        fs::create_dir_all(&dir)?;

        let existing = scan_snapshots(&dir)?;
        if !existing.is_empty() {
            if !overwrite {
                return Err(Error::new(
                    InvalidData,
                    format!(
                        "training history already exists at {}; \
                         use --overwrite to replace it",
                        dir.display()
                    ),
                ));
            }
            for snapshot in existing {
                fs::remove_dir_all(&snapshot.dir)?;
            }
        }

        TrainingMeta {
            dataset: dataset.to_string(),
        }
        .save(&dir)?;

        Ok(SnapshotRecorder { dir })
    }

    /// Resumes recording into an existing snapshot directory. Snapshots whose
    /// epoch is greater than `from_epoch` are removed (rewinding the trajectory).
    pub fn resume<P: AsRef<Path>>(path: P, from_epoch: usize) -> Result<Self> {
        let dir = Path::combine_safe_with_cwd(path)?;
        fs::create_dir_all(&dir)?;

        let to_remove: Vec<SnapshotRef> = scan_snapshots(&dir)?
            .into_iter()
            .filter(|snapshot| snapshot.epoch > from_epoch)
            .collect();

        if !to_remove.is_empty() {
            eprintln!(
                "Removing {} snapshot(s) after epoch {from_epoch}",
                to_remove.len()
            );
            for snapshot in to_remove {
                fs::remove_dir_all(&snapshot.dir)?;
            }
        }

        Ok(SnapshotRecorder { dir })
    }

    /// Writes `snapshot-{epoch:06}/` containing the model weights and evaluations.
    pub fn write(
        &self,
        model: &NeuralNetwork,
        evaluation: &EvaluationSet,
        epoch: usize,
    ) -> Result<()> {
        let snapshot_dir = self.dir.join(format!("snapshot-{epoch:06}"));
        fs::create_dir_all(&snapshot_dir)?;

        let mut entries = Vec::new();
        let mut metadata = HashMap::new();
        model.collect_tensors(&mut entries, &mut metadata);
        tensors::save(snapshot_dir.join("model"), entries, metadata)?;

        json::save(
            &SnapshotEvals::from(evaluation),
            snapshot_dir.join("evaluations"),
        )?;

        Ok(())
    }
}

impl TrainingCallback for SnapshotRecorder {
    fn on_evaluate(
        &mut self,
        model: &NeuralNetwork,
        eval: &EvaluationSet,
        epoch: usize,
    ) -> Result<()> {
        self.write(model, eval, epoch)
    }
}

/// Lazy read access to a directory of snapshots written by [`SnapshotRecorder`].
///
/// [`SnapshotArchive::load`] only scans directory names — no files are read until
/// [`model_at`](SnapshotArchive::model_at) or [`history`](SnapshotArchive::history)
/// is called.
pub struct SnapshotArchive {
    snapshots: Vec<SnapshotRef>,
}

impl SnapshotArchive {
    /// Scans `dir` for `snapshot-*` subdirectories, sorted by epoch.
    pub fn load<P: AsRef<Path>>(dir: P) -> Result<Self> {
        let dir = Path::combine_safe_with_cwd(dir)?;
        Ok(SnapshotArchive {
            snapshots: scan_snapshots(&dir)?,
        })
    }

    /// Returns the number of snapshots found.
    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    /// Returns true if no snapshots were found.
    pub fn is_empty(&self) -> bool {
        self.snapshots.is_empty()
    }

    /// Returns the absolute epoch number of the snapshot at `index`, from the
    /// scanned directory name (zero I/O).
    pub fn epoch_at(&self, index: usize) -> Option<usize> {
        self.snapshots.get(index).map(|s| s.epoch)
    }

    /// Loads the model at position `index` from its snapshot directory.
    pub fn model_at(&self, index: usize) -> Result<NeuralNetwork> {
        let snapshot = self.snapshots.get(index).ok_or_else(|| {
            Error::new(
                InvalidData,
                format!(
                    "snapshot index {index} out of range (archive has {} snapshots)",
                    self.snapshots.len()
                ),
            )
        })?;

        let bytes = secure_read(snapshot.dir.join("model.safetensors"))?;
        let st =
            SafeTensors::deserialize(&bytes).map_err(|e| Error::new(InvalidData, e.to_string()))?;
        let metadata = tensors::read_metadata(&bytes)?;
        NeuralNetwork::from_tensors(&st, &metadata)
    }

    /// Reads all `evaluations.json` files into a pure [`TrainingHistory`].
    pub fn history(&self) -> Result<TrainingHistory> {
        let mut evaluations = Vec::with_capacity(self.snapshots.len());
        let mut snapshot_epochs = Vec::with_capacity(self.snapshots.len());

        for snapshot in &self.snapshots {
            let evals: SnapshotEvals = json::load(snapshot.dir.join("evaluations"))?;
            evaluations.push(evals.into());
            snapshot_epochs.push(snapshot.epoch);
        }

        Ok(TrainingHistory::new(evaluations, snapshot_epochs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::RELU;
    use crate::model::NeuronLayerSpec;
    use ndarray::Array2;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = PathBuf::from(format!(
            "target/nrn_snapshot_{}_{}",
            tag,
            std::process::id()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn cleanup(dir: &Path) {
        let _ = fs::remove_dir_all(dir);
    }

    fn sample_model() -> NeuralNetwork {
        let specs = NeuronLayerSpec::network_for(vec![3], &*RELU, 2);
        NeuralNetwork::initialization(2, &specs)
    }

    fn make_eval(loss: f32, with_validation: bool) -> EvaluationSet {
        EvaluationSet {
            train: Evaluation {
                loss,
                accuracy: 0.5,
            },
            validation: with_validation.then_some(Evaluation {
                loss: loss + 1.0,
                accuracy: 0.1,
            }),
            test: Evaluation {
                loss: loss + 100.0,
                accuracy: 0.9,
            },
        }
    }

    #[test]
    fn meta_json_written_by_create() {
        let dir = temp_dir("meta");
        SnapshotRecorder::create(&dir, "my_dataset", false).unwrap();

        let meta = TrainingMeta::load(&dir).unwrap();
        cleanup(&dir);

        assert_eq!(meta.dataset, "my_dataset");
    }

    #[test]
    fn create_errors_if_snapshots_exist_without_overwrite() {
        let dir = temp_dir("no_overwrite");
        let recorder = SnapshotRecorder::create(&dir, "ds", false).unwrap();
        recorder
            .write(&sample_model(), &make_eval(0.0, false), 0)
            .unwrap();

        let result = SnapshotRecorder::create(&dir, "ds", false);
        cleanup(&dir);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("already exists"));
    }

    #[test]
    fn create_with_overwrite_purges_previous_snapshots() {
        let dir = temp_dir("overwrite");
        let recorder = SnapshotRecorder::create(&dir, "ds", false).unwrap();
        for i in 0..3 {
            recorder
                .write(&sample_model(), &make_eval(i as f32, false), i * 10)
                .unwrap();
        }

        let recorder = SnapshotRecorder::create(&dir, "ds", true).unwrap();
        recorder
            .write(&sample_model(), &make_eval(0.0, false), 0)
            .unwrap();

        let archive = SnapshotArchive::load(&dir).unwrap();
        cleanup(&dir);

        assert_eq!(archive.len(), 1, "stale snapshots were not purged");
    }

    #[test]
    fn write_names_snapshot_dir_by_epoch() {
        let dir = temp_dir("write_epoch");
        let recorder = SnapshotRecorder::create(&dir, "ds", false).unwrap();
        recorder
            .write(&sample_model(), &make_eval(0.0, false), 37)
            .unwrap();

        let exists = dir.join("snapshot-000037").is_dir();
        cleanup(&dir);

        assert!(exists);
    }

    #[test]
    fn roundtrip_with_validation() {
        let dir = temp_dir("roundtrip_val");
        let recorder = SnapshotRecorder::create(&dir, "ds", false).unwrap();
        for i in 0..3 {
            recorder
                .write(&sample_model(), &make_eval(i as f32, true), i * 10)
                .unwrap();
        }

        let history = SnapshotArchive::load(&dir).unwrap().history().unwrap();
        cleanup(&dir);

        assert_eq!(history.len(), 3);
        for (i, eval) in history.evaluations.iter().enumerate() {
            assert_eq!(eval.train.loss, i as f32);
            assert!(eval.validation.is_some());
        }
    }

    #[test]
    fn roundtrip_without_validation() {
        let dir = temp_dir("roundtrip_noval");
        let recorder = SnapshotRecorder::create(&dir, "ds", false).unwrap();
        for i in 0..3 {
            recorder
                .write(&sample_model(), &make_eval(i as f32, false), i * 10)
                .unwrap();
        }

        let history = SnapshotArchive::load(&dir).unwrap().history().unwrap();
        cleanup(&dir);

        assert_eq!(history.len(), 3);
        for eval in &history.evaluations {
            assert!(eval.validation.is_none());
        }
    }

    #[test]
    fn predictions_survive_roundtrip() {
        let dir = temp_dir("predictions");
        let model = sample_model();
        let inputs = Array2::from_shape_fn((2, 4), |(i, j)| (i + j) as f32 * 0.3);

        let recorder = SnapshotRecorder::create(&dir, "ds", false).unwrap();
        recorder.write(&model, &make_eval(0.0, false), 0).unwrap();

        let archive = SnapshotArchive::load(&dir).unwrap();
        let loaded = archive.model_at(0).unwrap();
        cleanup(&dir);

        assert_eq!(model.predict(inputs.view()), loaded.predict(inputs.view()));
    }

    #[test]
    fn numeric_sort_beats_lexical() {
        let dir = temp_dir("sort");
        let model = sample_model();
        let recorder = SnapshotRecorder::create(&dir, "ds", false).unwrap();
        for i in 0..12 {
            recorder
                .write(&model, &make_eval(i as f32, false), i)
                .unwrap();
        }

        let archive = SnapshotArchive::load(&dir).unwrap();
        cleanup(&dir);

        assert_eq!(archive.len(), 12);
        for i in 0..12 {
            assert_eq!(archive.epoch_at(i), Some(i));
        }
    }

    #[test]
    fn resume_trims_snapshots_after_from_epoch() {
        let dir = temp_dir("resume_trim");
        let recorder = SnapshotRecorder::create(&dir, "ds", false).unwrap();
        for i in 0..5 {
            recorder
                .write(&sample_model(), &make_eval(i as f32, false), i * 10)
                .unwrap();
        }

        SnapshotRecorder::resume(&dir, 20).unwrap();

        let archive = SnapshotArchive::load(&dir).unwrap();
        cleanup(&dir);

        assert_eq!(archive.len(), 3); // epochs 0, 10, 20 kept; 30, 40 removed
    }

    #[test]
    fn resume_keeps_all_when_from_epoch_is_last() {
        let dir = temp_dir("resume_keep_all");
        let recorder = SnapshotRecorder::create(&dir, "ds", false).unwrap();
        for i in 0..3 {
            recorder
                .write(&sample_model(), &make_eval(i as f32, false), i * 10)
                .unwrap();
        }

        SnapshotRecorder::resume(&dir, 20).unwrap();

        let archive = SnapshotArchive::load(&dir).unwrap();
        cleanup(&dir);

        assert_eq!(archive.len(), 3);
    }

    #[test]
    fn resume_continues_writing_after_resume_point() {
        let dir = temp_dir("resume_write");
        let recorder = SnapshotRecorder::create(&dir, "ds", false).unwrap();
        for i in 0..3 {
            recorder
                .write(&sample_model(), &make_eval(i as f32, false), i * 10)
                .unwrap(); // 0, 10, 20
        }

        let recorder = SnapshotRecorder::resume(&dir, 10).unwrap();
        recorder
            .write(&sample_model(), &make_eval(99.0, false), 20)
            .unwrap();

        let archive = SnapshotArchive::load(&dir).unwrap();
        cleanup(&dir);

        assert_eq!(archive.len(), 3); // 0, 10, 20 (rewritten)
    }

    #[test]
    fn empty_dir_archive_is_empty() {
        let dir = temp_dir("empty");
        let archive = SnapshotArchive::load(&dir).unwrap();
        cleanup(&dir);

        assert!(archive.is_empty());
        assert_eq!(archive.len(), 0);
    }

    #[test]
    fn load_ignores_non_directory_snapshot_file() {
        let dir = temp_dir("non_dir_snap");
        fs::write(dir.join("snapshot-000000"), b"not a dir").unwrap();
        let archive = SnapshotArchive::load(&dir).unwrap();
        cleanup(&dir);

        assert!(archive.is_empty());
    }

    #[test]
    fn load_ignores_snapshot_dir_with_non_numeric_suffix() {
        let dir = temp_dir("non_numeric_snap");
        fs::create_dir_all(dir.join("snapshot-abc")).unwrap();
        let archive = SnapshotArchive::load(&dir).unwrap();
        cleanup(&dir);

        assert!(archive.is_empty());
    }

    #[test]
    fn model_at_out_of_range_gives_range_error() {
        let dir = temp_dir("model_at_oob");
        let recorder = SnapshotRecorder::create(&dir, "ds", false).unwrap();
        recorder
            .write(&sample_model(), &make_eval(0.0, false), 0)
            .unwrap();

        let archive = SnapshotArchive::load(&dir).unwrap();
        let msg = archive.model_at(99).err().unwrap().to_string();
        cleanup(&dir);

        assert!(msg.contains("out of range"), "got: {msg}");
    }

    #[test]
    fn model_at_missing_model_file_fails() {
        let dir = temp_dir("model_at_missing");
        let recorder = SnapshotRecorder::create(&dir, "ds", false).unwrap();
        recorder
            .write(&sample_model(), &make_eval(0.0, false), 0)
            .unwrap();
        fs::remove_file(dir.join("snapshot-000000").join("model.safetensors")).unwrap();

        let archive = SnapshotArchive::load(&dir).unwrap();
        let result = archive.model_at(0);
        cleanup(&dir);

        assert!(result.is_err());
    }

    #[test]
    fn model_at_corrupt_model_file_fails() {
        let dir = temp_dir("model_at_corrupt");
        let recorder = SnapshotRecorder::create(&dir, "ds", false).unwrap();
        recorder
            .write(&sample_model(), &make_eval(0.0, false), 0)
            .unwrap();
        fs::write(
            dir.join("snapshot-000000").join("model.safetensors"),
            b"garbage",
        )
        .unwrap();

        let archive = SnapshotArchive::load(&dir).unwrap();
        let result = archive.model_at(0);
        cleanup(&dir);

        assert!(result.is_err());
    }

    #[test]
    fn history_corrupted_evaluations_json_fails() {
        let dir = temp_dir("corrupt_evals");
        let recorder = SnapshotRecorder::create(&dir, "ds", false).unwrap();
        recorder
            .write(&sample_model(), &make_eval(0.0, false), 0)
            .unwrap();
        fs::write(
            dir.join("snapshot-000000").join("evaluations.json"),
            b"not valid json",
        )
        .unwrap();

        let archive = SnapshotArchive::load(&dir).unwrap();
        let result = archive.history();
        cleanup(&dir);

        assert!(result.is_err());
    }

    #[test]
    fn on_evaluate_writes_a_snapshot() {
        let dir = temp_dir("on_evaluate");
        let mut recorder = SnapshotRecorder::create(&dir, "ds", false).unwrap();

        recorder
            .on_evaluate(&sample_model(), &make_eval(0.0, false), 5)
            .unwrap();

        let archive = SnapshotArchive::load(&dir).unwrap();
        cleanup(&dir);

        assert_eq!(archive.len(), 1);
        assert_eq!(archive.epoch_at(0), Some(5));
    }

    #[test]
    fn write_fails_when_evaluations_path_is_a_directory() {
        let dir = temp_dir("write_evals_dir_conflict");
        let recorder = SnapshotRecorder::create(&dir, "ds", false).unwrap();

        // Pre-create "evaluations.json" as a directory so json::save's fs::write fails.
        fs::create_dir_all(dir.join("snapshot-000000").join("evaluations.json")).unwrap();

        let result = recorder.write(&sample_model(), &make_eval(0.0, false), 0);
        cleanup(&dir);

        assert!(result.is_err());
    }

    #[test]
    fn create_rejects_path_traversal() {
        let result = SnapshotRecorder::create("../../nrn_traversal_test", "x", false);
        assert!(result.is_err());
    }

    #[test]
    fn resume_rejects_path_traversal() {
        let result = SnapshotRecorder::resume("../../nrn_traversal_test", 0);
        assert!(result.is_err());
    }

    #[test]
    fn load_rejects_path_traversal() {
        let result = SnapshotArchive::load("../../nrn_traversal_test");
        assert!(result.is_err());
    }
}
