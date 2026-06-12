use crate::evaluation::{Evaluation, EvaluationSet};
use crate::evaluation_history::{EpochEvaluation, EvaluationHistory};
use crate::io::bytes::secure_read;
use crate::io::json;
use crate::io::path::PathExt;
use crate::io::tensors;
use crate::model::NeuralNetwork;
use crate::training::TrainingCallback;
use safetensors::SafeTensors;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::ErrorKind::{AlreadyExists, InvalidData};
use std::io::{Error, Result};
use std::path::{Path, PathBuf};
use std::result::Result as StdResult;

/// Top-level metadata for a training history directory.
/// Written once by [`CheckpointRecorder::create`] into `meta.json`.
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
struct CheckpointEvaluationSet {
    train: CheckpointEvaluation,
    #[serde(skip_serializing_if = "Option::is_none")]
    validation: Option<CheckpointEvaluation>,
    test: CheckpointEvaluation,
}

#[derive(Serialize, Deserialize)]
struct CheckpointEvaluation {
    loss: f32,
    accuracy: f32,
}

impl From<Evaluation> for CheckpointEvaluation {
    fn from(eval: Evaluation) -> Self {
        CheckpointEvaluation {
            loss: eval.loss,
            accuracy: eval.accuracy,
        }
    }
}

impl From<CheckpointEvaluation> for Evaluation {
    fn from(pair: CheckpointEvaluation) -> Self {
        Evaluation {
            loss: pair.loss,
            accuracy: pair.accuracy,
        }
    }
}

impl From<&EvaluationSet> for CheckpointEvaluationSet {
    fn from(eval: &EvaluationSet) -> Self {
        CheckpointEvaluationSet {
            train: eval.train.into(),
            validation: eval.validation.map(Into::into),
            test: eval.test.into(),
        }
    }
}

impl From<CheckpointEvaluationSet> for EvaluationSet {
    fn from(evals: CheckpointEvaluationSet) -> Self {
        EvaluationSet {
            train: evals.train.into(),
            validation: evals.validation.map(Into::into),
            test: evals.test.into(),
        }
    }
}

/// A reference to a checkpoint subdirectory, named `checkpoint-{epoch:06}`.
struct CheckpointRef {
    epoch: usize,
    dir: PathBuf,
}

/// Scans `dir` for `checkpoint-*` subdirectories, sorted by their numeric epoch
/// (not lexically, so 10+ checkpoints sort correctly). Reads no other files.
fn scan_checkpoints(dir: &Path) -> Result<Vec<CheckpointRef>> {
    let mut checkpoints: Vec<CheckpointRef> = fs::read_dir(dir)?
        .filter_map(StdResult::ok)
        .filter_map(|entry| {
            let path = entry.path();
            let name = path.file_name()?.to_str()?;
            if path.is_dir() {
                let epoch = name.strip_prefix("checkpoint-")?.parse::<usize>().ok()?;
                Some(CheckpointRef { epoch, dir: path })
            } else {
                None
            }
        })
        .collect();

    checkpoints.sort_by_key(|s| s.epoch);
    Ok(checkpoints)
}

/// Writes model checkpoints to a directory.
///
/// Each call to [`write`](CheckpointRecorder::write) creates a subdirectory
/// `checkpoint-{epoch:06}/` containing `model.safetensors` and `evaluations.json`.
/// Implements [`TrainingCallback`]: [`on_evaluate`](TrainingCallback::on_evaluate)
/// writes a checkpoint.
#[derive(Debug)]
pub struct CheckpointRecorder {
    dir: PathBuf,
}

impl CheckpointRecorder {
    /// Creates a fresh recorder at `path`, writing `meta.json` with the dataset name.
    ///
    /// Returns an error if `checkpoint-*` subdirectories already exist and
    /// `overwrite` is `false`; otherwise they are removed.
    pub fn create<P: AsRef<Path>>(path: P, dataset: &str, overwrite: bool) -> Result<Self> {
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

        TrainingMeta {
            dataset: dataset.to_string(),
        }
        .save(&dir)?;

        Ok(CheckpointRecorder { dir })
    }

    /// Resumes recording into an existing checkpoint directory. Checkpoints whose
    /// epoch is greater than `from_epoch` are removed (rewinding the trajectory).
    /// Returns the recorder along with the number of checkpoints removed.
    pub fn resume<P: AsRef<Path>>(path: P, from_epoch: usize) -> Result<(Self, usize)> {
        let dir = Path::combine_safe_with_cwd(path)?;
        fs::create_dir_all(&dir)?;

        let to_remove: Vec<CheckpointRef> = scan_checkpoints(&dir)?
            .into_iter()
            .filter(|checkpoint| checkpoint.epoch > from_epoch)
            .collect();

        let trimmed = to_remove.len();
        for checkpoint in to_remove {
            fs::remove_dir_all(&checkpoint.dir)?;
        }

        Ok((CheckpointRecorder { dir }, trimmed))
    }

    /// Writes `checkpoint-{epoch:06}/` containing the model weights and evaluations.
    pub fn write(
        &self,
        model: &NeuralNetwork,
        evaluation: &EvaluationSet,
        epoch: usize,
    ) -> Result<()> {
        let checkpoint_dir = self.dir.join(format!("checkpoint-{epoch:06}"));
        fs::create_dir_all(&checkpoint_dir)?;

        let mut entries = Vec::new();
        let mut metadata = HashMap::new();
        model.collect_tensors(&mut entries, &mut metadata);
        tensors::save(checkpoint_dir.join("model"), entries, metadata)?;

        json::save(
            &CheckpointEvaluationSet::from(evaluation),
            checkpoint_dir.join("evaluations"),
        )?;

        Ok(())
    }
}

impl TrainingCallback for CheckpointRecorder {
    fn on_evaluate(
        &mut self,
        model: &NeuralNetwork,
        eval: &EvaluationSet,
        epoch: usize,
    ) -> Result<()> {
        self.write(model, eval, epoch)
    }
}

/// Lazy read access to a directory of checkpoints written by [`CheckpointRecorder`].
///
/// [`CheckpointArchive::load`] only scans directory names — no files are read until
/// [`model_at`](CheckpointArchive::model_at) or
/// [`evaluation_history`](CheckpointArchive::evaluation_history) is called.
pub struct CheckpointArchive {
    entries: Vec<CheckpointRef>,
}

impl CheckpointArchive {
    /// Scans `dir` for `checkpoint-*` subdirectories, sorted by epoch.
    pub fn load<P: AsRef<Path>>(dir: P) -> Result<Self> {
        let dir = Path::combine_safe_with_cwd(dir)?;
        Ok(CheckpointArchive {
            entries: scan_checkpoints(&dir)?,
        })
    }

    /// Returns the number of checkpoints found.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns true if no checkpoints were found.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns the absolute epoch number of the checkpoint at `index`, from the
    /// scanned directory name (zero I/O).
    pub fn epoch_at(&self, index: usize) -> Option<usize> {
        self.entries.get(index).map(|s| s.epoch)
    }

    /// Loads the model at position `index` from its checkpoint directory.
    pub fn model_at(&self, index: usize) -> Result<NeuralNetwork> {
        let entry = self.entries.get(index).ok_or_else(|| {
            Error::new(
                InvalidData,
                format!(
                    "checkpoint index {index} out of range (archive has {} checkpoints)",
                    self.entries.len()
                ),
            )
        })?;

        let bytes = secure_read(entry.dir.join("model.safetensors"))?;
        let st =
            SafeTensors::deserialize(&bytes).map_err(|e| Error::new(InvalidData, e.to_string()))?;
        let metadata = tensors::read_metadata(&bytes)?;
        NeuralNetwork::from_tensors(&st, &metadata)
    }

    /// Reads all `evaluations.json` files into a pure [`EvaluationHistory`].
    pub fn evaluation_history(&self) -> Result<EvaluationHistory> {
        let mut history = Vec::with_capacity(self.entries.len());

        for entry in &self.entries {
            let evals: CheckpointEvaluationSet = json::load(entry.dir.join("evaluations"))?;
            history.push(EpochEvaluation {
                epoch: entry.epoch,
                evaluation: evals.into(),
            });
        }

        Ok(EvaluationHistory::new(history))
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
            "target/nrn_checkpoint_{}_{}",
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
        CheckpointRecorder::create(&dir, "my_dataset", false).unwrap();

        let meta = TrainingMeta::load(&dir).unwrap();
        cleanup(&dir);

        assert_eq!(meta.dataset, "my_dataset");
    }

    #[test]
    fn create_errors_if_checkpoints_exist_without_overwrite() {
        let dir = temp_dir("no_overwrite");
        let recorder = CheckpointRecorder::create(&dir, "ds", false).unwrap();
        recorder
            .write(&sample_model(), &make_eval(0.0, false), 0)
            .unwrap();

        let result = CheckpointRecorder::create(&dir, "ds", false);
        cleanup(&dir);

        let err = result.unwrap_err();
        assert_eq!(err.kind(), AlreadyExists);
        assert!(err.to_string().contains("already exists"));
    }

    #[test]
    fn create_with_overwrite_purges_previous_checkpoints() {
        let dir = temp_dir("overwrite");
        let recorder = CheckpointRecorder::create(&dir, "ds", false).unwrap();
        for i in 0..3 {
            recorder
                .write(&sample_model(), &make_eval(i as f32, false), i * 10)
                .unwrap();
        }

        let recorder = CheckpointRecorder::create(&dir, "ds", true).unwrap();
        recorder
            .write(&sample_model(), &make_eval(0.0, false), 0)
            .unwrap();

        let archive = CheckpointArchive::load(&dir).unwrap();
        cleanup(&dir);

        assert_eq!(archive.len(), 1, "stale checkpoints were not purged");
    }

    #[test]
    fn write_names_checkpoint_dir_by_epoch() {
        let dir = temp_dir("write_epoch");
        let recorder = CheckpointRecorder::create(&dir, "ds", false).unwrap();
        recorder
            .write(&sample_model(), &make_eval(0.0, false), 37)
            .unwrap();

        let exists = dir.join("checkpoint-000037").is_dir();
        cleanup(&dir);

        assert!(exists);
    }

    #[test]
    fn roundtrip_with_validation() {
        let dir = temp_dir("roundtrip_val");
        let recorder = CheckpointRecorder::create(&dir, "ds", false).unwrap();
        for i in 0..3 {
            recorder
                .write(&sample_model(), &make_eval(i as f32, true), i * 10)
                .unwrap();
        }

        let history = CheckpointArchive::load(&dir)
            .unwrap()
            .evaluation_history()
            .unwrap();
        cleanup(&dir);

        assert_eq!(history.len(), 3);
        for (i, loss) in history.train_losses().iter().enumerate() {
            assert_eq!(*loss, i as f32);
        }
        assert!(!history.validation_losses().is_empty());
    }

    #[test]
    fn roundtrip_without_validation() {
        let dir = temp_dir("roundtrip_noval");
        let recorder = CheckpointRecorder::create(&dir, "ds", false).unwrap();
        for i in 0..3 {
            recorder
                .write(&sample_model(), &make_eval(i as f32, false), i * 10)
                .unwrap();
        }

        let history = CheckpointArchive::load(&dir)
            .unwrap()
            .evaluation_history()
            .unwrap();
        cleanup(&dir);

        assert_eq!(history.len(), 3);
        assert!(history.validation_losses().is_empty());
    }

    #[test]
    fn predictions_survive_roundtrip() {
        let dir = temp_dir("predictions");
        let model = sample_model();
        let inputs = Array2::from_shape_fn((2, 4), |(i, j)| (i + j) as f32 * 0.3);

        let recorder = CheckpointRecorder::create(&dir, "ds", false).unwrap();
        recorder.write(&model, &make_eval(0.0, false), 0).unwrap();

        let archive = CheckpointArchive::load(&dir).unwrap();
        let loaded = archive.model_at(0).unwrap();
        cleanup(&dir);

        assert_eq!(model.predict(inputs.view()), loaded.predict(inputs.view()));
    }

    #[test]
    fn numeric_sort_beats_lexical() {
        let dir = temp_dir("sort");
        let model = sample_model();
        let recorder = CheckpointRecorder::create(&dir, "ds", false).unwrap();
        for i in 0..12 {
            recorder
                .write(&model, &make_eval(i as f32, false), i)
                .unwrap();
        }

        let archive = CheckpointArchive::load(&dir).unwrap();
        cleanup(&dir);

        assert_eq!(archive.len(), 12);
        for i in 0..12 {
            assert_eq!(archive.epoch_at(i), Some(i));
        }
    }

    #[test]
    fn resume_trims_checkpoints_after_from_epoch() {
        let dir = temp_dir("resume_trim");
        let recorder = CheckpointRecorder::create(&dir, "ds", false).unwrap();
        for i in 0..5 {
            recorder
                .write(&sample_model(), &make_eval(i as f32, false), i * 10)
                .unwrap();
        }

        let (_, trimmed) = CheckpointRecorder::resume(&dir, 20).unwrap();

        let archive = CheckpointArchive::load(&dir).unwrap();
        cleanup(&dir);

        assert_eq!(archive.len(), 3); // epochs 0, 10, 20 kept; 30, 40 removed
        assert_eq!(trimmed, 2);
    }

    #[test]
    fn resume_keeps_all_when_from_epoch_is_last() {
        let dir = temp_dir("resume_keep_all");
        let recorder = CheckpointRecorder::create(&dir, "ds", false).unwrap();
        for i in 0..3 {
            recorder
                .write(&sample_model(), &make_eval(i as f32, false), i * 10)
                .unwrap();
        }

        let (_, trimmed) = CheckpointRecorder::resume(&dir, 20).unwrap();

        let archive = CheckpointArchive::load(&dir).unwrap();
        cleanup(&dir);

        assert_eq!(archive.len(), 3);
        assert_eq!(trimmed, 0);
    }

    #[test]
    fn resume_continues_writing_after_resume_point() {
        let dir = temp_dir("resume_write");
        let recorder = CheckpointRecorder::create(&dir, "ds", false).unwrap();
        for i in 0..3 {
            recorder
                .write(&sample_model(), &make_eval(i as f32, false), i * 10)
                .unwrap(); // 0, 10, 20
        }

        let (recorder, _) = CheckpointRecorder::resume(&dir, 10).unwrap();
        recorder
            .write(&sample_model(), &make_eval(99.0, false), 20)
            .unwrap();

        let archive = CheckpointArchive::load(&dir).unwrap();
        cleanup(&dir);

        assert_eq!(archive.len(), 3); // 0, 10, 20 (rewritten)
    }

    #[test]
    fn empty_dir_archive_is_empty() {
        let dir = temp_dir("empty");
        let archive = CheckpointArchive::load(&dir).unwrap();
        cleanup(&dir);

        assert!(archive.is_empty());
        assert_eq!(archive.len(), 0);
    }

    #[test]
    fn load_ignores_non_directory_checkpoint_file() {
        let dir = temp_dir("non_dir_snap");
        fs::write(dir.join("checkpoint-000000"), b"not a dir").unwrap();
        let archive = CheckpointArchive::load(&dir).unwrap();
        cleanup(&dir);

        assert!(archive.is_empty());
    }

    #[test]
    fn load_ignores_checkpoint_dir_with_non_numeric_suffix() {
        let dir = temp_dir("non_numeric_snap");
        fs::create_dir_all(dir.join("checkpoint-abc")).unwrap();
        let archive = CheckpointArchive::load(&dir).unwrap();
        cleanup(&dir);

        assert!(archive.is_empty());
    }

    #[test]
    fn model_at_out_of_range_gives_range_error() {
        let dir = temp_dir("model_at_oob");
        let recorder = CheckpointRecorder::create(&dir, "ds", false).unwrap();
        recorder
            .write(&sample_model(), &make_eval(0.0, false), 0)
            .unwrap();

        let archive = CheckpointArchive::load(&dir).unwrap();
        let msg = archive.model_at(99).unwrap_err().to_string();
        cleanup(&dir);

        assert!(msg.contains("out of range"), "got: {msg}");
    }

    #[test]
    fn model_at_missing_model_file_fails() {
        let dir = temp_dir("model_at_missing");
        let recorder = CheckpointRecorder::create(&dir, "ds", false).unwrap();
        recorder
            .write(&sample_model(), &make_eval(0.0, false), 0)
            .unwrap();
        fs::remove_file(dir.join("checkpoint-000000").join("model.safetensors")).unwrap();

        let archive = CheckpointArchive::load(&dir).unwrap();
        let result = archive.model_at(0);
        cleanup(&dir);

        assert!(result.is_err());
    }

    #[test]
    fn model_at_corrupt_model_file_fails() {
        let dir = temp_dir("model_at_corrupt");
        let recorder = CheckpointRecorder::create(&dir, "ds", false).unwrap();
        recorder
            .write(&sample_model(), &make_eval(0.0, false), 0)
            .unwrap();
        fs::write(
            dir.join("checkpoint-000000").join("model.safetensors"),
            b"garbage",
        )
        .unwrap();

        let archive = CheckpointArchive::load(&dir).unwrap();
        let result = archive.model_at(0);
        cleanup(&dir);

        assert!(result.is_err());
    }

    #[test]
    fn evaluation_history_corrupted_evaluations_json_fails() {
        let dir = temp_dir("corrupt_evals");
        let recorder = CheckpointRecorder::create(&dir, "ds", false).unwrap();
        recorder
            .write(&sample_model(), &make_eval(0.0, false), 0)
            .unwrap();
        fs::write(
            dir.join("checkpoint-000000").join("evaluations.json"),
            b"not valid json",
        )
        .unwrap();

        let archive = CheckpointArchive::load(&dir).unwrap();
        let result = archive.evaluation_history();
        cleanup(&dir);

        assert!(result.is_err());
    }

    #[test]
    fn on_evaluate_writes_a_checkpoint() {
        let dir = temp_dir("on_evaluate");
        let mut recorder = CheckpointRecorder::create(&dir, "ds", false).unwrap();

        recorder
            .on_evaluate(&sample_model(), &make_eval(0.0, false), 5)
            .unwrap();

        let archive = CheckpointArchive::load(&dir).unwrap();
        cleanup(&dir);

        assert_eq!(archive.len(), 1);
        assert_eq!(archive.epoch_at(0), Some(5));
    }

    #[test]
    fn write_fails_when_evaluations_path_is_a_directory() {
        let dir = temp_dir("write_evals_dir_conflict");
        let recorder = CheckpointRecorder::create(&dir, "ds", false).unwrap();

        // Pre-create "evaluations.json" as a directory so json::save's fs::write fails.
        fs::create_dir_all(dir.join("checkpoint-000000").join("evaluations.json")).unwrap();

        let result = recorder.write(&sample_model(), &make_eval(0.0, false), 0);
        cleanup(&dir);

        assert!(result.is_err());
    }

    #[test]
    fn create_rejects_path_traversal() {
        let result = CheckpointRecorder::create("../../nrn_traversal_test", "x", false);
        assert!(result.is_err());
    }

    #[test]
    fn resume_rejects_path_traversal() {
        let result = CheckpointRecorder::resume("../../nrn_traversal_test", 0);
        assert!(result.is_err());
    }

    #[test]
    fn load_rejects_path_traversal() {
        let result = CheckpointArchive::load("../../nrn_traversal_test");
        assert!(result.is_err());
    }
}
