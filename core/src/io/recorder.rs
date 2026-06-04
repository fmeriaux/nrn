use crate::evaluation::EvaluationSet;
use crate::io::json;
use crate::io::path::PathExt;
use crate::io::tensors;
use crate::model::NeuralNetwork;
use crate::recorders::SnapshotRecorder;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::ErrorKind::InvalidData;
use std::io::{Error, Result};
use std::path::{Path, PathBuf};

/// Top-level metadata for a training history directory.
/// Written once by [`FileSnapshotRecorder::create`] into `meta.json`.
#[derive(Serialize, Deserialize)]
pub struct TrainingMeta {
    pub dataset: String,
    pub interval: usize,
}

impl TrainingMeta {
    /// Loads the metadata from `meta.json` inside `dir`.
    pub fn load<P: AsRef<Path>>(dir: P) -> Result<Self> {
        let dir = Path::combine_safe_with_cwd(dir)?;
        json::load(dir.join("meta"))
    }
}

#[derive(Serialize, Deserialize)]
pub(crate) struct SnapshotEvals {
    pub(crate) epoch: usize,
    pub(crate) train: MetricPair,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) validation: Option<MetricPair>,
    pub(crate) test: MetricPair,
}

#[derive(Serialize, Deserialize)]
pub(crate) struct MetricPair {
    pub(crate) loss: f32,
    pub(crate) accuracy: f32,
}

/// Writes model snapshots to a directory incrementally during training.
///
/// Each call to [`record`] creates a subdirectory `snapshot-{n:06}/` containing
/// `model.safetensors` and `evaluations.json`.
#[derive(Debug)]
pub struct FileSnapshotRecorder {
    dir: PathBuf,
    count: usize,
}

impl FileSnapshotRecorder {
    /// Creates a fresh recorder at `path`, starting count at 0.
    ///
    /// Writes `meta.json` with dataset name and checkpoint interval.
    /// Returns an error if `snapshot-*` subdirectories already exist and
    /// `overwrite` is `false`.
    pub fn create<P: AsRef<Path>>(
        path: P,
        interval: usize,
        dataset: &str,
        overwrite: bool,
    ) -> Result<Self> {
        let dir = Path::combine_safe_with_cwd(path)?;
        fs::create_dir_all(&dir)?;

        let existing: Vec<PathBuf> = fs::read_dir(&dir)?
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("snapshot-"))
                    .unwrap_or(false)
                    && p.is_dir()
            })
            .collect();

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
            for p in existing {
                fs::remove_dir_all(&p)?;
            }
        }

        json::save(
            &TrainingMeta {
                dataset: dataset.to_string(),
                interval,
            },
            dir.join("meta"),
        )?;

        Ok(FileSnapshotRecorder { dir, count: 0 })
    }

    /// Resumes recording into an existing snapshot directory, starting count at
    /// `from_count + 1`. Snapshots with index > `from_count` are removed.
    pub fn resume<P: AsRef<Path>>(path: P, from_count: usize) -> Result<Self> {
        let dir = Path::combine_safe_with_cwd(path)?;
        fs::create_dir_all(&dir)?;

        for entry in fs::read_dir(&dir)? {
            let entry = entry?;
            let p = entry.path();
            if let Some(idx) = p
                .file_name()
                .and_then(|n| n.to_str())
                .and_then(|n| n.strip_prefix("snapshot-"))
                .and_then(|s| s.parse::<usize>().ok())
                && p.is_dir()
                && idx > from_count
            {
                fs::remove_dir_all(&p)?;
            }
        }

        Ok(FileSnapshotRecorder {
            dir,
            count: from_count + 1,
        })
    }
}

impl SnapshotRecorder for FileSnapshotRecorder {
    fn record(
        &mut self,
        model: &NeuralNetwork,
        evaluation: &EvaluationSet,
        epoch: usize,
    ) -> Result<()> {
        let snapshot_dir = self.dir.join(format!("snapshot-{:06}", self.count));
        fs::create_dir_all(&snapshot_dir)?;

        let mut entries = Vec::new();
        let mut metadata = HashMap::new();
        model.collect_tensors(&mut entries, &mut metadata);
        tensors::save(snapshot_dir.join("model"), entries, metadata)?;

        let evals = SnapshotEvals {
            epoch,
            train: MetricPair {
                loss: evaluation.train.loss,
                accuracy: evaluation.train.accuracy,
            },
            validation: evaluation.validation.map(|v| MetricPair {
                loss: v.loss,
                accuracy: v.accuracy,
            }),
            test: MetricPair {
                loss: evaluation.test.loss,
                accuracy: evaluation.test.accuracy,
            },
        };
        json::save(&evals, snapshot_dir.join("evaluations"))?;

        self.count += 1;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::RELU;
    use crate::evaluation::Evaluation;
    use crate::model::{NeuralNetwork, NeuronLayerSpec};
    use crate::training_history::TrainingHistory;
    use ndarray::Array2;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = PathBuf::from(format!("target/nrn_sr_{}_{}", tag, std::process::id()));
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

    fn make_evaluation(i: usize, with_validation: bool) -> EvaluationSet {
        EvaluationSet {
            train: Evaluation {
                loss: i as f32,
                accuracy: 0.5 + i as f32,
            },
            validation: with_validation.then_some(Evaluation {
                loss: 10.0 + i as f32,
                accuracy: 0.1 * i as f32,
            }),
            test: Evaluation {
                loss: 100.0 + i as f32,
                accuracy: 0.9 - 0.1 * i as f32,
            },
        }
    }

    fn write_n(dir: &Path, n: usize, with_validation: bool) {
        let mut recorder = FileSnapshotRecorder::create(dir, 10, "test_dataset", false).unwrap();
        for i in 0..n {
            recorder
                .record(
                    &sample_model(),
                    &make_evaluation(i, with_validation),
                    i * 10,
                )
                .unwrap();
        }
    }

    #[test]
    fn roundtrip_with_validation() {
        let dir = temp_dir("roundtrip_val");
        write_n(&dir, 3, true);
        let history = TrainingHistory::load(&dir).unwrap();
        cleanup(&dir);

        assert_eq!(history.interval, 10);
        assert_eq!(history.len(), 3);
        for (i, eval) in history.evaluations.iter().enumerate() {
            assert_eq!(eval.train.loss, i as f32);
            assert!(eval.validation.is_some());
        }
    }

    #[test]
    fn roundtrip_without_validation() {
        let dir = temp_dir("roundtrip_noval");
        write_n(&dir, 3, false);
        let history = TrainingHistory::load(&dir).unwrap();
        cleanup(&dir);

        assert_eq!(history.len(), 3);
        for eval in &history.evaluations {
            assert!(eval.validation.is_none());
        }
    }

    #[test]
    fn epoch_stored_from_caller() {
        let dir = temp_dir("epoch");
        let mut recorder = FileSnapshotRecorder::create(&dir, 5, "ds", false).unwrap();
        recorder
            .record(&sample_model(), &make_evaluation(0, false), 37)
            .unwrap();

        // Load the raw evaluations.json to verify epoch was stored correctly.
        let bytes = std::fs::read(dir.join("snapshot-000000").join("evaluations.json")).unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        cleanup(&dir);

        assert_eq!(json["epoch"], 37);
    }

    #[test]
    fn predictions_survive_roundtrip() {
        let dir = temp_dir("predictions");
        let model = sample_model();
        let inputs = Array2::from_shape_fn((2, 4), |(i, j)| (i + j) as f32 * 0.3);

        let mut recorder = FileSnapshotRecorder::create(&dir, 5, "ds", false).unwrap();
        recorder
            .record(&model, &make_evaluation(0, false), 0)
            .unwrap();

        let history = TrainingHistory::load(&dir).unwrap();
        let loaded = history.model_at(0).unwrap();
        cleanup(&dir);

        assert_eq!(model.predict(inputs.view()), loaded.predict(inputs.view()));
    }

    #[test]
    fn numeric_sort_beats_lexical() {
        let dir = temp_dir("sort");
        let model = sample_model();
        let mut recorder = FileSnapshotRecorder::create(&dir, 1, "ds", false).unwrap();
        for i in 0..12 {
            recorder
                .record(&model, &make_evaluation(i, false), i)
                .unwrap();
        }

        let history = TrainingHistory::load(&dir).unwrap();
        cleanup(&dir);

        assert_eq!(history.len(), 12);
        for (i, eval) in history.evaluations.iter().enumerate() {
            assert_eq!(eval.train.loss, i as f32);
        }
    }

    #[test]
    fn meta_json_written_by_create() {
        let dir = temp_dir("meta_written");
        FileSnapshotRecorder::create(&dir, 10, "my_dataset", false).unwrap();

        let meta = TrainingMeta::load(&dir).unwrap();
        cleanup(&dir);

        assert_eq!(meta.dataset, "my_dataset");
        assert_eq!(meta.interval, 10);
    }

    #[test]
    fn load_reads_interval_from_meta() {
        let dir = temp_dir("meta_interval");
        write_n(&dir, 2, false);
        let history = TrainingHistory::load(&dir).unwrap();
        cleanup(&dir);
        assert_eq!(history.interval, 10);
    }

    #[test]
    fn create_errors_if_snapshots_exist_without_overwrite() {
        let dir = temp_dir("no_overwrite");
        write_n(&dir, 2, false);

        let result = FileSnapshotRecorder::create(&dir, 10, "ds", false);
        cleanup(&dir);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("already exists"));
    }

    #[test]
    fn create_with_overwrite_purges_previous_snapshots() {
        let dir = temp_dir("overwrite");
        write_n(&dir, 3, false);

        let mut recorder = FileSnapshotRecorder::create(&dir, 10, "ds", true).unwrap();
        for i in 0..2 {
            recorder
                .record(&sample_model(), &make_evaluation(i, false), i * 10)
                .unwrap();
        }
        let history = TrainingHistory::load(&dir).unwrap();
        cleanup(&dir);

        assert_eq!(history.len(), 2, "stale snapshots were not purged");
    }

    #[test]
    fn resume_continues_from_correct_count() {
        let dir = temp_dir("resume_count");
        write_n(&dir, 3, false); // snapshots 0, 1, 2

        let mut recorder = FileSnapshotRecorder::resume(&dir, 2).unwrap();
        recorder
            .record(&sample_model(), &make_evaluation(99, false), 99)
            .unwrap();

        let history = TrainingHistory::load(&dir).unwrap();
        cleanup(&dir);

        assert_eq!(history.len(), 4);
        assert_eq!(history.evaluations[3].train.loss, 99.0);
    }

    #[test]
    fn resume_removes_snapshots_after_resume_point() {
        let dir = temp_dir("resume_trim");
        write_n(&dir, 5, false);

        FileSnapshotRecorder::resume(&dir, 2).unwrap();

        let history = TrainingHistory::load(&dir).unwrap();
        cleanup(&dir);

        assert_eq!(history.len(), 3);
    }

    #[test]
    fn validation_tensor_written_iff_present() {
        let dir = temp_dir("val_tensor");
        let model = sample_model();
        let mut recorder = FileSnapshotRecorder::create(&dir, 1, "ds", false).unwrap();
        recorder
            .record(&model, &make_evaluation(0, false), 0)
            .unwrap();
        recorder
            .record(&model, &make_evaluation(1, true), 1)
            .unwrap();

        let history = TrainingHistory::load(&dir).unwrap();
        cleanup(&dir);

        assert!(history.evaluations[0].validation.is_none());
        assert!(history.evaluations[1].validation.is_some());
    }

    #[test]
    fn create_rejects_path_traversal() {
        let result = FileSnapshotRecorder::create("../../nrn_traversal_test", 1, "x", false);
        assert!(result.is_err());
    }

    #[test]
    fn resume_rejects_path_traversal() {
        let result = FileSnapshotRecorder::resume("../../nrn_traversal_test", 0);
        assert!(result.is_err());
    }
}
