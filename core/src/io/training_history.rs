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

#[derive(Serialize, Deserialize)]
struct SnapshotEvals {
    interval: usize,
    epoch: usize,
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

/// Writes model snapshots to a directory incrementally during training.
///
/// Each call to [`record`] creates a subdirectory `snapshot-{n:06}/` containing
/// `model.safetensors` and `evaluations.json`. A reader can observe new directories
/// appearing while training is still running.
pub struct SnapshotRecorder {
    dir: PathBuf,
    interval: usize,
    count: usize,
}

impl SnapshotRecorder {
    /// Creates (or resets) the snapshot directory and returns a recorder.
    ///
    /// Any existing `snapshot-*` subdirectories in `path` are removed so that
    /// a re-run does not leave stale snapshots from a previous run.
    pub fn create<P: AsRef<Path>>(path: P, interval: usize) -> Result<Self> {
        let dir = Path::combine_safe_with_cwd(path)?;
        fs::create_dir_all(&dir)?;

        for entry in fs::read_dir(&dir)? {
            let entry = entry?;
            let p = entry.path();
            if let Some(name) = p.file_name().and_then(|n| n.to_str())
                && name.starts_with("snapshot-")
                && p.is_dir()
            {
                fs::remove_dir_all(&p)?;
            }
        }

        Ok(SnapshotRecorder {
            dir,
            interval,
            count: 0,
        })
    }

    /// Writes the current model and its evaluations as the next snapshot directory.
    ///
    /// Returns the path of the directory that was created.
    pub fn record(&mut self, model: &NeuralNetwork, evaluation: &EvaluationSet) -> Result<PathBuf> {
        let snapshot_dir = self.dir.join(format!("snapshot-{:06}", self.count));
        fs::create_dir_all(&snapshot_dir)?;

        let mut entries = Vec::new();
        let mut metadata = HashMap::new();
        model.collect_tensors(&mut entries, &mut metadata);
        tensors::save(snapshot_dir.join("model"), entries, metadata)?;

        let evals = SnapshotEvals {
            interval: self.interval,
            epoch: self.count * self.interval,
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
        Ok(snapshot_dir)
    }

    /// Returns the directory where snapshots are being written.
    pub fn dir(&self) -> &Path {
        &self.dir
    }
}

impl TrainingHistory {
    /// Loads a `TrainingHistory` from a directory of snapshot subdirectories.
    ///
    /// Only evaluations are loaded into memory. Model weights stay on disk and are
    /// read on demand via [`model_at`]. Subdirectories must match `snapshot-{n}`;
    /// they are sorted by their numeric index (not lexically), so 10 or more
    /// snapshots sort correctly. An empty directory returns an empty history with
    /// `interval = 0`.
    pub fn load<P: AsRef<Path>>(dir: P) -> Result<Self> {
        let dir = Path::combine_safe_with_cwd(dir)?;

        let mut indexed: Vec<(usize, PathBuf)> = fs::read_dir(&dir)?
            .filter_map(Result::ok)
            .filter_map(|entry| {
                let p = entry.path();
                let name = p.file_name()?.to_str()?;
                if name.starts_with("snapshot-") && p.is_dir() {
                    let idx = name.strip_prefix("snapshot-")?.parse::<usize>().ok()?;
                    Some((idx, p))
                } else {
                    None
                }
            })
            .collect();

        indexed.sort_by_key(|(idx, _)| *idx);

        if indexed.is_empty() {
            return Ok(TrainingHistory {
                interval: 0,
                evaluations: Vec::new(),
                snapshot_paths: Vec::new(),
            });
        }

        let n = indexed.len();
        let mut evaluations = Vec::with_capacity(n);
        let mut snapshot_paths = Vec::with_capacity(n);
        let mut interval = 0usize;

        for (i, (_, path)) in indexed.into_iter().enumerate() {
            let evals: SnapshotEvals = json::load(path.join("evaluations"))?;

            if i == 0 {
                interval = evals.interval;
            }

            evaluations.push(EvaluationSet {
                train: Evaluation {
                    loss: evals.train.loss,
                    accuracy: evals.train.accuracy,
                },
                validation: evals.validation.map(|v| Evaluation {
                    loss: v.loss,
                    accuracy: v.accuracy,
                }),
                test: Evaluation {
                    loss: evals.test.loss,
                    accuracy: evals.test.accuracy,
                },
            });
            snapshot_paths.push(path);
        }

        Ok(TrainingHistory {
            interval,
            evaluations,
            snapshot_paths,
        })
    }

    /// Loads the model at position `index` from its snapshot directory on disk.
    ///
    /// Only one model is read into memory per call. Returns an error when `index`
    /// is out of range or when this history was not loaded from disk (no paths).
    pub fn model_at(&self, index: usize) -> Result<NeuralNetwork> {
        if self.snapshot_paths.is_empty() {
            return Err(Error::new(
                InvalidData,
                "model_at requires a disk-loaded history; \
                 use TrainingHistory::load instead of by_interval/record",
            ));
        }
        let dir = self.snapshot_paths.get(index).ok_or_else(|| {
            Error::new(
                InvalidData,
                format!(
                    "snapshot index {index} out of range \
                     (history has {} snapshots)",
                    self.snapshot_paths.len()
                ),
            )
        })?;
        let bytes = secure_read(dir.join("model.safetensors"))?;
        let st =
            SafeTensors::deserialize(&bytes).map_err(|e| Error::new(InvalidData, e.to_string()))?;
        let metadata = tensors::read_metadata(&bytes)?;
        NeuralNetwork::from_tensors(&st, &metadata)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::RELU;
    use crate::evaluation::Evaluation;
    use crate::model::{NeuralNetwork, NeuronLayerSpec};
    use ndarray::Array2;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = PathBuf::from(format!("target/nrn_th_{}_{}", tag, std::process::id()));
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
        let mut recorder = SnapshotRecorder::create(dir, 10).unwrap();
        for i in 0..n {
            recorder
                .record(&sample_model(), &make_evaluation(i, with_validation))
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
            assert_eq!(eval.train.accuracy, 0.5 + i as f32);
            assert!(eval.validation.is_some());
            assert_eq!(eval.validation.unwrap().loss, 10.0 + i as f32);
            assert_eq!(eval.test.loss, 100.0 + i as f32);
        }
    }

    #[test]
    fn roundtrip_without_validation() {
        let dir = temp_dir("roundtrip_noval");
        write_n(&dir, 3, false);
        let history = TrainingHistory::load(&dir).unwrap();
        cleanup(&dir);

        assert_eq!(history.interval, 10);
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

        let mut recorder = SnapshotRecorder::create(&dir, 5).unwrap();
        recorder.record(&model, &make_evaluation(0, false)).unwrap();

        let history = TrainingHistory::load(&dir).unwrap();
        let loaded_model = history.model_at(0).unwrap();
        cleanup(&dir);

        assert_eq!(
            model.predict(inputs.view()),
            loaded_model.predict(inputs.view())
        );
    }

    #[test]
    fn numeric_sort_beats_lexical() {
        let dir = temp_dir("sort");
        let model = sample_model();
        let mut recorder = SnapshotRecorder::create(&dir, 1).unwrap();
        for i in 0..12 {
            recorder.record(&model, &make_evaluation(i, false)).unwrap();
        }

        let history = TrainingHistory::load(&dir).unwrap();
        cleanup(&dir);

        assert_eq!(history.len(), 12);
        for (i, eval) in history.evaluations.iter().enumerate() {
            assert_eq!(
                eval.train.loss, i as f32,
                "snapshot {i} out of order: got loss {}",
                eval.train.loss
            );
        }
    }

    #[test]
    fn create_purges_previous_snapshots() {
        let dir = temp_dir("purge");

        write_n(&dir, 3, false);
        assert_eq!(TrainingHistory::load(&dir).unwrap().len(), 3);

        write_n(&dir, 2, false);
        let history = TrainingHistory::load(&dir).unwrap();
        cleanup(&dir);

        assert_eq!(
            history.len(),
            2,
            "stale snapshots from first run were not purged"
        );
    }

    #[test]
    fn empty_dir_returns_empty_history() {
        let dir = temp_dir("empty");
        let history = TrainingHistory::load(&dir).unwrap();
        cleanup(&dir);

        assert_eq!(history.interval, 0);
        assert!(history.is_empty());
    }

    #[test]
    fn nonexistent_dir_fails() {
        let result = TrainingHistory::load("target/nrn_th_does_not_exist_9999999");
        assert!(result.is_err());
    }

    #[test]
    fn corrupted_snapshot_fails() {
        let dir = temp_dir("corrupt");

        write_n(&dir, 1, false);
        // Add a second snapshot dir with a corrupt evaluations.json.
        let bad_dir = dir.join("snapshot-000001");
        fs::create_dir_all(&bad_dir).unwrap();
        fs::write(bad_dir.join("evaluations.json"), b"not valid json").unwrap();

        let result = TrainingHistory::load(&dir);
        cleanup(&dir);
        assert!(result.is_err());
    }

    #[test]
    fn missing_interval_metadata_fails() {
        let dir = temp_dir("no_interval");

        let bad_dir = dir.join("snapshot-000000");
        fs::create_dir_all(&bad_dir).unwrap();
        // evaluations.json without required 'interval' field.
        fs::write(
            bad_dir.join("evaluations.json"),
            br#"{"epoch":0,"train":{"loss":0.0,"accuracy":0.0},"test":{"loss":0.0,"accuracy":0.0}}"#,
        )
        .unwrap();

        let result = TrainingHistory::load(&dir);
        cleanup(&dir);
        assert!(result.is_err());
    }

    #[test]
    fn non_numeric_interval_fails() {
        let dir = temp_dir("bad_interval");

        let bad_dir = dir.join("snapshot-000000");
        fs::create_dir_all(&bad_dir).unwrap();
        // 'interval' is a string instead of a usize.
        fs::write(
            bad_dir.join("evaluations.json"),
            br#"{"interval":"not_a_number","epoch":0,"train":{"loss":0.0,"accuracy":0.0},"test":{"loss":0.0,"accuracy":0.0}}"#,
        )
        .unwrap();

        let result = TrainingHistory::load(&dir);
        cleanup(&dir);
        assert!(result.is_err());
    }

    #[test]
    fn record_returns_expected_path() {
        let dir = temp_dir("ret_path");
        let mut recorder = SnapshotRecorder::create(&dir, 5).unwrap();
        let path = recorder
            .record(&sample_model(), &make_evaluation(0, false))
            .unwrap();
        cleanup(&dir);

        let name = path.file_name().unwrap().to_str().unwrap();
        assert_eq!(name, "snapshot-000000");
    }

    #[test]
    fn validation_tensor_written_iff_present() {
        let dir = temp_dir("val_tensor");
        let model = sample_model();

        let mut recorder = SnapshotRecorder::create(&dir, 1).unwrap();
        recorder.record(&model, &make_evaluation(0, false)).unwrap();
        recorder.record(&model, &make_evaluation(1, true)).unwrap();

        let history = TrainingHistory::load(&dir).unwrap();
        cleanup(&dir);

        assert!(history.evaluations[0].validation.is_none());
        assert!(history.evaluations[1].validation.is_some());
    }

    #[test]
    fn model_at_out_of_range_fails() {
        let dir = temp_dir("model_at_oob");
        write_n(&dir, 2, false);
        let history = TrainingHistory::load(&dir).unwrap();
        cleanup(&dir);

        assert!(history.model_at(2).is_err());
    }

    #[test]
    fn model_at_in_memory_history_fails_with_clear_message() {
        let mut th = TrainingHistory::by_interval(1, 1).unwrap();
        th.record(&crate::evaluation::EvaluationSet {
            train: crate::evaluation::Evaluation {
                loss: 0.0,
                accuracy: 0.0,
            },
            validation: None,
            test: crate::evaluation::Evaluation {
                loss: 0.0,
                accuracy: 0.0,
            },
        });
        let err = th.model_at(0).err().unwrap();
        let msg = err.to_string();
        assert!(
            msg.contains("disk"),
            "error message should mention 'disk', got: {msg}"
        );
    }

    #[test]
    fn model_at_out_of_range_gives_range_error() {
        let dir = temp_dir("model_at_oob2");
        write_n(&dir, 2, false);
        let history = TrainingHistory::load(&dir).unwrap();
        cleanup(&dir);

        let err = history.model_at(99).err().unwrap();
        let msg = err.to_string();
        assert!(
            msg.contains("out of range") || msg.contains("index"),
            "expected range error, got: {msg}"
        );
    }
}
