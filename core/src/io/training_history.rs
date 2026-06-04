use crate::evaluation::{Evaluation, EvaluationSet};
use crate::io::bytes::secure_read;
use crate::io::json;
use crate::io::path::PathExt;
use crate::io::recorder::{SnapshotEvals, TrainingMeta};
use crate::io::tensors;
use crate::model::NeuralNetwork;
use crate::training_history::TrainingHistory;
use safetensors::SafeTensors;
use std::fs;
use std::io::ErrorKind::InvalidData;
use std::io::{Error, Result};
use std::path::{Path, PathBuf};

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

        let interval = TrainingMeta::load(&dir).map(|m| m.interval).unwrap_or(0);

        if indexed.is_empty() {
            return Ok(TrainingHistory {
                interval,
                evaluations: Vec::new(),
                snapshot_paths: Vec::new(),
            });
        }

        let n = indexed.len();
        let mut evaluations = Vec::with_capacity(n);
        let mut snapshot_paths = Vec::with_capacity(n);

        for (_, path) in indexed {
            let evals: SnapshotEvals = json::load(path.join("evaluations"))?;

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
    use crate::io::recorder::FileSnapshotRecorder;
    use crate::model::{NeuralNetwork, NeuronLayerSpec};
    use crate::recorders::SnapshotRecorder;

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

    fn write_n_snapshots(dir: &Path, n: usize) {
        let mut recorder = FileSnapshotRecorder::create(dir, 10, "test_dataset", false).unwrap();
        for i in 0..n {
            recorder
                .record(
                    &sample_model(),
                    &crate::evaluation::EvaluationSet {
                        train: crate::evaluation::Evaluation {
                            loss: i as f32,
                            accuracy: 0.5,
                        },
                        validation: None,
                        test: crate::evaluation::Evaluation {
                            loss: i as f32,
                            accuracy: 0.5,
                        },
                    },
                    i * 10,
                )
                .unwrap();
        }
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
        assert!(TrainingHistory::load("target/nrn_th_does_not_exist_9999999").is_err());
    }

    #[test]
    fn corrupted_evaluations_json_fails() {
        let dir = temp_dir("corrupt_evals");
        write_n_snapshots(&dir, 1);
        let bad_dir = dir.join("snapshot-000001");
        fs::create_dir_all(&bad_dir).unwrap();
        fs::write(bad_dir.join("evaluations.json"), b"not valid json").unwrap();

        let result = TrainingHistory::load(&dir);
        cleanup(&dir);
        assert!(result.is_err());
    }

    #[test]
    fn missing_meta_json_returns_interval_zero() {
        let dir = temp_dir("no_meta");
        let snap_dir = dir.join("snapshot-000000");
        fs::create_dir_all(&snap_dir).unwrap();
        fs::write(
            snap_dir.join("evaluations.json"),
            br#"{"epoch":0,"train":{"loss":0.0,"accuracy":0.0},"test":{"loss":0.0,"accuracy":0.0}}"#,
        )
        .unwrap();

        let history = TrainingHistory::load(&dir).unwrap();
        cleanup(&dir);
        assert_eq!(history.interval, 0);
    }

    #[test]
    fn load_ignores_non_directory_snapshot_file() {
        let dir = temp_dir("non_dir_snap");
        fs::write(dir.join("snapshot-000000"), b"not a dir").unwrap();
        let history = TrainingHistory::load(&dir).unwrap();
        cleanup(&dir);
        assert!(history.is_empty());
    }

    #[test]
    fn load_ignores_snapshot_dir_with_non_numeric_suffix() {
        let dir = temp_dir("non_numeric_snap");
        fs::create_dir_all(dir.join("snapshot-abc")).unwrap();
        let history = TrainingHistory::load(&dir).unwrap();
        cleanup(&dir);
        assert!(history.is_empty());
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
        let msg = th.model_at(0).err().unwrap().to_string();
        assert!(msg.contains("disk"), "got: {msg}");
    }

    #[test]
    fn model_at_out_of_range_gives_range_error() {
        let dir = temp_dir("model_at_oob");
        write_n_snapshots(&dir, 2);
        let history = TrainingHistory::load(&dir).unwrap();
        cleanup(&dir);

        let msg = history.model_at(99).err().unwrap().to_string();
        assert!(
            msg.contains("out of range") || msg.contains("index"),
            "got: {msg}"
        );
    }

    #[test]
    fn model_at_missing_model_file_fails() {
        let dir = temp_dir("model_at_missing");
        write_n_snapshots(&dir, 1);
        let history = TrainingHistory::load(&dir).unwrap();
        fs::remove_file(dir.join("snapshot-000000").join("model.safetensors")).unwrap();

        assert!(history.model_at(0).is_err());
        cleanup(&dir);
    }

    #[test]
    fn model_at_corrupt_model_file_fails() {
        let dir = temp_dir("model_at_corrupt");
        write_n_snapshots(&dir, 1);
        let history = TrainingHistory::load(&dir).unwrap();
        fs::write(
            dir.join("snapshot-000000").join("model.safetensors"),
            b"garbage",
        )
        .unwrap();

        assert!(history.model_at(0).is_err());
        cleanup(&dir);
    }
}
