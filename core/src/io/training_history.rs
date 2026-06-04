use crate::evaluation::{Evaluation, EvaluationSet};
use crate::io::bytes::secure_read;
use crate::io::path::PathExt;
use crate::io::tensors;
use crate::model::NeuralNetwork;
use crate::training_history::TrainingHistory;
use ndarray::Array1;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::fs;
use std::io::ErrorKind::InvalidData;
use std::io::{Error, Result};
use std::path::{Path, PathBuf};

/// Writes model snapshots to a directory incrementally during training.
///
/// Each call to [`record`] writes one self-contained `snapshot-{n:06}.safetensors`
/// file. A reader can observe new files appearing while training is still running.
pub struct TrainingHistoryWriter {
    dir: PathBuf,
    interval: usize,
    count: usize,
}

impl TrainingHistoryWriter {
    /// Creates (or resets) the snapshot directory and returns a writer.
    ///
    /// Any existing `snapshot-*.safetensors` files in `path` are removed so
    /// that a re-run of training does not leave stale snapshots from a previous run.
    pub fn create<P: AsRef<Path>>(path: P, interval: usize) -> Result<Self> {
        let dir = Path::combine_safe_with_cwd(path)?;
        fs::create_dir_all(&dir)?;

        for entry in fs::read_dir(&dir)? {
            let entry = entry?;
            let p = entry.path();
            if let Some(name) = p.file_name().and_then(|n| n.to_str())
                && name.starts_with("snapshot-")
                && name.ends_with(".safetensors")
            {
                fs::remove_file(&p)?;
            }
        }

        Ok(TrainingHistoryWriter {
            dir,
            interval,
            count: 0,
        })
    }

    /// Writes the current model and its evaluations as the next snapshot file.
    ///
    /// Returns the path of the file that was written.
    pub fn record(&mut self, model: &NeuralNetwork, evaluation: &EvaluationSet) -> Result<PathBuf> {
        let mut entries = Vec::new();
        let mut metadata = HashMap::new();

        model.collect_tensors(&mut entries, &mut metadata);

        let eval_data = Array1::from_vec(vec![
            evaluation.train.loss,
            evaluation.train.accuracy,
            evaluation.test.loss,
            evaluation.test.accuracy,
        ]);
        entries.push(("evaluation".to_string(), tensors::tensor(&eval_data)));

        if let Some(validation) = evaluation.validation {
            let val_data = Array1::from_vec(vec![validation.loss, validation.accuracy]);
            entries.push(("validation".to_string(), tensors::tensor(&val_data)));
        }

        metadata.insert("interval".to_string(), self.interval.to_string());
        metadata.insert(
            "epoch".to_string(),
            (self.count * self.interval).to_string(),
        );

        let snapshot_path = self.dir.join(format!("snapshot-{:06}", self.count));
        let written = tensors::save(snapshot_path, entries, metadata)?;
        self.count += 1;
        Ok(written)
    }

    /// Returns the directory where snapshots are being written.
    pub fn dir(&self) -> &Path {
        &self.dir
    }
}

impl TrainingHistory {
    /// Loads a `TrainingHistory` from a directory of snapshot files.
    ///
    /// Only evaluations are loaded into memory. Model weights stay on disk and are
    /// read on demand via [`model_at`]. Files must match `snapshot-{n}.safetensors`;
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
                if name.starts_with("snapshot-") && name.ends_with(".safetensors") {
                    let idx = name
                        .strip_prefix("snapshot-")?
                        .strip_suffix(".safetensors")?
                        .parse::<usize>()
                        .ok()?;
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
            let bytes = secure_read(&path)?;
            let st = SafeTensors::deserialize(&bytes)
                .map_err(|e| Error::new(InvalidData, e.to_string()))?;

            // Extract interval once from the first file (every snapshot carries it).
            if i == 0 {
                let meta = tensors::read_metadata(&bytes)?;
                interval = tensors::meta(&meta, "interval")?
                    .parse()
                    .map_err(|e| Error::new(InvalidData, format!("invalid interval: {e}")))?;
            }

            let eval_1d = tensors::read_array1("evaluation", &st)?;
            let validation = tensors::read_array1("validation", &st)
                .ok()
                .map(|v| Evaluation {
                    loss: v[0],
                    accuracy: v[1],
                });
            evaluations.push(EvaluationSet {
                train: Evaluation {
                    loss: eval_1d[0],
                    accuracy: eval_1d[1],
                },
                validation,
                test: Evaluation {
                    loss: eval_1d[2],
                    accuracy: eval_1d[3],
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

    /// Loads the model at position `index` from its snapshot file on disk.
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
        let path = self.snapshot_paths.get(index).ok_or_else(|| {
            Error::new(
                InvalidData,
                format!(
                    "snapshot index {index} out of range \
                     (history has {} snapshots)",
                    self.snapshot_paths.len()
                ),
            )
        })?;
        let bytes = secure_read(path)?;
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
        let mut writer = TrainingHistoryWriter::create(dir, 10).unwrap();
        for i in 0..n {
            writer
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

        let mut writer = TrainingHistoryWriter::create(&dir, 5).unwrap();
        writer.record(&model, &make_evaluation(0, false)).unwrap();

        let history = TrainingHistory::load(&dir).unwrap();
        // Load model lazily — only one in memory at a time.
        let loaded_model = history.model_at(0).unwrap();
        cleanup(&dir);

        assert_eq!(
            model.predict(inputs.view()),
            loaded_model.predict(inputs.view())
        );
    }

    #[test]
    fn numeric_sort_beats_lexical() {
        // 12 snapshots: lexical sort puts snapshot-10 before snapshot-2.
        // Numeric sort must restore the correct order.
        let dir = temp_dir("sort");
        let model = sample_model();
        let mut writer = TrainingHistoryWriter::create(&dir, 1).unwrap();
        for i in 0..12 {
            writer.record(&model, &make_evaluation(i, false)).unwrap();
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

        // First run: 3 snapshots.
        write_n(&dir, 3, false);
        assert_eq!(TrainingHistory::load(&dir).unwrap().len(), 3);

        // Second run with a fresh writer: should see only 2 new snapshots.
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

        // Write one valid snapshot, then add a corrupt file.
        write_n(&dir, 1, false);
        fs::write(dir.join("snapshot-000001.safetensors"), b"not safetensors").unwrap();

        let result = TrainingHistory::load(&dir);
        cleanup(&dir);
        assert!(result.is_err());
    }

    #[test]
    fn missing_interval_metadata_fails() {
        let dir = temp_dir("no_interval");

        let model = sample_model();
        let mut entries = Vec::new();
        let mut metadata = HashMap::new();
        model.collect_tensors(&mut entries, &mut metadata);
        // omit "interval" intentionally
        let eval_data = Array1::from_vec(vec![0.0_f32, 0.0, 0.0, 0.0]);
        entries.push(("evaluation".to_string(), tensors::tensor(&eval_data)));
        tensors::save(dir.join("snapshot-000000"), entries, metadata).unwrap();

        let result = TrainingHistory::load(&dir);
        cleanup(&dir);
        assert!(result.is_err());
    }

    #[test]
    fn non_numeric_interval_fails() {
        let dir = temp_dir("bad_interval");

        let model = sample_model();
        let mut entries = Vec::new();
        let mut metadata = HashMap::new();
        model.collect_tensors(&mut entries, &mut metadata);
        metadata.insert("interval".to_string(), "not_a_number".to_string());
        let eval_data = Array1::from_vec(vec![0.0_f32, 0.0, 0.0, 0.0]);
        entries.push(("evaluation".to_string(), tensors::tensor(&eval_data)));
        tensors::save(dir.join("snapshot-000000"), entries, metadata).unwrap();

        let result = TrainingHistory::load(&dir);
        cleanup(&dir);
        assert!(result.is_err());
    }

    #[test]
    fn record_returns_expected_path() {
        let dir = temp_dir("ret_path");
        let mut writer = TrainingHistoryWriter::create(&dir, 5).unwrap();
        let path = writer
            .record(&sample_model(), &make_evaluation(0, false))
            .unwrap();
        cleanup(&dir);

        let name = path.file_name().unwrap().to_str().unwrap();
        assert_eq!(name, "snapshot-000000.safetensors");
    }

    #[test]
    fn validation_tensor_written_iff_present() {
        let dir = temp_dir("val_tensor");
        let model = sample_model();

        let mut writer = TrainingHistoryWriter::create(&dir, 1).unwrap();
        writer.record(&model, &make_evaluation(0, false)).unwrap();
        writer.record(&model, &make_evaluation(1, true)).unwrap();

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
        // A history built with by_interval has no snapshot_paths.
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
        // Must explicitly say "disk" so callers know the issue is no disk backing,
        // not an out-of-range index.
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
