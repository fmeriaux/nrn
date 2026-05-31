use crate::checkpoints::Checkpoints;
use crate::evaluation::{Evaluation, EvaluationSet};
use crate::io::tensors;
use crate::model::NeuralNetwork;
use ndarray::Array2;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::io::ErrorKind::InvalidData;
use std::io::{Error, Result};
use std::path::{Path, PathBuf};

impl Checkpoints {
    /// Saves the checkpoints to a `.safetensors` file.
    ///
    /// Each snapshot's weights live under a `snapshot{i}.` prefix; the per-split
    /// loss/accuracy series are stored as `f32` tensors (`evaluations`, and the
    /// optional `validations`). The validation series is emitted only when every
    /// checkpoint recorded one, so its presence alone encodes the `Option`.
    /// # Arguments
    /// - `path`: The path to the file where the checkpoints will be saved.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<PathBuf> {
        let mut entries = Vec::new();
        let mut metadata = HashMap::new();

        metadata.insert("interval".to_string(), self.interval.to_string());
        metadata.insert("n_snapshots".to_string(), self.snapshots.len().to_string());

        for (i, snapshot) in self.snapshots.iter().enumerate() {
            snapshot.collect_tensors(&format!("snapshot{i}."), &mut entries, &mut metadata);
        }

        if !self.evaluations.is_empty() {
            let n = self.evaluations.len();

            let mut main = Array2::<f32>::zeros((n, 4));
            for (i, eval) in self.evaluations.iter().enumerate() {
                main[[i, 0]] = eval.train.loss;
                main[[i, 1]] = eval.train.accuracy;
                main[[i, 2]] = eval.test.loss;
                main[[i, 3]] = eval.test.accuracy;
            }
            entries.push(("evaluations".to_string(), tensors::tensor(&main)));

            if self.evaluations.iter().all(|e| e.validation.is_some()) {
                let mut validations = Array2::<f32>::zeros((n, 2));
                for (i, eval) in self.evaluations.iter().enumerate() {
                    let validation = eval.validation.expect("checked all are Some");
                    validations[[i, 0]] = validation.loss;
                    validations[[i, 1]] = validation.accuracy;
                }
                entries.push(("validations".to_string(), tensors::tensor(&validations)));
            }
        }

        tensors::save(path, entries, metadata)
    }

    /// Loads the checkpoints from a `.safetensors` file.
    /// # Arguments
    /// - `path`: The path to the file to load the checkpoints from.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let bytes = tensors::load(path)?;
        let st =
            SafeTensors::deserialize(&bytes).map_err(|e| Error::new(InvalidData, e.to_string()))?;
        let metadata = tensors::read_metadata(&bytes)?;

        let interval: usize = tensors::meta(&metadata, "interval")?
            .parse()
            .map_err(|e| Error::new(InvalidData, format!("invalid interval: {e}")))?;

        let n_snapshots: usize = tensors::meta(&metadata, "n_snapshots")?
            .parse()
            .map_err(|e| Error::new(InvalidData, format!("invalid snapshot count: {e}")))?;

        let mut snapshots = Vec::with_capacity(n_snapshots);
        for i in 0..n_snapshots {
            snapshots.push(NeuralNetwork::from_tensors(
                &format!("snapshot{i}."),
                &st,
                &metadata,
            )?);
        }

        // A missing `evaluations` tensor means none were recorded.
        let evaluations = match tensors::read_array2("evaluations", &st) {
            Ok(main) => {
                let validations = tensors::read_array2("validations", &st).ok();
                (0..main.nrows())
                    .map(|i| EvaluationSet {
                        train: Evaluation {
                            loss: main[[i, 0]],
                            accuracy: main[[i, 1]],
                        },
                        validation: validations.as_ref().map(|v| Evaluation {
                            loss: v[[i, 0]],
                            accuracy: v[[i, 1]],
                        }),
                        test: Evaluation {
                            loss: main[[i, 2]],
                            accuracy: main[[i, 3]],
                        },
                    })
                    .collect()
            }
            Err(_) => Vec::new(),
        };

        Ok(Checkpoints {
            interval,
            snapshots,
            evaluations,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::activations::RELU;
    use crate::checkpoints::Checkpoints;
    use crate::evaluation::{Evaluation, EvaluationSet};
    use crate::model::{NeuralNetwork, NeuronLayerSpec};
    use ndarray::Array2;
    use std::path::{Path, PathBuf};

    fn temp_path(tag: &str) -> PathBuf {
        PathBuf::from(format!("target/nrn_test_{tag}_{}", std::process::id()))
    }

    fn cleanup(path: &Path) {
        let _ = std::fs::remove_file(path.with_extension("safetensors"));
    }

    fn sample_model() -> NeuralNetwork {
        let specs = NeuronLayerSpec::network_for(vec![3], &*RELU, 2);
        NeuralNetwork::initialization(2, &specs)
    }

    fn checkpoints(with_validation: bool) -> Checkpoints {
        let mut checkpoints = Checkpoints::by_interval(10, 30).unwrap();
        for i in 0..3 {
            let evaluation = EvaluationSet {
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
            };
            checkpoints.record(&sample_model(), &evaluation);
        }
        checkpoints
    }

    fn assert_roundtrip(with_validation: bool) {
        let original = checkpoints(with_validation);
        let path = temp_path(if with_validation {
            "ckpt_val"
        } else {
            "ckpt_noval"
        });
        original.save(&path).unwrap();
        let loaded = Checkpoints::load(&path).unwrap();
        cleanup(&path);

        assert_eq!(loaded.interval, original.interval);
        assert_eq!(loaded.snapshots.len(), original.snapshots.len());
        assert_eq!(loaded.evaluations.len(), original.evaluations.len());

        // Evaluations round-trip exactly (no transformation, just f32 storage).
        for (a, b) in original.evaluations.iter().zip(&loaded.evaluations) {
            assert_eq!(a.train.loss, b.train.loss);
            assert_eq!(a.train.accuracy, b.train.accuracy);
            assert_eq!(a.test.loss, b.test.loss);
            assert_eq!(a.test.accuracy, b.test.accuracy);
            assert_eq!(a.validation.is_some(), b.validation.is_some());
            if let (Some(va), Some(vb)) = (a.validation, b.validation) {
                assert_eq!(va.loss, vb.loss);
                assert_eq!(va.accuracy, vb.accuracy);
            }
        }

        // Snapshots round-trip to identical predictions.
        let inputs = Array2::from_shape_fn((2, 4), |(i, j)| (i + j) as f32 * 0.3);
        for (a, b) in original.snapshots.iter().zip(&loaded.snapshots) {
            assert_eq!(a.predict(inputs.view()), b.predict(inputs.view()));
        }
    }

    #[test]
    fn roundtrip_with_validation() {
        assert_roundtrip(true);
    }

    #[test]
    fn roundtrip_without_validation() {
        assert_roundtrip(false);
    }

    #[test]
    fn empty_checkpoints_roundtrip() {
        // No snapshots and no evaluations: the evaluation tensors are skipped on
        // save and reload as empty vectors.
        let original = Checkpoints {
            interval: 7,
            snapshots: Vec::new(),
            evaluations: Vec::new(),
        };
        let path = temp_path("ckpt_empty");
        original.save(&path).unwrap();
        let loaded = Checkpoints::load(&path).unwrap();
        cleanup(&path);

        assert_eq!(loaded.interval, 7);
        assert!(loaded.snapshots.is_empty());
        assert!(loaded.evaluations.is_empty());
    }

    #[test]
    fn load_rejects_corrupt_file() {
        let path = temp_path("ckpt_corrupt");
        std::fs::write(
            path.with_extension("safetensors"),
            b"not a safetensors buffer",
        )
        .unwrap();

        assert!(Checkpoints::load(&path).is_err());
        cleanup(&path);
    }

    #[test]
    fn load_rejects_non_numeric_metadata() {
        use crate::io::tensors;
        use ndarray::array;
        use std::collections::HashMap;

        let entry = || vec![("placeholder".to_string(), tensors::tensor(&array![0.0_f32]))];

        // Non-numeric interval.
        let path = temp_path("ckpt_bad_interval");
        let mut metadata = HashMap::new();
        metadata.insert("interval".to_string(), "xx".to_string());
        metadata.insert("n_snapshots".to_string(), "0".to_string());
        tensors::save(&path, entry(), metadata).unwrap();
        assert!(Checkpoints::load(&path).is_err());
        cleanup(&path);

        // Non-numeric snapshot count.
        let path = temp_path("ckpt_bad_count");
        let mut metadata = HashMap::new();
        metadata.insert("interval".to_string(), "1".to_string());
        metadata.insert("n_snapshots".to_string(), "xx".to_string());
        tensors::save(&path, entry(), metadata).unwrap();
        assert!(Checkpoints::load(&path).is_err());
        cleanup(&path);
    }

    #[test]
    fn load_rejects_missing_snapshot_tensors() {
        // Metadata announces a snapshot whose weight tensors are absent, so the
        // model reconstruction must fail and propagate the error.
        use crate::io::tensors;
        use ndarray::array;
        use std::collections::HashMap;

        let path = temp_path("ckpt_bad_snapshot");
        let mut metadata = HashMap::new();
        metadata.insert("interval".to_string(), "1".to_string());
        metadata.insert("n_snapshots".to_string(), "1".to_string());
        metadata.insert("snapshot0.n_layers".to_string(), "1".to_string());
        let entries = vec![("placeholder".to_string(), tensors::tensor(&array![0.0_f32]))];
        tensors::save(&path, entries, metadata).unwrap();

        let result = Checkpoints::load(&path);
        cleanup(&path);
        assert!(result.is_err());
    }
}
