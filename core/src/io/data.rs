use crate::data::{Dataset, DatasetOrigin};
use crate::io::tensors;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::io::ErrorKind::InvalidData;
use std::io::{Error, Result};
use std::path::{Path, PathBuf};

fn invalid<E: std::fmt::Display>(error: E) -> Error {
    Error::new(InvalidData, error.to_string())
}

// safetensors tensor names.
const FEATURES: &str = "features";
const LABELS: &str = "labels";

// `__metadata__` keys carrying the dataset's origin. The wire format is an I/O
// concern, so the keys and discriminant live here; the `DatasetOrigin` type
// itself stays serde-free in `crate::data`.
const ORIGIN_KIND: &str = "origin";
const ORIGIN_DISTRIBUTION: &str = "distribution";
const ORIGIN_SOURCE: &str = "source";
const ORIGIN_SEED: &str = "seed";
const KIND_SYNTHETIC: &str = "synthetic";
const KIND_ENCODED: &str = "encoded";

fn origin_into_metadata(origin: &DatasetOrigin) -> HashMap<String, String> {
    match origin {
        DatasetOrigin::Synthetic { distribution, seed } => HashMap::from([
            (ORIGIN_KIND.to_string(), KIND_SYNTHETIC.to_string()),
            (ORIGIN_DISTRIBUTION.to_string(), distribution.clone()),
            (ORIGIN_SEED.to_string(), seed.to_string()),
        ]),
        DatasetOrigin::Encoded { source, seed } => HashMap::from([
            (ORIGIN_KIND.to_string(), KIND_ENCODED.to_string()),
            (ORIGIN_SOURCE.to_string(), source.clone()),
            (ORIGIN_SEED.to_string(), seed.to_string()),
        ]),
    }
}

fn origin_from_metadata(metadata: &HashMap<String, String>) -> Option<DatasetOrigin> {
    match metadata.get(ORIGIN_KIND)?.as_str() {
        KIND_SYNTHETIC => Some(DatasetOrigin::Synthetic {
            distribution: metadata.get(ORIGIN_DISTRIBUTION)?.clone(),
            seed: metadata.get(ORIGIN_SEED)?.parse().ok()?,
        }),
        KIND_ENCODED => Some(DatasetOrigin::Encoded {
            source: metadata.get(ORIGIN_SOURCE)?.clone(),
            seed: metadata.get(ORIGIN_SEED)?.parse().ok()?,
        }),
        _ => None,
    }
}

impl Dataset {
    /// Saves the dataset to a `.safetensors` file, persisting its [`origin`] (when
    /// recorded) in the `__metadata__` map.
    ///
    /// [`origin`]: Dataset::origin
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<PathBuf> {
        let entries = vec![
            (FEATURES.to_string(), tensors::tensor(self.features())),
            (LABELS.to_string(), tensors::tensor(self.labels())),
        ];
        let metadata = self.origin().map(origin_into_metadata).unwrap_or_default();
        tensors::save(path, entries, metadata)
    }

    /// Loads a dataset from a `.safetensors` file, restoring its origin and
    /// validating the tensors through [`Dataset::new`], so an ill-formed file is
    /// rejected at the I/O boundary.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Dataset> {
        let bytes = tensors::load(path)?;
        let st = SafeTensors::deserialize(&bytes).map_err(invalid)?;

        let features = tensors::read_array2(FEATURES, &st)?;
        let labels = tensors::read_array1(LABELS, &st)?;
        let origin = origin_from_metadata(&tensors::read_metadata(&bytes)?);

        Dataset::new(features, labels, origin).map_err(invalid)
    }
}

#[cfg(test)]
mod tests {
    use crate::data::{Dataset, DatasetOrigin};
    use ndarray::{Array2, array};
    use std::path::{Path, PathBuf};

    fn temp_path(tag: &str) -> PathBuf {
        let dir = PathBuf::from("target/nrn_tests");
        std::fs::create_dir_all(&dir).ok();
        dir.join(format!("nrn_test_{tag}_{}", std::process::id()))
    }

    fn cleanup(path: &Path) {
        let _ = std::fs::remove_file(path.with_extension("safetensors"));
    }

    fn dataset_with(origin: Option<DatasetOrigin>) -> Dataset {
        Dataset::new(
            Array2::from_shape_fn((5, 3), |(i, j)| (i * 3 + j) as f32 * 0.25),
            array![0.0, 1.0, 0.0, 2.0, 1.0],
            origin,
        )
        .unwrap()
    }

    #[test]
    fn dataset_roundtrips_data_and_no_origin() {
        let dataset = dataset_with(None);
        let path = temp_path("dataset");

        dataset.save(&path).unwrap();
        let loaded = Dataset::load(&path).unwrap();
        cleanup(&path);

        assert_eq!(dataset.features(), loaded.features());
        assert_eq!(dataset.labels(), loaded.labels());
        assert_eq!(loaded.origin(), None);
    }

    #[test]
    fn dataset_roundtrips_a_synthetic_origin() {
        let origin = DatasetOrigin::Synthetic {
            distribution: "spiral".to_string(),
            seed: 42,
        };
        let path = temp_path("dataset_synthetic");

        dataset_with(Some(origin.clone())).save(&path).unwrap();
        let loaded = Dataset::load(&path).unwrap();
        cleanup(&path);

        assert_eq!(loaded.origin(), Some(&origin));
    }

    #[test]
    fn dataset_roundtrips_an_encoded_origin() {
        let origin = DatasetOrigin::Encoded {
            source: "images/digits".to_string(),
            seed: 7,
        };
        let path = temp_path("dataset_encoded");

        dataset_with(Some(origin.clone())).save(&path).unwrap();
        let loaded = Dataset::load(&path).unwrap();
        cleanup(&path);

        assert_eq!(loaded.origin(), Some(&origin));
    }

    #[test]
    fn malformed_or_unknown_origin_metadata_degrades_to_no_origin() {
        // A recorded origin is best-effort: an unknown discriminant or a missing /
        // unparseable field drops the origin rather than failing the load.
        use super::{
            KIND_ENCODED, KIND_SYNTHETIC, ORIGIN_DISTRIBUTION, ORIGIN_KIND, ORIGIN_SEED,
            origin_from_metadata,
        };
        use std::collections::HashMap;

        let cases = [
            ("no kind", HashMap::new()),
            (
                "unknown kind",
                HashMap::from([(ORIGIN_KIND.to_string(), "mystery".to_string())]),
            ),
            (
                "synthetic without distribution",
                HashMap::from([
                    (ORIGIN_KIND.to_string(), KIND_SYNTHETIC.to_string()),
                    (ORIGIN_SEED.to_string(), "7".to_string()),
                ]),
            ),
            (
                "synthetic without seed",
                HashMap::from([
                    (ORIGIN_KIND.to_string(), KIND_SYNTHETIC.to_string()),
                    (ORIGIN_DISTRIBUTION.to_string(), "spiral".to_string()),
                ]),
            ),
            (
                "synthetic with non-numeric seed",
                HashMap::from([
                    (ORIGIN_KIND.to_string(), KIND_SYNTHETIC.to_string()),
                    (ORIGIN_DISTRIBUTION.to_string(), "spiral".to_string()),
                    (ORIGIN_SEED.to_string(), "not-a-number".to_string()),
                ]),
            ),
            (
                "encoded without source",
                HashMap::from([(ORIGIN_KIND.to_string(), KIND_ENCODED.to_string())]),
            ),
        ];

        for (label, metadata) in cases {
            assert_eq!(origin_from_metadata(&metadata), None, "{label}");
        }
    }

    #[test]
    fn load_rejects_corrupt_file() {
        let path = temp_path("data_corrupt");
        std::fs::write(
            path.with_extension("safetensors"),
            b"not a safetensors buffer",
        )
        .unwrap();

        assert!(Dataset::load(&path).is_err());

        cleanup(&path);
    }

    #[test]
    fn load_rejects_tensors_that_violate_dataset_invariants() {
        use super::{FEATURES, LABELS};
        use crate::io::tensors;
        use std::collections::HashMap;

        // The tensors are individually well-formed, but two feature rows against
        // three labels is a shape mismatch that `Dataset::new` rejects — so the
        // load must fail at the I/O boundary rather than yield a bad dataset.
        let path = temp_path("data_invariant");
        tensors::save(
            &path,
            vec![
                (
                    FEATURES.to_string(),
                    tensors::tensor(&array![[0.0_f32, 1.0, 2.0], [3.0, 4.0, 5.0]]),
                ),
                (
                    LABELS.to_string(),
                    tensors::tensor(&array![0.0_f32, 1.0, 0.0]),
                ),
            ],
            HashMap::new(),
        )
        .unwrap();

        assert!(Dataset::load(&path).is_err());
        cleanup(&path);
    }
}
