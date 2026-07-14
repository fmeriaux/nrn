//! Parquet serializer for a [`Dataset`].
//!
//! Samples-major `inputs` (and rank-N `targets`) are stored as the canonical
//! [`arrow.fixed_shape_tensor`] extension: a `FixedSizeList<f32>` whose per-sample
//! shape rides in the column's `ARROW:extension:metadata`. Rank-1 `targets` become
//! a `label: Float32` column. The dataset's [`origin`] travels in the schema-level
//! metadata.
//!
//! [`arrow.fixed_shape_tensor`]: https://arrow.apache.org/docs/format/CanonicalExtensions.html#fixed-shape-tensor
//! [`origin`]: Dataset::origin

use crate::data::{Dataset, DatasetOrigin};
use arrow::array::{Array, ArrayRef, FixedSizeListArray, Float32Array, RecordBatch};
use arrow::datatypes::{DataType, Field, FieldRef, Schema};
use ndarray::{Array1, ArrayD, IxDyn};
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::ErrorKind::InvalidData;
use std::io::{Error, Result};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::io::path::PathExt;

/// File extension for every dataset artifact.
const EXTENSION: &str = "parquet";

/// Arrow column names.
const INPUTS: &str = "inputs";
const TARGETS: &str = "targets";
const LABEL: &str = "label";

/// Canonical extension keys marking a `FixedSizeList` column as a tensor.
const EXT_NAME: &str = "ARROW:extension:name";
const EXT_METADATA: &str = "ARROW:extension:metadata";
const FIXED_SHAPE_TENSOR: &str = "arrow.fixed_shape_tensor";

fn invalid<E: std::fmt::Display>(error: E) -> Error {
    Error::new(InvalidData, error.to_string())
}

// Schema-metadata keys carrying the dataset's origin. The wire format is an I/O
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
        DatasetOrigin::Encoded { source } => HashMap::from([
            (ORIGIN_KIND.to_string(), KIND_ENCODED.to_string()),
            (ORIGIN_SOURCE.to_string(), source.clone()),
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
        }),
        _ => None,
    }
}

/// The per-sample shape carried by a tensor column's extension metadata.
#[derive(Serialize, Deserialize)]
struct TensorExtension {
    shape: Vec<usize>,
}

/// Builds a `fixed_shape_tensor` column from a samples-major array: the leading
/// axis becomes the rows, the remaining axes the per-sample shape.
fn tensor_column(name: &str, array: &ArrayD<f32>) -> Result<(FieldRef, ArrayRef)> {
    let per_sample: Vec<usize> = array.shape()[1..].to_vec();
    let size = per_sample.iter().product::<usize>() as i32;

    let standard = array.as_standard_layout();
    let values: Vec<f32> = standard.iter().copied().collect();

    let item = Arc::new(Field::new("item", DataType::Float32, false));
    let list = FixedSizeListArray::new(
        item.clone(),
        size,
        Arc::new(Float32Array::from(values)),
        None,
    );

    let extension =
        serde_json::to_string(&TensorExtension { shape: per_sample }).map_err(invalid)?;
    let metadata = HashMap::from([
        (EXT_NAME.to_string(), FIXED_SHAPE_TENSOR.to_string()),
        (EXT_METADATA.to_string(), extension),
    ]);
    let field = Arc::new(
        Field::new(name, DataType::FixedSizeList(item, size), false).with_metadata(metadata),
    );

    Ok((field, Arc::new(list) as ArrayRef))
}

/// Reads a tensor column's per-sample shape back from its extension metadata.
fn tensor_shape(field: &Field) -> Result<Vec<usize>> {
    let raw = field.metadata().get(EXT_METADATA).ok_or_else(|| {
        invalid(format!(
            "column `{}` is missing `{EXT_METADATA}`",
            field.name()
        ))
    })?;
    let extension: TensorExtension = serde_json::from_str(raw).map_err(invalid)?;
    Ok(extension.shape)
}

/// Appends the flattened `f32` values of a `fixed_shape_tensor` column to `out`.
fn append_tensor(column: &dyn Array, out: &mut Vec<f32>) -> Result<()> {
    let list = column
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .ok_or_else(|| invalid("expected a fixed-size-list column"))?;
    let values = list
        .values()
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| invalid("tensor column is not f32"))?;
    out.extend_from_slice(values.values());
    Ok(())
}

/// Appends the `f32` values of a scalar column to `out`.
fn append_floats(column: &dyn Array, out: &mut Vec<f32>) -> Result<()> {
    let values = column
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| invalid("label column is not f32"))?;
    out.extend_from_slice(values.values());
    Ok(())
}

/// Prepends the sample count to a per-sample shape.
fn with_samples(n_samples: usize, per_sample: &[usize]) -> IxDyn {
    let mut shape = Vec::with_capacity(per_sample.len() + 1);
    shape.push(n_samples);
    shape.extend_from_slice(per_sample);
    IxDyn(&shape)
}

impl Dataset {
    /// Saves the dataset to a `.parquet` file, persisting its [`origin`] (when
    /// recorded) in the schema metadata.
    ///
    /// [`origin`]: Dataset::origin
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<PathBuf> {
        let (inputs_field, inputs_column) = tensor_column(INPUTS, self.inputs())?;
        let mut fields = vec![inputs_field];
        let mut columns = vec![inputs_column];

        let targets = self.targets();
        if targets.ndim() == 1 {
            let labels: Vec<f32> = targets.iter().copied().collect();
            fields.push(Arc::new(Field::new(LABEL, DataType::Float32, false)));
            columns.push(Arc::new(Float32Array::from(labels)) as ArrayRef);
        } else {
            let (field, column) = tensor_column(TARGETS, targets)?;
            fields.push(field);
            columns.push(column);
        }

        let metadata = self.origin().map(origin_into_metadata).unwrap_or_default();
        let schema = Arc::new(Schema::new_with_metadata(fields, metadata));
        let batch = RecordBatch::try_new(schema.clone(), columns).map_err(invalid)?;

        let filepath = path.as_ref().with_extension(EXTENSION);
        let filepath = Path::combine_safe_with_cwd(filepath)?;
        filepath.create_parents()?;

        let mut buffer = Vec::new();
        let mut writer = ArrowWriter::try_new(&mut buffer, schema, None).map_err(invalid)?;
        writer.write(&batch).map_err(invalid)?;
        writer.close().map_err(invalid)?;
        fs::write(&filepath, buffer)?;

        Ok(filepath)
    }

    /// Loads a dataset from a `.parquet` file written by [`Dataset::save`],
    /// restoring its origin and validating structure through [`Dataset::new`], so
    /// an ill-formed file is rejected at the I/O boundary.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Dataset> {
        let filepath = Path::combine_safe_with_cwd(path.as_ref().with_extension(EXTENSION))?;
        let file = File::open(&filepath)
            .map_err(|e| Error::new(e.kind(), format!("{}: {e}", filepath.display())))?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(invalid)?;
        let schema = builder.schema().clone();

        let inputs_index = schema.index_of(INPUTS).map_err(invalid)?;
        let inputs_shape = tensor_shape(schema.field(inputs_index))?;

        // Targets are either a scalar `label` column or a rank-N tensor column.
        let targets = match schema.index_of(LABEL) {
            Ok(index) => Targets::Label(index),
            Err(_) => {
                let index = schema.index_of(TARGETS).map_err(invalid)?;
                Targets::Tensor(index, tensor_shape(schema.field(index))?)
            }
        };

        let mut inputs_values = Vec::new();
        let mut targets_values = Vec::new();
        let mut n_samples = 0;
        for batch in builder.build().map_err(invalid)? {
            let batch = batch.map_err(invalid)?;
            n_samples += batch.num_rows();
            append_tensor(batch.column(inputs_index).as_ref(), &mut inputs_values)?;
            match &targets {
                Targets::Label(index) => {
                    append_floats(batch.column(*index).as_ref(), &mut targets_values)?
                }
                Targets::Tensor(index, _) => {
                    append_tensor(batch.column(*index).as_ref(), &mut targets_values)?
                }
            }
        }

        let inputs = ArrayD::from_shape_vec(with_samples(n_samples, &inputs_shape), inputs_values)
            .map_err(invalid)?;
        let targets = match &targets {
            Targets::Label(_) => Array1::from(targets_values).into_dyn(),
            Targets::Tensor(_, per_sample) => {
                ArrayD::from_shape_vec(with_samples(n_samples, per_sample), targets_values)
                    .map_err(invalid)?
            }
        };
        let origin = origin_from_metadata(schema.metadata());

        Dataset::new(inputs, targets, origin).map_err(invalid)
    }
}

/// How the target column is laid out in the file.
enum Targets {
    /// A scalar `label` column at the given index.
    Label(usize),
    /// A `fixed_shape_tensor` column at the given index, with its per-sample shape.
    Tensor(usize, Vec<usize>),
}

#[cfg(test)]
mod tests {
    use crate::data::{Dataset, DatasetOrigin};
    use ndarray::{Array2, ArrayD, IxDyn, array};
    use std::path::{Path, PathBuf};

    fn temp_path(tag: &str) -> PathBuf {
        let dir = PathBuf::from("target/nrn_tests");
        std::fs::create_dir_all(&dir).ok();
        dir.join(format!("nrn_test_{tag}_{}", std::process::id()))
    }

    fn cleanup(path: &Path) {
        let _ = std::fs::remove_file(path.with_extension("parquet"));
    }

    fn dataset_with(origin: Option<DatasetOrigin>) -> Dataset {
        Dataset::tabular(
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

        assert_eq!(dataset.inputs(), loaded.inputs());
        assert_eq!(dataset.targets(), loaded.targets());
        assert_eq!(loaded.origin(), None);
    }

    #[test]
    fn dataset_roundtrips_a_rank4_tensor_dataset() {
        // Inputs shaped (samples, channels, height, width) and one-hot rank-2
        // targets both ride the fixed_shape_tensor encoding.
        let inputs =
            ArrayD::from_shape_fn(IxDyn(&[6, 2, 3, 3]), |ix| (ix[0] * 10 + ix[3]) as f32 * 0.1);
        let targets = ArrayD::from_shape_fn(IxDyn(&[6, 2]), |ix| (ix[1] == ix[0] % 2) as u8 as f32);
        let dataset = Dataset::new(inputs, targets, None).unwrap();

        let path = temp_path("dataset_rank4");
        dataset.save(&path).unwrap();
        let loaded = Dataset::load(&path).unwrap();
        cleanup(&path);

        assert_eq!(dataset.inputs(), loaded.inputs());
        assert_eq!(dataset.targets(), loaded.targets());
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
        std::fs::write(path.with_extension("parquet"), b"not a parquet file").unwrap();

        assert!(Dataset::load(&path).is_err());

        cleanup(&path);
    }
}
