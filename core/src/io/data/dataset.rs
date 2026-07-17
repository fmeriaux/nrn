//! Parquet serializer for a [`Dataset`].
//!
//! Samples-major `inputs` (and rank-N `Value` targets) are stored as the canonical
//! [`arrow.fixed_shape_tensor`] extension: a `FixedSizeList<f32>` whose per-sample
//! shape rides in the column's `ARROW:extension:metadata`. Rank-1 targets become a
//! scalar `label` column — `Int64` for `ClassLabel`, `Float32` for `Value`.
//! `ClassLabel` names (when known) and the dataset's [`info`] both ride in the
//! schema-level metadata.
//!
//! [`arrow.fixed_shape_tensor`]: https://arrow.apache.org/docs/format/CanonicalExtensions.html#fixed-shape-tensor
//! [`info`]: Dataset::info

use crate::data::{Dataset, DatasetInfo, Targets};
use arrow::array::{Array, ArrayRef, FixedSizeListArray, Float32Array, Int64Array, RecordBatch};
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

/// Schema-metadata keys carrying the dataset's free-text information and, for
/// `ClassLabel` targets, their names.
const DESCRIPTION: &str = "description";
const CLASS_NAMES: &str = "class_names";

fn invalid<E: std::fmt::Display>(error: E) -> Error {
    Error::new(InvalidData, error.to_string())
}

fn info_into_metadata(info: Option<&DatasetInfo>) -> HashMap<String, String> {
    info.and_then(|info| info.description.clone())
        .map(|description| HashMap::from([(DESCRIPTION.to_string(), description)]))
        .unwrap_or_default()
}

fn info_from_metadata(metadata: &HashMap<String, String>) -> Option<DatasetInfo> {
    metadata.get(DESCRIPTION).map(|description| DatasetInfo {
        description: Some(description.clone()),
    })
}

fn class_names_into_metadata(names: Option<&[String]>, metadata: &mut HashMap<String, String>) {
    if let Some(names) = names
        && let Ok(encoded) = serde_json::to_string(names)
    {
        metadata.insert(CLASS_NAMES.to_string(), encoded);
    }
}

/// Reads the dataset's class names back from its schema metadata, best-effort: a
/// missing or malformed entry degrades to `None` rather than failing the load.
fn class_names_from_metadata(metadata: &HashMap<String, String>) -> Option<Vec<String>> {
    let encoded = metadata.get(CLASS_NAMES)?;
    serde_json::from_str(encoded).ok()
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

fn append_floats(column: &dyn Array, out: &mut Vec<f32>) -> Result<()> {
    let values = column
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| invalid("label column is not f32"))?;
    out.extend_from_slice(values.values());
    Ok(())
}

fn append_class_ids(column: &dyn Array, out: &mut Vec<u32>) -> Result<()> {
    let values = column
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| invalid("label column is not i64"))?;
    for &v in values.values() {
        out.push(u32::try_from(v).map_err(|_| invalid(format!("class id {v} is out of range")))?);
    }
    Ok(())
}

fn with_samples(n_samples: usize, per_sample: &[usize]) -> IxDyn {
    let mut shape = Vec::with_capacity(per_sample.len() + 1);
    shape.push(n_samples);
    shape.extend_from_slice(per_sample);
    IxDyn(&shape)
}

impl Dataset {
    /// Saves the dataset to a `.parquet` file, persisting its [`info`] (when
    /// recorded) and, for `ClassLabel` targets, their names, in the schema metadata.
    ///
    /// [`info`]: Dataset::info
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<PathBuf> {
        let (inputs_field, inputs_column) = tensor_column(INPUTS, self.features())?;
        let mut fields = vec![inputs_field];
        let mut columns = vec![inputs_column];

        let mut metadata = info_into_metadata(self.info());

        match self.targets() {
            Targets::ClassLabel(label) => {
                let ids: Vec<i64> = label.ids().iter().map(|&id| id as i64).collect();
                fields.push(Arc::new(Field::new(LABEL, DataType::Int64, false)));
                columns.push(Arc::new(Int64Array::from(ids)) as ArrayRef);
                class_names_into_metadata(label.names(), &mut metadata);
            }
            Targets::Value(values) => {
                let array = values.as_array();
                if array.ndim() == 1 {
                    let values: Vec<f32> = array.iter().copied().collect();
                    fields.push(Arc::new(Field::new(LABEL, DataType::Float32, false)));
                    columns.push(Arc::new(Float32Array::from(values)) as ArrayRef);
                } else {
                    let (field, column) = tensor_column(TARGETS, array)?;
                    fields.push(field);
                    columns.push(column);
                }
            }
        }

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
    /// restoring its recorded information and validating structure through
    /// [`Dataset::new`], so an ill-formed file is rejected at the I/O boundary.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Dataset> {
        let filepath = Path::combine_safe_with_cwd(path.as_ref().with_extension(EXTENSION))?;
        let file = File::open(&filepath)
            .map_err(|e| Error::new(e.kind(), format!("{}: {e}", filepath.display())))?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(invalid)?;
        let schema = builder.schema().clone();

        let inputs_index = schema.index_of(INPUTS).map_err(invalid)?;
        let inputs_shape = tensor_shape(schema.field(inputs_index))?;

        let layout = if let Ok(index) = schema.index_of(LABEL) {
            match schema.field(index).data_type() {
                DataType::Int64 => TargetsLayout::ClassLabel(index),
                DataType::Float32 => TargetsLayout::Value(index, Vec::new()),
                other => {
                    return Err(invalid(format!(
                        "label column has unexpected type {other:?}"
                    )));
                }
            }
        } else {
            let index = schema.index_of(TARGETS).map_err(invalid)?;
            TargetsLayout::Value(index, tensor_shape(schema.field(index))?)
        };

        let mut inputs_values = Vec::new();
        let mut class_ids = Vec::new();
        let mut target_values = Vec::new();
        let mut n_samples = 0;
        for batch in builder.build().map_err(invalid)? {
            let batch = batch.map_err(invalid)?;
            n_samples += batch.num_rows();
            append_tensor(batch.column(inputs_index).as_ref(), &mut inputs_values)?;
            match &layout {
                TargetsLayout::ClassLabel(index) => {
                    append_class_ids(batch.column(*index).as_ref(), &mut class_ids)?
                }
                TargetsLayout::Value(index, shape) if shape.is_empty() => {
                    append_floats(batch.column(*index).as_ref(), &mut target_values)?
                }
                TargetsLayout::Value(index, _) => {
                    append_tensor(batch.column(*index).as_ref(), &mut target_values)?
                }
            }
        }

        let inputs = ArrayD::from_shape_vec(with_samples(n_samples, &inputs_shape), inputs_values)
            .map_err(invalid)?;
        let targets = match &layout {
            TargetsLayout::ClassLabel(_) => {
                let names = class_names_from_metadata(schema.metadata());
                Targets::class_label(Array1::from(class_ids), names).map_err(invalid)?
            }
            TargetsLayout::Value(_, per_sample) => {
                let array =
                    ArrayD::from_shape_vec(with_samples(n_samples, per_sample), target_values)
                        .map_err(invalid)?;
                Targets::value(array).map_err(invalid)?
            }
        };
        let info = info_from_metadata(schema.metadata());

        Dataset::new(inputs, targets, info).map_err(invalid)
    }
}

/// How the target column is laid out in the file, as a schema column index.
enum TargetsLayout {
    /// The `label` column, `Int64`.
    ClassLabel(usize),
    /// A `Value` column: the `label` column (`Float32`, empty shape) or the
    /// `targets` `fixed_shape_tensor` column (its per-sample shape).
    Value(usize, Vec<usize>),
}

#[cfg(test)]
mod tests {
    use crate::data::{Dataset, DatasetInfo, Targets};
    use arrow::array::{ArrayRef, RecordBatch};
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

    fn dataset_with(info: Option<DatasetInfo>) -> Dataset {
        Dataset::tabular(
            Array2::from_shape_fn((5, 3), |(i, j)| (i * 3 + j) as f32 * 0.25),
            Targets::class_label(array![0u32, 1, 0, 2, 1], None).unwrap(),
            info,
        )
        .unwrap()
    }

    #[test]
    fn dataset_roundtrips_class_label_data_and_no_info() {
        let dataset = dataset_with(None);
        let path = temp_path("dataset");

        dataset.save(&path).unwrap();
        let loaded = Dataset::load(&path).unwrap();
        cleanup(&path);

        assert_eq!(dataset.features(), loaded.features());
        assert_eq!(dataset.targets(), loaded.targets());
        assert_eq!(loaded.info(), None);
    }

    #[test]
    fn dataset_roundtrips_class_names() {
        let names = vec!["cat".to_string(), "dog".to_string(), "bird".to_string()];
        let dataset = Dataset::tabular(
            Array2::from_shape_fn((5, 3), |(i, j)| (i * 3 + j) as f32 * 0.25),
            Targets::class_label(array![0u32, 1, 0, 2, 1], Some(names)).unwrap(),
            None,
        )
        .unwrap();
        let path = temp_path("dataset_class_names");

        dataset.save(&path).unwrap();
        let loaded = Dataset::load(&path).unwrap();
        cleanup(&path);

        assert_eq!(dataset.targets(), loaded.targets());
    }

    #[test]
    fn dataset_roundtrips_rank1_value_targets() {
        let dataset = Dataset::tabular(
            Array2::from_shape_fn((5, 3), |(i, j)| (i * 3 + j) as f32 * 0.25),
            Targets::value(array![0.1f32, 0.2, 0.3, 0.4, 0.5].into_dyn()).unwrap(),
            None,
        )
        .unwrap();

        let path = temp_path("dataset_value_rank1");
        dataset.save(&path).unwrap();
        let loaded = Dataset::load(&path).unwrap();
        cleanup(&path);

        assert_eq!(dataset.targets(), loaded.targets());
    }

    #[test]
    fn dataset_roundtrips_a_rank4_tensor_dataset_with_rank2_value_targets() {
        // Inputs shaped (samples, channels, height, width) and rank-2 `Value`
        // targets both ride the fixed_shape_tensor encoding.
        let inputs =
            ArrayD::from_shape_fn(IxDyn(&[6, 2, 3, 3]), |ix| (ix[0] * 10 + ix[3]) as f32 * 0.1);
        let targets = ArrayD::from_shape_fn(IxDyn(&[6, 2]), |ix| (ix[1] == ix[0] % 2) as u8 as f32);
        let dataset = Dataset::new(inputs, Targets::value(targets).unwrap(), None).unwrap();

        let path = temp_path("dataset_rank4");
        dataset.save(&path).unwrap();
        let loaded = Dataset::load(&path).unwrap();
        cleanup(&path);

        assert_eq!(dataset.features(), loaded.features());
        assert_eq!(dataset.targets(), loaded.targets());
    }

    #[test]
    fn dataset_roundtrips_an_info_description() {
        let info = DatasetInfo {
            description: Some("Encoded from images/digits".to_string()),
        };
        let path = temp_path("dataset_info");

        dataset_with(Some(info.clone())).save(&path).unwrap();
        let loaded = Dataset::load(&path).unwrap();
        cleanup(&path);

        assert_eq!(loaded.info(), Some(&info));
    }

    #[test]
    fn malformed_class_names_metadata_degrades_to_none() {
        use super::class_names_from_metadata;
        use std::collections::HashMap;

        let cases = [
            ("missing", HashMap::new()),
            (
                "not json",
                HashMap::from([(super::CLASS_NAMES.to_string(), "not json".to_string())]),
            ),
        ];

        for (label, metadata) in cases {
            assert_eq!(class_names_from_metadata(&metadata), None, "{label}");
        }
    }

    #[test]
    fn load_rejects_corrupt_file() {
        let path = temp_path("data_corrupt");
        std::fs::write(path.with_extension("parquet"), b"not a parquet file").unwrap();

        assert!(Dataset::load(&path).is_err());

        cleanup(&path);
    }

    #[test]
    fn tensor_shape_rejects_a_field_without_extension_metadata() {
        use arrow::datatypes::{DataType, Field};

        let field = Field::new(super::INPUTS, DataType::Float32, false);
        let err = super::tensor_shape(&field).unwrap_err();
        assert!(err.to_string().contains(super::EXT_METADATA), "got: {err}");
    }

    #[test]
    fn load_rejects_an_unexpected_label_column_type() {
        use arrow::array::StringArray;
        use arrow::datatypes::{DataType, Field, Schema};
        use parquet::arrow::ArrowWriter;
        use std::sync::Arc;

        let inputs = Array2::<f32>::zeros((2, 3)).into_dyn();
        let (inputs_field, inputs_column) = super::tensor_column(super::INPUTS, &inputs).unwrap();
        let label_field = Arc::new(Field::new(super::LABEL, DataType::Utf8, false));
        let label_column = Arc::new(StringArray::from(vec!["a", "b"])) as ArrayRef;

        let schema = Arc::new(Schema::new(vec![inputs_field, label_field]));
        let batch =
            RecordBatch::try_new(schema.clone(), vec![inputs_column, label_column]).unwrap();

        let path = temp_path("data_bad_label_type");
        let filepath = path.with_extension("parquet");
        let mut buffer = Vec::new();
        let mut writer = ArrowWriter::try_new(&mut buffer, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        std::fs::write(&filepath, buffer).unwrap();

        let err = Dataset::load(&path).err().unwrap();
        assert!(err.to_string().contains("unexpected type"), "got: {err}");

        cleanup(&path);
    }

    #[test]
    fn load_rejects_a_class_id_outside_u32_range() {
        use arrow::array::Int64Array;
        use arrow::datatypes::{DataType, Field, Schema};
        use parquet::arrow::ArrowWriter;
        use std::sync::Arc;

        let inputs = Array2::<f32>::zeros((2, 3)).into_dyn();
        let (inputs_field, inputs_column) = super::tensor_column(super::INPUTS, &inputs).unwrap();
        let label_field = Arc::new(Field::new(super::LABEL, DataType::Int64, false));
        let label_column = Arc::new(Int64Array::from(vec![0i64, -1])) as ArrayRef;

        let schema = Arc::new(Schema::new(vec![inputs_field, label_field]));
        let batch =
            RecordBatch::try_new(schema.clone(), vec![inputs_column, label_column]).unwrap();

        let path = temp_path("data_bad_class_id");
        let filepath = path.with_extension("parquet");
        let mut buffer = Vec::new();
        let mut writer = ArrowWriter::try_new(&mut buffer, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        std::fs::write(&filepath, buffer).unwrap();

        let err = Dataset::load(&path).err().unwrap();
        assert!(err.to_string().contains("out of range"), "got: {err}");

        cleanup(&path);
    }
}
