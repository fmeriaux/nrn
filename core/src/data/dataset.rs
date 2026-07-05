use crate::data::origin::DatasetOrigin;
use crate::data::scalers::{Scaler, ScalerFeatureMismatch, ScalerKind, ScalerMethod};
use ndarray::{Array, Array1, Array2, ArrayD, Axis, Dimension, Ix2};
use ndarray_rand::rand::Rng;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand::prelude::{SliceRandom, StdRng};
use std::collections::HashSet;
use std::error::Error;

/// A dataset containing features and labels for classification tasks.
///
/// Construction goes exclusively through [`Dataset::new`], the single boundary
/// where structural and label invariants are enforced. The fields are private so
/// an invalid `Dataset` cannot be represented: every value of this type is
/// guaranteed non-empty, shape-consistent, and labelled with contiguous
/// 0-indexed class ids over at least two classes.
#[derive(Clone)]
pub struct Dataset {
    /// A 2D array where each row is a sample and each column is a feature
    features: Array2<f32>,
    /// A 1D array where each element is the label for the corresponding sample
    labels: Array1<f32>,
    /// Where the dataset came from, when known. Optional provenance that travels
    /// with the data but plays no part in its validity.
    origin: Option<DatasetOrigin>,
}

pub struct ModelDataset {
    /// A samples-last array: the leading axes are the per-sample features and the
    /// trailing axis indexes samples. Rank-2 `(features, samples)` for tabular data,
    /// higher rank for spatial data.
    inputs: ArrayD<f32>,
    /// A 2D array where each column is a sample and each row is a target (one-hot encoded for multi-class)
    targets: Array2<f32>,
}

/// A structure representing a split dataset for training, validation, and testing.
pub struct ModelSplit {
    /// A `Dataset` containing the training samples.
    pub train: ModelDataset,
    /// An optional `Dataset` containing the validation samples.
    pub validation: Option<ModelDataset>,
    /// A `Dataset` containing the testing samples.
    pub test: ModelDataset,
}

/// Errors returned when features and labels cannot form a valid [`Dataset`].
#[derive(Debug, PartialEq, Eq)]
pub enum DatasetError {
    /// The dataset has no features.
    NoFeatures,
    /// The dataset has no samples.
    NoSamples,
    /// Features and labels disagree on the sample count.
    ShapeMismatch { features: usize, labels: usize },
    /// Fewer than two distinct classes are present (carries the count found).
    TooFewClasses(usize),
    /// Labels are not contiguous 0-indexed class ids in `[0, n_classes)`.
    InvalidLabels,
}

impl std::fmt::Display for DatasetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DatasetError::NoFeatures => write!(f, "dataset has no features"),
            DatasetError::NoSamples => write!(f, "dataset has no samples"),
            DatasetError::ShapeMismatch { features, labels } => write!(
                f,
                "features and labels disagree on sample count: {features} rows vs {labels} labels"
            ),
            DatasetError::TooFewClasses(n) => {
                write!(f, "a classifier needs at least 2 classes, but found {n}")
            }
            DatasetError::InvalidLabels => write!(
                f,
                "labels must be contiguous 0-indexed class ids in [0, n_classes)"
            ),
        }
    }
}

impl Error for DatasetError {}

impl Dataset {
    /// Builds a validated dataset from raw `features` (rows = samples, columns =
    /// features) and `labels`.
    ///
    /// This is the single boundary where dataset invariants are enforced, so every
    /// downstream consumer — the layer-spec constructors
    /// ([`crate::model::NeuronLayerSpec::output_for`],
    /// [`crate::model::NeuronLayerSpec::infer_from`]) and [`Self::to_model_dataset`] —
    /// can assume valid input and stay infallible.
    ///
    /// # Errors
    /// - [`DatasetError::ShapeMismatch`] when the feature rows and labels disagree
    ///   on the sample count.
    /// - [`DatasetError::NoFeatures`] when the dataset has zero features.
    /// - [`DatasetError::NoSamples`] when the dataset has zero samples.
    /// - [`DatasetError::TooFewClasses`] when fewer than two classes are present.
    /// - [`DatasetError::InvalidLabels`] when labels are not contiguous 0-indexed
    ///   class ids in `[0, n_classes)` (this also rejects non-integer labels).
    pub fn new(
        features: Array2<f32>,
        labels: Array1<f32>,
        origin: Option<DatasetOrigin>,
    ) -> Result<Self, DatasetError> {
        if features.nrows() != labels.len() {
            return Err(DatasetError::ShapeMismatch {
                features: features.nrows(),
                labels: labels.len(),
            });
        }
        if features.ncols() == 0 {
            return Err(DatasetError::NoFeatures);
        }
        if labels.is_empty() {
            return Err(DatasetError::NoSamples);
        }

        let dataset = Self {
            features,
            labels,
            origin,
        };

        let n_classes = dataset.n_classes();
        if n_classes < 2 {
            return Err(DatasetError::TooFewClasses(n_classes));
        }
        // Contiguous 0-indexed ids: with exactly `n_classes` distinct values, every
        // label lying in `[0, n_classes)` (and integral) forces the set to be
        // {0, 1, …, n_classes-1}. This subsumes the binary {0, 1} requirement.
        let valid_labels = dataset
            .labels
            .iter()
            .all(|&l| l >= 0.0 && l.fract() == 0.0 && (l as usize) < n_classes);
        if !valid_labels {
            return Err(DatasetError::InvalidLabels);
        }

        Ok(dataset)
    }

    /// Returns the dataset's recorded origin, if any.
    pub fn origin(&self) -> Option<&DatasetOrigin> {
        self.origin.as_ref()
    }

    /// A deterministic identifier for this dataset, pairing its origin with its
    /// shape (classes, features, samples). Two datasets reproduced from the same
    /// origin share it; datasets with no recorded origin fall back to a neutral
    /// prefix.
    pub fn id(&self) -> String {
        let origin = self
            .origin
            .as_ref()
            .map(DatasetOrigin::label)
            .unwrap_or_else(|| "dataset".to_string());
        format!(
            "{origin}-c{}-f{}-n{}",
            self.n_classes(),
            self.n_features(),
            self.n_samples()
        )
    }

    /// Returns the features, with one row per sample and one column per feature.
    pub fn features(&self) -> &Array2<f32> {
        &self.features
    }

    /// Returns the per-sample labels.
    pub fn labels(&self) -> &Array1<f32> {
        &self.labels
    }

    /// Returns the number of samples in the dataset.
    pub fn n_samples(&self) -> usize {
        self.labels.len()
    }

    /// Returns the number of features in the dataset.
    pub fn n_features(&self) -> usize {
        self.features.ncols()
    }

    /// Returns the number of unique classes (labels) in the dataset.
    pub fn n_classes(&self) -> usize {
        self.unique_labels().len()
    }

    /// Returns a vector of unique labels present in the dataset.
    /// # Returns
    /// A vector containing all unique label values found in the `labels` array.
    /// # Notes
    /// - The order of labels in the returned vector is not guaranteed.
    pub fn unique_labels(&self) -> Vec<f32> {
        let set: HashSet<u32> = self.labels.iter().map(|&label| label.to_bits()).collect();

        set.into_iter().map(f32::from_bits).collect()
    }

    /// Returns all feature rows corresponding to a specific label.
    /// # Arguments
    /// - `label`: The label value for which to retrieve the feature rows.
    /// # Returns
    /// A 2D array containing all feature rows where the corresponding label matches the input.
    pub fn get_features_for_label(&self, label: f32) -> Array2<f32> {
        let indices: Vec<usize> = self
            .labels
            .iter()
            .enumerate()
            .filter_map(|(i, &l)| if l == label { Some(i) } else { None })
            .collect();

        self.features.select(Axis(0), &indices).to_owned()
    }

    /// Computes the minimum and maximum values for each feature in the dataset.
    ///
    /// A [`Dataset`] always has at least one sample (enforced by [`Dataset::new`]),
    /// so the range is always defined.
    /// # Returns
    /// A tuple of two vectors:
    /// - The first vector contains the minimum values for each feature.
    /// - The second vector contains the maximum values for each feature.
    pub fn feature_range(&self) -> (Vec<f32>, Vec<f32>) {
        let n_features = self.n_features();

        let mut mins = vec![f32::INFINITY; n_features];
        let mut maxs = vec![f32::NEG_INFINITY; n_features];

        for (i, feature) in self.features.axis_iter(Axis(1)).enumerate() {
            for &value in feature.iter() {
                mins[i] = mins[i].min(value);
                maxs[i] = maxs[i].max(value);
            }
        }

        (mins, maxs)
    }

    /// Transforms the dataset into a shape suitable for model input and output.
    ///
    /// # Returns
    /// A `ModelDataset` struct containing:
    /// - `inputs` is a 2D array view of shape `(n_features, n_samples)`.
    /// - `targets` is a 2D array owned of shape `(n_classes, n_samples)` for multi-class (one-hot encoded labels),
    ///   or `(1, n_samples)` for binary labels.
    ///
    /// # Details
    /// - If the labels are not binary (i.e., `max_label > 1`), the function one-hot encodes the labels.
    /// - For binary labels, it simply adds an axis to match the expected shape.
    ///
    pub fn to_model_dataset(&self) -> ModelDataset {
        let inputs = self.features.t().to_owned();

        let n_classes = self.n_classes();

        // Label validity (contiguous 0-indexed ids, binary ⊆ {0, 1}) is guaranteed
        // by [`Self::new`], so no re-check is needed here.
        let targets: Array2<f32> = if n_classes > 2 {
            let mut one_hot = Array2::zeros((n_classes, self.n_samples()));
            for (i, &label) in self.labels.iter().enumerate() {
                one_hot[[label as usize, i]] = 1.0;
            }
            one_hot
        } else {
            // Binary labels (0.0 / 1.0) are used directly as a single target row.
            self.labels.to_owned().insert_axis(Axis(0))
        };

        ModelDataset::new(inputs, targets)
    }

    /// Builds a dataset from encoded `images` and their `labels`, stamped with
    /// [`DatasetOrigin::Encoded`] provenance. Samples are kept in the order they
    /// were encoded (grouped by class); shuffling is deferred to
    /// [`ModelDataset::split`], so the stored order carries no randomness.
    /// # Arguments
    /// - `source`: Name of the source the images were encoded from.
    /// - `images`: A vector of images represented as 1D arrays of pixel values.
    /// - `labels`: A vector of labels corresponding to each image.
    pub fn from_encoded(
        source: impl Into<String>,
        images: Vec<Array1<f32>>,
        labels: Vec<usize>,
    ) -> Result<Self, Box<dyn Error>> {
        if images.is_empty() {
            return Err(DatasetError::NoSamples.into());
        }

        let features: Array2<f32> = Array2::from_shape_vec(
            (images.len(), images[0].len()),
            images.into_iter().flatten().collect(),
        )?;

        let labels: Array1<f32> =
            Array1::from(labels.into_iter().map(|x| x as f32).collect::<Vec<_>>());

        let origin = DatasetOrigin::Encoded {
            source: source.into(),
        };

        Ok(Dataset::new(features, labels, Some(origin))?)
    }

    /// Shuffles the samples (seeded by `seed`) and partitions them into training,
    /// validation, and testing sets by the given ratios, each as a [`ModelDataset`].
    ///
    /// # Parameters
    /// - `val_ratio`: The ratio of the dataset to be used for validation (between 0 and 1).
    /// - `test_ratio`: The ratio of the dataset to be used for testing (between 0 and 1).
    /// - `seed`: Seeds the shuffle, so the partition is reproducible from it.
    ///
    /// # Panics
    /// - When `val_ratio` is not between 0 and 1.
    /// - When `test_ratio` is not between 0 and 1 or is equal to 0.
    /// - When the sum of `val_ratio` and `test_ratio` is greater than or equal to 1.
    //
    pub fn split(&self, val_ratio: f32, test_ratio: f32, seed: u64) -> ModelSplit {
        assert!(
            (0.0..1.0).contains(&val_ratio),
            "Validation ratio must be between 0 and 1"
        );

        assert!(
            (0.0..1.0).contains(&test_ratio) && test_ratio > 0.0,
            "Test ratio must be between 0 and 1 and greater than 0"
        );

        assert!(
            val_ratio + test_ratio < 1.0,
            "Sum of ratios must be less than 1"
        );

        let model = self.to_model_dataset();
        let n_samples = model.targets.ncols();

        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.shuffle(&mut StdRng::seed_from_u64(seed));

        let size = |ratio: f32| (n_samples as f32 * ratio).round() as usize;
        let select = |idx: &[usize]| {
            ModelDataset::new(
                model.select_samples(idx),
                model.targets.select(Axis(1), idx),
            )
        };

        let test_size = size(test_ratio);
        let val_size = size(val_ratio);
        let train_size = n_samples - test_size - val_size;

        ModelSplit {
            train: select(&indices[..train_size]),
            validation: if val_size > 0 {
                Some(select(&indices[train_size..train_size + val_size]))
            } else {
                None
            },
            test: select(&indices[train_size + val_size..]),
        }
    }
}

impl ModelDataset {
    /// Pairs samples-last `inputs` (trailing axis indexes samples) with their
    /// `targets` `(targets, samples)`. `inputs` may be any rank: rank-2
    /// `(features, samples)` for tabular data, higher rank for spatial data.
    ///
    /// # Panics
    /// - When `inputs` and `targets` disagree on the sample count (`inputs`'
    ///   trailing axis versus `targets`' columns).
    pub fn new<D: Dimension>(inputs: Array<f32, D>, targets: Array2<f32>) -> Self {
        let inputs = inputs.into_dyn();
        assert_eq!(
            inputs.shape()[inputs.ndim() - 1],
            targets.ncols(),
            "Inputs and targets must have the same number of samples."
        );
        ModelDataset { inputs, targets }
    }

    /// Returns the inputs, samples-last (trailing axis indexes samples).
    pub fn inputs(&self) -> &ArrayD<f32> {
        &self.inputs
    }

    /// The sample axis, the trailing axis of the samples-last inputs.
    fn sample_axis(&self) -> Axis {
        Axis(self.inputs.ndim() - 1)
    }

    /// Gathers the samples at `indices` along the sample axis into a fresh owned array.
    /// Rank-2 inputs gather on the static `Ix2` type, which avoids the dynamic-rank
    /// indexing overhead of [`ArrayD::select`] on the tabular hot path.
    fn select_samples(&self, indices: &[usize]) -> ArrayD<f32> {
        match self.inputs.view().into_dimensionality::<Ix2>() {
            Ok(inputs) => inputs.select(Axis(1), indices).into_dyn(),
            Err(_) => self.inputs.select(self.sample_axis(), indices),
        }
    }

    /// Returns the targets, `(targets, samples)`.
    pub fn targets(&self) -> &Array2<f32> {
        &self.targets
    }

    /// Returns a lazy iterator over shuffled mini-batches of `size` samples each.
    /// The last batch may be smaller if `n_samples` is not divisible by `size`.
    /// Each batch is allocated on demand rather than all upfront.
    pub fn batches<R: Rng>(
        &self,
        size: usize,
        rng: &mut R,
    ) -> impl Iterator<Item = ModelDataset> + '_ {
        let mut indices: Vec<usize> = (0..self.targets.ncols()).collect();
        indices.shuffle(rng);
        let n = indices.len();
        let mut pos = 0;
        std::iter::from_fn(move || {
            if pos >= n {
                return None;
            }
            let end = (pos + size).min(n);
            let chunk = &indices[pos..end];
            pos = end;
            Some(ModelDataset::new(
                self.select_samples(chunk),
                self.targets.select(Axis(1), chunk),
            ))
        })
    }

    /// Fits a scaler of the given `kind` on these inputs, laid out samples-last with
    /// features along the leading axis.
    pub fn fit_scaler(&self, kind: ScalerKind) -> ScalerMethod {
        kind.fit(self.inputs.view())
    }

    /// Applies `scaler` to the inputs in place. The inputs are laid out samples-last
    /// with features along the leading axis.
    ///
    /// # Errors
    /// [`ScalerFeatureMismatch`] when the scaler's fitted feature count does not match
    /// these inputs.
    pub fn scale_inplace(&mut self, scaler: &dyn Scaler) -> Result<(), ScalerFeatureMismatch> {
        scaler.apply_inplace(self.inputs.view_mut())
    }
}

impl ModelSplit {
    /// Returns the number of training samples in the dataset.
    pub fn train_size(&self) -> usize {
        self.train.targets.ncols()
    }

    /// Returns the number of validation samples in the dataset.
    pub fn validation_size(&self) -> usize {
        self.validation
            .as_ref()
            .map_or(0, |val| val.targets.ncols())
    }

    /// Returns the number of testing samples in the dataset.
    pub fn test_size(&self) -> usize {
        self.test.targets.ncols()
    }

    /// Applies `scaler` in-place to every split. The scaler is fitted on the
    /// train split alone; transforming all splits keeps validation and test
    /// inputs on the same scale the model is trained on.
    ///
    /// # Errors
    /// [`ScalerFeatureMismatch`] when the scaler's fitted feature count does not match
    /// the split's inputs.
    pub fn scale_inplace(&mut self, scaler: &dyn Scaler) -> Result<(), ScalerFeatureMismatch> {
        self.train.scale_inplace(scaler)?;
        if let Some(validation) = self.validation.as_mut() {
            validation.scale_inplace(scaler)?;
        }
        self.test.scale_inplace(scaler)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2, array};

    #[test]
    fn new_rejects_out_of_range_multiclass_label() {
        // labels = [0, 1, 2, 5]: n_classes=4, but label 5 >= n_classes=4
        let features = Array2::zeros((4, 2));
        let labels = array![0.0f32, 1.0, 2.0, 5.0];
        assert_eq!(
            Dataset::new(features, labels, None).err(),
            Some(DatasetError::InvalidLabels)
        );
    }

    #[test]
    fn valid_multiclass_labels_produce_correct_one_hot() {
        let features = Array2::zeros((3, 2));
        let labels = array![0.0f32, 1.0, 2.0];
        let dataset = Dataset::new(features, labels, None).unwrap();
        let model_dataset = dataset.to_model_dataset();
        // targets shape: (n_classes=3, n_samples=3)
        assert_eq!(model_dataset.targets.shape(), &[3, 3]);
        // Each column is one-hot: column i has 1.0 at row i
        for i in 0..3 {
            assert_eq!(model_dataset.targets[[i, i]], 1.0);
        }
    }

    #[test]
    fn model_dataset_new_pairs_aligned_inputs_and_targets() {
        let dataset = ModelDataset::new(Array2::zeros((3, 4)), Array2::zeros((2, 4)));
        assert_eq!(dataset.inputs().shape(), &[3, 4]);
        assert_eq!(dataset.targets().shape(), &[2, 4]);
    }

    #[test]
    #[should_panic(expected = "same number of samples")]
    fn model_dataset_new_rejects_mismatched_sample_counts() {
        // 4 input columns but 3 target columns: the invariant must reject this.
        ModelDataset::new(Array2::zeros((3, 4)), Array2::zeros((2, 3)));
    }

    #[test]
    fn binary_labels_produce_single_row_targets() {
        let features = Array2::zeros((3, 2));
        let labels = array![0.0f32, 1.0, 0.0];
        let model_dataset = Dataset::new(features, labels, None)
            .unwrap()
            .to_model_dataset();
        // Binary labels stay as a single (1, n_samples) row, not one-hot encoded.
        assert_eq!(model_dataset.targets.shape(), &[1, 3]);
        assert_eq!(model_dataset.targets.row(0).to_vec(), vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn new_rejects_non_contiguous_binary_labels() {
        // labels = [1, 2]: two classes, but not the contiguous {0, 1} set.
        let features = Array2::zeros((2, 2));
        let labels = array![1.0f32, 2.0];
        assert_eq!(
            Dataset::new(features, labels, None).err(),
            Some(DatasetError::InvalidLabels)
        );
    }

    #[test]
    fn new_accepts_a_well_formed_dataset() {
        let features = Array2::zeros((4, 2));
        let labels = array![0.0f32, 1.0, 0.0, 1.0];
        assert!(Dataset::new(features, labels, None).is_ok());
    }

    #[test]
    fn id_pairs_origin_label_with_shape() {
        let features = Array2::zeros((4, 2));
        let labels = array![0.0f32, 1.0, 0.0, 1.0];
        let origin = DatasetOrigin::Synthetic {
            distribution: "spiral".to_string(),
            seed: 42,
        };
        let dataset = Dataset::new(features, labels, Some(origin)).unwrap();
        assert_eq!(dataset.id(), "spiral-seed42-c2-f2-n4");
    }

    #[test]
    fn id_falls_back_to_a_neutral_prefix_without_origin() {
        let features = Array2::zeros((4, 2));
        let labels = array![0.0f32, 1.0, 0.0, 1.0];
        let dataset = Dataset::new(features, labels, None).unwrap();
        assert_eq!(dataset.id(), "dataset-c2-f2-n4");
    }

    #[test]
    fn new_rejects_single_class_dataset() {
        let features = Array2::zeros((3, 2));
        let labels = array![1.0f32, 1.0, 1.0];
        assert_eq!(
            Dataset::new(features, labels, None).err(),
            Some(DatasetError::TooFewClasses(1))
        );
    }

    #[test]
    fn new_rejects_dataset_without_features() {
        let features = Array2::zeros((3, 0));
        let labels = array![0.0f32, 1.0, 0.0];
        assert_eq!(
            Dataset::new(features, labels, None).err(),
            Some(DatasetError::NoFeatures)
        );
    }

    #[test]
    fn new_rejects_empty_dataset() {
        let features = Array2::zeros((0, 2));
        let labels = Array1::zeros(0);
        assert_eq!(
            Dataset::new(features, labels, None).err(),
            Some(DatasetError::NoSamples)
        );
    }

    #[test]
    fn new_rejects_shape_mismatch() {
        // 3 feature rows but only 2 labels.
        let features = Array2::zeros((3, 2));
        let labels = array![0.0f32, 1.0];
        assert_eq!(
            Dataset::new(features, labels, None).err(),
            Some(DatasetError::ShapeMismatch {
                features: 3,
                labels: 2
            })
        );
    }

    #[test]
    fn split_ratios_produce_correct_sizes() {
        // 100 samples, 20% test, 10% val → 70 train / 10 val / 20 test
        let features = Array2::zeros((100, 2));
        let labels = Array1::from_shape_fn(100, |i| (i % 2) as f32);
        let dataset = Dataset::new(features, labels, None).unwrap();
        let split = dataset.split(0.1, 0.2, 0);
        assert_eq!(split.train_size(), 70);
        assert_eq!(split.validation_size(), 10);
        assert_eq!(split.test_size(), 20);
    }

    #[test]
    fn split_without_validation_yields_no_validation_set() {
        // val_ratio 0.0 → the validation split is None.
        let features = Array2::from_shape_fn((100, 2), |(i, _)| i as f32);
        let labels = Array1::from_shape_fn(100, |i| (i % 2) as f32);
        let mut split = Dataset::new(features, labels, None)
            .unwrap()
            .split(0.0, 0.2, 0);
        assert!(split.validation.is_none());
        assert_eq!(split.train_size(), 80);
        assert_eq!(split.test_size(), 20);

        // Scaling skips the absent validation split without panicking.
        let scaler = split.train.fit_scaler(ScalerKind::MinMax);
        split.scale_inplace(&scaler).unwrap();
        assert!(split.train.inputs.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn from_encoded_rejects_empty_images() {
        // No images → no samples, surfaced as a Result rather than a panic.
        let err = Dataset::from_encoded("test", vec![], vec![]).err().unwrap();
        assert_eq!(err.to_string(), DatasetError::NoSamples.to_string());
    }

    #[test]
    fn from_encoded_rejects_inconsistent_image_sizes() {
        // Images of differing lengths cannot form a rectangular matrix → Err.
        let images = vec![array![0.0f32, 1.0], array![2.0, 3.0, 4.0]];
        let labels = vec![0usize, 1];
        assert!(Dataset::from_encoded("test", images, labels).is_err());
    }

    #[test]
    fn from_encoded_propagates_dataset_validation_error() {
        // The images form a valid rectangular matrix, but `from_encoded` does not
        // check that there are as many labels as images — it delegates to
        // `Dataset::new`, which rejects the count mismatch.
        let images = vec![array![0.0f32, 1.0], array![2.0, 3.0]];
        let labels = vec![0usize]; // one label for two images
        let err = Dataset::from_encoded("test", images, labels).err().unwrap();
        assert!(
            err.to_string().contains("disagree on sample count"),
            "got: {err}"
        );
    }

    #[test]
    fn unique_labels_deduplicates_values() {
        let dataset =
            Dataset::new(Array2::zeros((4, 1)), array![0.0f32, 1.0, 1.0, 2.0], None).unwrap();
        let mut unique = dataset.unique_labels();
        unique.sort_by(f32::total_cmp);
        assert_eq!(unique, vec![0.0, 1.0, 2.0]);
    }

    #[test]
    fn get_features_for_label_selects_matching_rows() {
        let dataset = Dataset::new(
            array![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
            array![0.0f32, 1.0, 0.0],
            None,
        )
        .unwrap();
        let rows = dataset.get_features_for_label(0.0);
        assert_eq!(rows.shape(), &[2, 2]);
        assert_eq!(rows.row(0).to_vec(), vec![1.0, 1.0]);
        assert_eq!(rows.row(1).to_vec(), vec![3.0, 3.0]);
    }

    #[test]
    fn feature_range_returns_per_feature_min_max() {
        let dataset = Dataset::new(
            array![[1.0, 10.0], [3.0, 5.0], [-2.0, 8.0]],
            array![0.0f32, 1.0, 0.0],
            None,
        )
        .unwrap();
        let (mins, maxs) = dataset.feature_range();
        assert_eq!(mins, vec![-2.0, 5.0]);
        assert_eq!(maxs, vec![3.0, 10.0]);
    }

    #[test]
    fn fit_scaler_then_scale_inplace_normalizes_model_inputs() {
        let dataset =
            Dataset::new(array![[0.0, 0.0], [10.0, 20.0]], array![0.0f32, 1.0], None).unwrap();
        let mut model = dataset.to_model_dataset();

        // Fit on the column-major inputs and apply back in place.
        let scaler = model.fit_scaler(ScalerKind::MinMax);
        model.scale_inplace(&scaler).unwrap();

        assert!(
            model
                .inputs
                .iter()
                .all(|&v| (0.0..=1.0 + 1e-5).contains(&v))
        );
    }

    #[test]
    fn split_scale_inplace_scales_every_split() {
        // 10 samples, two features on different scales; split keeps a validation
        // set so all three branches of `ModelSplit::scale_inplace` are exercised.
        let features = Array2::from_shape_fn((10, 2), |(i, j)| {
            (i as f32) * if j == 0 { 1.0 } else { 5.0 }
        });
        let labels = Array1::from_shape_fn(10, |i| (i % 2) as f32);
        let dataset = Dataset::new(features, labels, None).unwrap();

        let mut split = dataset.split(0.2, 0.2, 0);
        let scaler = split.train.fit_scaler(ScalerKind::MinMax);
        split.scale_inplace(&scaler).unwrap();

        for part in [
            Some(&split.train),
            split.validation.as_ref(),
            Some(&split.test),
        ]
        .into_iter()
        .flatten()
        {
            assert!(part.inputs.iter().all(|&v| v.is_finite()));
        }
    }

    #[test]
    fn model_dataset_accepts_rank4_inputs_and_batches_on_the_sample_axis() {
        use ndarray::Array4;

        // (channels=1, height=2, width=2, samples=6): samples on the trailing axis.
        let inputs = Array4::from_shape_fn((1, 2, 2, 6), |(_, h, w, s)| (h + w + s) as f32);
        let targets = Array2::from_shape_fn((1, 6), |(_, s)| (s % 2) as f32);
        let dataset = ModelDataset::new(inputs, targets);
        assert_eq!(dataset.inputs().shape(), &[1, 2, 2, 6]);

        let mut rng = StdRng::seed_from_u64(0);
        let batches: Vec<_> = dataset.batches(4, &mut rng).collect();
        // 6 samples in size-4 batches → a full batch of 4 and a remainder of 2, with
        // only the trailing sample axis chunked and the feature axes preserved.
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].inputs().shape(), &[1, 2, 2, 4]);
        assert_eq!(batches[0].targets().shape(), &[1, 4]);
        assert_eq!(batches[1].inputs().shape(), &[1, 2, 2, 2]);
    }

    #[test]
    fn fit_scaler_normalizes_rank_n_inputs_per_leading_feature() {
        use ndarray::Array4;

        // (features=2, height=1, width=2, samples=3): feature 0 spans [0, 10] and
        // feature 1 [0, 100] across all spatial cells and samples. Fitting and scaling
        // through the dataset keeps the spatial shape and normalizes per feature, not
        // per spatial cell.
        let inputs = Array4::from_shape_fn((2, 1, 2, 3), |(f, _h, w, s)| {
            let scale = if f == 0 { 1.0 } else { 10.0 };
            let cell = if w == 0 { 0.0 } else { 8.0 };
            (cell + s as f32) * scale
        });
        let targets = Array2::from_shape_fn((1, 3), |(_, s)| (s % 2) as f32);
        let mut model = ModelDataset::new(inputs, targets);

        let scaler = model.fit_scaler(ScalerKind::MinMax);
        model.scale_inplace(&scaler).unwrap();

        assert_eq!(model.inputs().shape(), &[2, 1, 2, 3]);
        assert!(
            model
                .inputs
                .iter()
                .all(|&v| (0.0..=1.0 + 1e-5).contains(&v))
        );
        // An interior sample of feature 0 (value 1 in [0, 10]) lands at 0.1, not 0.5:
        // the feature is normalized as a whole, not per spatial cell.
        assert!((model.inputs[[0, 0, 0, 1]] - 0.1).abs() < 1e-5);
        assert!((model.inputs[[0, 0, 1, 0]] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn from_encoded_builds_dataset_from_images() {
        let images = vec![array![0.0f32, 1.0], array![2.0, 3.0], array![4.0, 5.0]];
        let labels = vec![0usize, 1, 2];

        let dataset = Dataset::from_encoded("digits", images, labels).unwrap();
        assert_eq!(dataset.features().shape(), &[3, 2]);
        assert_eq!(dataset.labels().len(), 3);
        let mut unique = dataset.unique_labels();
        unique.sort_by(f32::total_cmp);
        assert_eq!(unique, vec![0.0, 1.0, 2.0]);

        // The source is stamped into the origin.
        assert_eq!(
            dataset.origin(),
            Some(&DatasetOrigin::Encoded {
                source: "digits".to_string(),
            })
        );
    }

    #[test]
    fn dataset_error_messages_are_descriptive() {
        assert_eq!(
            DatasetError::NoFeatures.to_string(),
            "dataset has no features"
        );
        assert_eq!(
            DatasetError::NoSamples.to_string(),
            "dataset has no samples"
        );
        assert_eq!(
            DatasetError::ShapeMismatch {
                features: 3,
                labels: 2
            }
            .to_string(),
            "features and labels disagree on sample count: 3 rows vs 2 labels"
        );
        assert_eq!(
            DatasetError::TooFewClasses(1).to_string(),
            "a classifier needs at least 2 classes, but found 1"
        );
        assert_eq!(
            DatasetError::InvalidLabels.to_string(),
            "labels must be contiguous 0-indexed class ids in [0, n_classes)"
        );
    }
}
