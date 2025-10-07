use crate::data::scalers::Scaler;
use ndarray::{Array1, Array2, ArrayView2, Axis, s};
use ndarray_rand::rand::Rng;
use ndarray_rand::rand::prelude::SliceRandom;
use std::collections::HashSet;
use std::error::Error;

/// A dataset containing features and labels for clustering tasks.
#[derive(Clone)]
pub struct Dataset {
    /// A 2D array where each row is a sample and each column is a feature
    pub features: Array2<f32>,
    /// A 1D array where each element is the label for the corresponding sample
    pub labels: Array1<f32>,
}

/// A structure representing a split dataset into training and testing sets.
#[derive(Clone)]
pub struct SplitDataset {
    /// A `Dataset` containing the training samples.
    pub train: Dataset,
    /// An optional `Dataset` containing the validation samples.
    pub validation: Option<Dataset>,
    /// A `Dataset` containing the testing samples.
    pub test: Dataset,
}

impl Dataset {
    /// Checks if the dataset is empty (i.e., has no features or labels).
    pub fn is_empty(&self) -> bool {
        self.features.is_empty() || self.labels.is_empty()
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
    /// # Returns
    /// An `Option` containing a tuple of two vectors:
    /// - The first vector contains the minimum values for each feature.
    /// - The second vector contains the maximum values for each feature.
    /// If the dataset has no samples, returns `None`.
    ///
    pub fn feature_range(&self) -> Option<(Vec<f32>, Vec<f32>)> {
        if self.features.nrows() == 0 {
            return None;
        }

        let n_features = self.n_features();

        let mut mins = vec![f32::INFINITY; n_features];
        let mut maxs = vec![f32::NEG_INFINITY; n_features];

        for (i, feature) in self.features.axis_iter(Axis(1)).enumerate() {
            for &value in feature.iter() {
                mins[i] = mins[i].min(value);
                maxs[i] = maxs[i].max(value);
            }
        }

        Some((mins, maxs))
    }

    /// Applies the specified scaling method to the training and test datasets.
    /// The transformation is done in-place.
    pub fn scale_inplace(&mut self, scaler: &dyn Scaler) {
        scaler.apply_inplace(self.features.view_mut());
    }

    /// Transforms the dataset into a shape suitable for model input and output.
    ///
    /// # Returns
    /// A tuple `(inputs, targets)` where:
    /// - `targets` is a 2D array owned of shape `(n_classes, n_samples)` for multi-class (one-hot encoded labels),
    ///   or `(1, n_samples)` for binary labels.
    ///
    /// # Details
    /// - If the labels are not binary (i.e., `max_label > 1`), the function one-hot encodes the labels.
    /// - For binary labels, it simply adds an axis to match the expected shape.
    ///
    /// # Example
    /// ```
    /// let (inputs, targets) = dataset.to_model_shape();
    /// // Use `inputs` and `targets` as model input/output
    /// ```
    pub fn to_model_shape(&self) -> (ArrayView2<'_, f32>, Array2<f32>) {
        let inputs: ArrayView2<f32> = self.features.t();

        let n_classes = self.n_classes();

        let targets: Array2<f32> = if n_classes > 2 {
            // If labels are not binary, we need to one-hot encode them
            let mut one_hot = Array2::zeros((n_classes, self.n_samples()));
            for (i, &label) in self.labels.iter().enumerate() {
                one_hot[[label as usize, i]] = 1.0;
            }
            one_hot
        } else {
            // If labels are binary, we can use them directly
            self.labels.to_owned().insert_axis(Axis(0))
        };

        (inputs, targets)
    }

    /// Splits the dataset into training and testing sets according to the given ratio.
    ///
    /// # Parameters
    /// - `val_ratio`: The ratio of the dataset to be used for validation (between 0 and 1).
    /// - `test_ratio`: The ratio of the dataset to be used for testing (between 0 and 1).
    ///
    /// # Important
    /// This method does **not** shuffle the dataset. It assumes the dataset has already been shuffled.
    /// If the dataset is not shuffled, the split may not be representative.
    ///
    /// # Panics
    /// - When `val_ratio` or `test_ratio` is not between 0 and 1.
    /// - When the sum of `val_ratio` and `test_ratio` is greater than or equal to 1.
    //
    pub fn split(&self, val_ratio: f32, test_ratio: f32) -> SplitDataset {
        assert!(
            val_ratio.min(test_ratio) > 0.0 && val_ratio.max(test_ratio) < 1.0,
            "Ratio must be between 0 and 1"
        );
        assert!(
            val_ratio + test_ratio < 1.0,
            "Sum of ratios must be less than 1"
        );

        let n_samples = self.features.nrows();
        //let n_train = (n_samples as f32 * ratio).round() as usize;

        let size = |ratio: f32| (n_samples as f32 * ratio).round() as usize;
        let slice = |start: usize, end: usize| Dataset {
            features: self.features.slice(s![start..end, ..]).to_owned(),
            labels: self.labels.slice(s![start..end]).to_owned(),
        };

        let test_size = size(test_ratio);
        let val_size = size(val_ratio);
        let train_size = n_samples - test_size - val_size;

        SplitDataset {
            train: slice(0, train_size),
            validation: if val_size > 0 {
                Some(slice(train_size, train_size + val_size))
            } else {
                None
            },
            test: slice(train_size + val_size, n_samples),
        }
    }

    /// Shuffles the dataset features and labels in unison using a random number generator.
    /// See [`shuffle_inplace`](Self::shuffle_inplace) for details.
    pub fn shuffled<R: Rng>(rng: &mut R, features: &Array2<f32>, labels: &Array1<f32>) -> Self {
        assert_eq!(
            features.nrows(),
            labels.len(),
            "Features and labels must have the same number of samples"
        );

        let mut dataset = Dataset {
            features: features.to_owned(),
            labels: labels.to_owned(),
        };
        dataset.shuffle_inplace(rng);
        dataset
    }

    /// Shuffles the dataset features and labels in unison using a random number generator.
    /// # Arguments
    /// - `rng`: A mutable reference to a random number generator.
    /// # Details
    /// - The method generates a vector of indices corresponding to the number of samples in the dataset
    /// - It shuffles these indices using the provided random number generator.
    /// - It then reorders both the `features` and `labels` arrays according to
    /// the shuffled indices, ensuring that the correspondence between features and labels is maintained.
    pub fn shuffle_inplace<R: Rng>(&mut self, rng: &mut R) {
        let mut indices: Vec<usize> = (0..self.features.nrows()).collect();
        indices.shuffle(rng);

        let shuffled_features = self.features.select(Axis(0), &indices);
        let shuffled_labels = Array1::from(
            indices
                .iter()
                .map(|&i| self.labels[i])
                .collect::<Vec<f32>>(),
        );

        self.features = shuffled_features.to_owned();
        self.labels = shuffled_labels;
    }

    /// Creates a new dataset from a vector of images and their corresponding labels.
    /// # Arguments
    /// - `rng`: A mutable reference to a random number generator for shuffling.
    /// - `images`: A vector of images represented as 1D arrays of pixel values.
    /// - `labels`: A vector of labels corresponding to each image.
    pub fn from_vec<R: Rng>(
        rng: &mut R,
        images: Vec<Array1<f32>>,
        labels: Vec<usize>,
    ) -> Result<Self, Box<dyn Error>> {
        assert!(
            !images.is_empty(),
            "Images vector must not be empty to create a dataset"
        );

        let features: Array2<f32> = Array2::from_shape_vec(
            (images.len(), images[0].len()),
            images.into_iter().flatten().collect(),
        )?;

        let labels: Array1<f32> =
            Array1::from(labels.into_iter().map(|x| x as f32).collect::<Vec<_>>());

        Ok(Dataset::shuffled(rng, &features, &labels))
    }
}

impl SplitDataset {
    /// Unsplits the `SplitDataset` back into a single `Dataset`.
    pub fn unsplit(self) -> Dataset {
        let mut features = self.train.features.to_owned();
        let mut labels = self.train.labels.to_owned();

        if let Some(validation) = self.validation {
            features
                .append(Axis(0), validation.features.view())
                .unwrap();
            labels.append(Axis(0), validation.labels.view()).unwrap();
        }

        features.append(Axis(0), self.test.features.view()).unwrap();
        labels.append(Axis(0), self.test.labels.view()).unwrap();

        Dataset { features, labels }
    }
}
