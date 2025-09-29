use crate::data::scalers::Scaler;
use ndarray::{Array1, Array2, ArrayView2, Axis, s};
use ndarray_rand::rand::Rng;
use ndarray_rand::rand::prelude::SliceRandom;
use std::error::Error;

/// A dataset containing features and labels for clustering tasks.
pub struct Dataset {
    /// A 2D array where each row is a sample and each column is a feature
    pub features: Array2<f32>,
    /// A 1D array where each element is the label for the corresponding sample
    pub labels: Array1<f32>,
}

/// A structure representing a split dataset into training and testing sets.
pub struct SplitDataset {
    /// A `Dataset` containing the training samples.
    pub train: Dataset,
    /// A `Dataset` containing the testing samples.
    pub test: Dataset,
}

impl Dataset {
    /// Returns the number of samples in the dataset.
    pub fn n_samples(&self) -> usize {
        self.labels.len()
    }

    /// Returns the number of features in the dataset.
    pub fn n_features(&self) -> usize {
        self.features.ncols()
    }

    /// Returns the maximum label value in the dataset.
    ///
    /// # Returns
    /// The largest value found in the `labels` array, cast to `usize`.
    ///
    /// # Notes
    /// - If the dataset is empty, returns `0`.
    /// - Assumes that all labels are non-negative and represent class indices.
    /// - Useful for determining the number of classes (e.g., for one-hot encoding).
    ///
    /// # Example
    /// ```
    /// let max = dataset.max_label();
    /// // For labels [0.0, 2.0, 1.0], returns 2
    /// ```
    pub fn max_label(&self) -> usize {
        self.labels.fold(0, |max, &label| max.max(label as usize))
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

        let max_label = self.max_label();

        let targets: Array2<f32> = if max_label > 1 {
            // If labels are not binary, we need to one-hot encode them
            let mut one_hot = Array2::zeros((max_label + 1, self.n_samples()));
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
    /// - `ratio`: Proportion of samples to include in the training set (e.g., 0.8 for 80/20 split)
    ///
    /// # Important
    /// This method does **not** shuffle the dataset. It assumes the dataset has already been shuffled.
    /// If the dataset is not shuffled, the split may not be representative.
    pub fn split(&self, ratio: f32) -> SplitDataset {
        assert!(ratio > 0.0 && ratio < 1.0, "Ratio must be between 0 and 1");
        let n_samples = self.features.nrows();
        let n_train = (n_samples as f32 * ratio).round() as usize;

        let slice = |start: usize, end: usize| Dataset {
            features: self.features.slice(s![start..end, ..]).to_owned(),
            labels: self.labels.slice(s![start..end]).to_owned(),
        };

        let train = slice(0, n_train);
        let test = slice(n_train, n_samples);
        SplitDataset { train, test }
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
    /// Shuffles can be useful to ensure that the data is randomly ordered before splitting into training
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
    /// Applies the specified scaling method to the training and test datasets.
    /// The transformation is done in-place.
    pub fn scale_inplace(&mut self, scaler: &dyn Scaler) {
        self.train.scale_inplace(scaler);
        self.test.scale_inplace(scaler);
    }

    /// Gives dataset groups by their usage.
    pub fn groups(&self) -> Vec<(&str, &Dataset)> {
        vec![("train", &self.train), ("test", &self.test)]
    }

    /// Unsplits the `SplitDataset` back into a single `Dataset`.
    pub fn unsplit(self) -> Dataset {
        let mut features = self.train.features.to_owned();
        let mut labels = self.train.labels.to_owned();

        features.append(Axis(0), self.test.features.view()).unwrap();
        labels.append(Axis(0), self.test.labels.view()).unwrap();

        Dataset { features, labels }
    }
}
