use crate::core::scalers::Scaler;
use ndarray::{Array1, Array2, ArrayView2, Axis};

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
    /// A tuple `(inputs, expectations)` where:
    /// - `inputs` is a 2D array view of shape `(n_features, n_samples)`, i.e., features are transposed so that each column is a sample.
    /// - `expectations` is a 2D array owned of shape `(n_classes, n_samples)` for multi-class (one-hot encoded labels),
    ///   or `(1, n_samples)` for binary labels.
    ///
    /// # Details
    /// - If the labels are not binary (i.e., `max_label > 1`), the function one-hot encodes the labels.
    /// - For binary labels, it simply adds an axis to match the expected shape.
    ///
    /// # Example
    /// ```
    /// let (inputs, expectations) = dataset.to_model_shape();
    /// // Use `inputs` and `expectations` as model input/output
    /// ```
    pub fn to_model_shape(&self) -> (ArrayView2<'_, f32>, Array2<f32>) {
        let inputs: ArrayView2<f32> = self.features.t();

        let max_label = self.max_label();

        let expectations: Array2<f32> = if max_label > 1 {
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

        (inputs, expectations)
    }
}

impl SplitDataset {
    /// Applies the specified scaling method to the training and test datasets.
    /// The transformation is done in-place.
    pub fn scale_inplace(&mut self, scaler: &dyn Scaler) {
        self.train.scale_inplace(scaler);
        self.test.scale_inplace(scaler);
    }
}
