//! Z-score normalization for feature standardization.
//!
//! Standardizes input data so that it has zero mean and unit variance, using the mean and standard deviation computed from the data.
//!
//! This scaler is recommended for use with synthetic datasets or real-world data where features are not naturally bounded or may have different scales. It is commonly used in MLPs to help the network train efficiently when input features vary widely in range or distribution.
//!
//! For image datasets like MNIST, where pixel values are already bounded, min-max scaling is usually preferred.
//!
//! # Example
//!
//! ```
//! use ndarray::array;
//! use crate::data::scalers::ZScoreScaler;
//!
//! // Example: standardizing synthetic features before training a neural network
//! let data = array![[1.2, 3.4, 5.6], [7.8, 9.0, 2.1]];
//! let scaler = ZScoreScaler::fit(&data);
//! let scaled = scaler.apply(&data);
//! ```
//!

use crate::scalers::Scaler;
use ndarray::{Array1, ArrayView2, ArrayViewMut2, Axis};
/// ZScoreScaler that normalizes each feature to zero mean and unit variance.
///
/// This scaler fits the mean and standard deviation per feature (column),
/// and applies the transformation: `(x - mean) / std_dev`.
pub struct ZScoreScaler {
    /// Mean value for each feature (computed during `fit`).
    pub mean: Array1<f32>,
    /// Standard deviation for each feature (computed during `fit`).
    pub std_dev: Array1<f32>,
}

impl ZScoreScaler {
    /// Fits the scaler by computing mean and standard deviation per feature,
    /// consuming the current scaler instance and returning a new fitted instance.
    ///
    /// # Panics
    /// This method panics if unable to compute mean (e.g., empty input).
    ///
    /// # Examples
    ///
    /// ```
    /// let scaler = ZScoreScaler::default().fit(data.view());
    /// ```
    pub fn fit(mut self, data: ArrayView2<f32>) -> Self {
        self.mean = data.mean_axis(Axis(0)).expect("Unable to compute mean");
        self.std_dev = data.std_axis(Axis(0), 0.0);
        // To avoid division by zero, replace any zero std_dev by epsilon
        self.std_dev
            .mapv_inplace(|x| if x == 0.0 { f32::EPSILON } else { x });
        self
    }
}

impl Default for ZScoreScaler {
    /// Creates a default `ZScoreScaler` with empty mean and unit std_dev.
    fn default() -> Self {
        Self {
            mean: Array1::zeros(0),
            std_dev: Array1::ones(0),
        }
    }
}

impl Scaler for ZScoreScaler {
    fn name(&self) -> &'static str {
        "z-score"
    }

    /// Applies the z-score normalization **in-place**, modifying the provided data.
    ///
    /// # Arguments
    ///
    /// * `data` - Mutable 2D array view whose features will be normalized.
    ///
    /// # Panics
    ///
    /// Panics if the feature dimension of the input data does not match the scaler.
    ///
    /// # Examples
    ///
    /// ```
    /// scaler.apply_inplace(data.view_mut());
    /// ```
    fn apply_inplace(&self, mut data: ArrayViewMut2<f32>) {
        assert_eq!(
            data.shape()[1],
            self.mean.len(),
            "Shape mismatch: data columns ({}) != mean/std_dev length ({})",
            data.shape()[1],
            self.mean.len()
        );

        for (mut col, (&mean, &std)) in data
            .axis_iter_mut(Axis(1))
            .zip(self.mean.iter().zip(self.std_dev.iter()))
        {
            for v in col.iter_mut() {
                *v = (*v - mean) / std;
            }
        }
    }
}
