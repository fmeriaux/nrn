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
//! use nrn::data::scalers::{ZScoreScaler, Scaler};
//! use ndarray::{array, Axis};
//!
//! let mut data = array![[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]];
//! let scaler = ZScoreScaler::default().fit(data.view());
//! scaler.apply_inplace(data.view_mut());
//! // Each feature now has approximately zero mean
//! let mean = data.mean_axis(Axis(0)).unwrap();
//! assert!(mean.iter().all(|&m| m.abs() < 1e-5));
//! ```
//!

use crate::data::scalers::Scaler;
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Axis, array};

    #[test]
    fn fit_computes_mean_per_feature() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let scaler = ZScoreScaler::default().fit(data.view());
        assert!((scaler.mean[0] - 3.0).abs() < 1e-5); // mean de [1, 3, 5]
        assert!((scaler.mean[1] - 4.0).abs() < 1e-5); // mean de [2, 4, 6]
    }

    #[test]
    #[should_panic(expected = "Shape mismatch: data columns")]
    fn apply_rejects_feature_count_mismatch() {
        // Fitted on 2 features, applied to 3-column data → guard must fire.
        let scaler = ZScoreScaler::default().fit(array![[1.0, 2.0], [3.0, 4.0]].view());
        let mut wrong = array![[0.0, 0.0, 0.0]];
        scaler.apply_inplace(wrong.view_mut());
    }

    #[test]
    fn scaled_data_has_zero_mean_and_unit_variance() {
        let data = array![
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
            [5.0, 50.0]
        ];
        let scaler = ZScoreScaler::default().fit(data.view());
        let mut scaled = data.clone();
        scaler.apply_inplace(scaled.view_mut());

        let mean = scaled.mean_axis(Axis(0)).unwrap();
        for &m in mean.iter() {
            assert!(m.abs() < 1e-5, "Mean {} not close to 0", m);
        }

        let std = scaled.std_axis(Axis(0), 0.0);
        for &s in std.iter() {
            assert!((s - 1.0).abs() < 1e-5, "Std {} not close to 1", s);
        }
    }

    #[test]
    fn known_values_transform_correctly() {
        // [0, 2] → mean=1, std_population=1 → [-1, 1]
        let data = array![[0.0], [2.0]];
        let scaler = ZScoreScaler::default().fit(data.view());
        let mut scaled = data.clone();
        scaler.apply_inplace(scaled.view_mut());
        assert!((scaled[[0, 0]] - (-1.0)).abs() < 1e-5);
        assert!((scaled[[1, 0]] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn constant_feature_does_not_panic() {
        // First feature is constant — std would be zero without epsilon guard
        let data = array![[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]];
        let scaler = ZScoreScaler::default().fit(data.view());
        let mut scaled = data.clone();
        scaler.apply_inplace(scaled.view_mut());
    }
}
