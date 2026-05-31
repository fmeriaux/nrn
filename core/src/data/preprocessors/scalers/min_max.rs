//! MinMax scaler for feature normalization.
//!
//! Rescales input data to a specified range, typically [0, 1], using the minimum and maximum values found in the data.
//! This is a common preprocessing step for neural networks, especially on image datasets such as MNIST.
//!
//! # Examples
//!
//! ```
//! use nrn::data::scalers::{MinMaxScaler, Scaler};
//! use ndarray::array;
//!
//! let mut data = array![[0.0, 5.0], [10.0, 20.0]];
//! let scaler = MinMaxScaler::default().fit(data.view());
//! scaler.apply_inplace(data.view_mut());
//! assert!(data.iter().all(|&v| v >= 0.0 && v <= 1.0 + 1e-5));
//! ```

use crate::data::scalers::Scaler;
use ndarray::{Array1, ArrayView2, ArrayViewMut2, Axis};

/// Scaler that linearly rescales data feature-wise to a target range using the min-max method.
/// See also the [`Scaler`] trait for interface details.
///
/// # Usage
/// Create a new scaler using [`MinMaxScaler::new`] or [`MinMaxScaler::default`].
/// Use [`MinMaxScaler::fit`] to compute min/max from training data, then [`MinMaxScaler::apply`] or [`MinMaxScaler::apply_inplace`] to transform data.
pub struct MinMaxScaler {
    /// Minimum value for each feature (computed during `fit`).
    pub min: Array1<f32>,
    /// Maximum value for each feature (computed during `fit`).
    pub max: Array1<f32>,
    /// Target range for scaling, usually (0.0, 1.0).
    pub range: (f32, f32),
}

impl MinMaxScaler {
    /// Creates a new `MinMaxScaler` with a specified `range`.
    ///
    /// # Arguments
    /// * `range` - Tuple `(min_target, max_target)` defining the scale output interval.
    ///
    pub fn new(range: (f32, f32)) -> Self {
        Self {
            min: Array1::zeros(0),
            max: Array1::zeros(0),
            range,
        }
    }

    /// Calculates the minimum and maximum values per feature from the input data,
    /// returning a new `MinMaxScaler` instance configured for scaling.
    ///
    /// This method consumes the current instance and returns a new one with
    /// fitted parameters.
    ///
    /// # Arguments
    /// * `data` - 2D array view of training data.
    ///
    /// # Returns
    /// A new `MinMaxScaler` configured with min/max per feature.
    ///
    pub fn fit(mut self, data: ArrayView2<f32>) -> Self {
        self.min = data.fold_axis(Axis(0), f32::INFINITY, |&a, &b| a.min(b));
        self.max = data.fold_axis(Axis(0), f32::NEG_INFINITY, |&a, &b| a.max(b));
        self
    }
}

impl Default for MinMaxScaler {
    /// Creates a `MinMaxScaler` with the default target range of [0.0, 1.0].
    fn default() -> Self {
        Self::new((0.0, 1.0))
    }
}

impl Scaler for MinMaxScaler {
    fn name(&self) -> &'static str {
        "min-max"
    }

    /// Applies min-max scaling in-place, modifying the input data directly.
    ///
    /// The transformation scales each feature independently to the configured
    /// target range using the min and max values computed during `fit`.
    ///
    /// Internally, when a feature’s max equals its min (constant feature),
    /// scaling avoids division by zero by adding a small epsilon.
    ///
    /// This means that such features will be mapped to the lower bound of the target range.
    ///
    /// # Panics
    ///
    /// Panics if the number of columns in `data` does not match the number of features
    /// in the scaler (`min` and `max` length).
    ///
    /// # Arguments
    ///
    /// * `data` - Mutable 2D array view of data to scale in-place.
    ///
    fn apply_inplace(&self, mut data: ArrayViewMut2<f32>) {
        assert_eq!(
            data.shape()[1],
            self.min.len(),
            "Shape mismatch: data columns ({}) != min/max length ({})",
            data.shape()[1],
            self.min.len()
        );

        let (min_range, max_range) = self.range;
        let scale = max_range - min_range;
        for (mut col, (&min, &max)) in data
            .axis_iter_mut(Axis(1))
            .zip(self.min.iter().zip(self.max.iter()))
        {
            let denom = max - min + f32::EPSILON;
            for v in col.iter_mut() {
                *v = ((*v - min) / denom) * scale + min_range;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn fit_computes_min_max_per_feature() {
        let data = array![[1.0, 10.0], [3.0, 20.0], [2.0, 30.0]];
        let scaler = MinMaxScaler::default().fit(data.view());
        assert_eq!(scaler.min[0], 1.0);
        assert_eq!(scaler.min[1], 10.0);
        assert_eq!(scaler.max[0], 3.0);
        assert_eq!(scaler.max[1], 30.0);
    }

    #[test]
    fn min_maps_to_zero_max_maps_to_one() {
        let data = array![[0.0, 0.0], [10.0, 100.0]];
        let scaler = MinMaxScaler::default().fit(data.view());
        let mut scaled = data.clone();
        scaler.apply_inplace(scaled.view_mut());
        assert!((scaled[[0, 0]] - 0.0).abs() < 1e-5);
        assert!((scaled[[1, 0]] - 1.0).abs() < 1e-5);
        assert!((scaled[[0, 1]] - 0.0).abs() < 1e-5);
        assert!((scaled[[1, 1]] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn all_values_within_zero_one_range() {
        let data = array![[1.0, 5.0], [2.0, 15.0], [3.0, 25.0]];
        let scaler = MinMaxScaler::default().fit(data.view());
        let mut scaled = data.clone();
        scaler.apply_inplace(scaled.view_mut());
        for &v in scaled.iter() {
            assert!((0.0..=1.0 + 1e-5).contains(&v), "Value {} out of [0, 1]", v);
        }
    }

    #[test]
    fn constant_feature_does_not_panic() {
        // First feature is constant — denom would be zero without epsilon guard
        let data = array![[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]];
        let scaler = MinMaxScaler::default().fit(data.view());
        let mut scaled = data.clone();
        scaler.apply_inplace(scaled.view_mut());
        // Constant feature maps to range lower bound (0.0)
        assert!((scaled[[0, 0]] - 0.0).abs() < 1e-3);
    }

    #[test]
    fn apply_single_matches_batch_apply() {
        let data = array![[1.0, 10.0], [3.0, 30.0]];
        let scaler = MinMaxScaler::default().fit(data.view());

        let mut single = ndarray::array![2.0, 20.0];
        scaler.apply_single_inplace(single.view_mut());

        let mut batch = array![[2.0, 20.0]];
        scaler.apply_inplace(batch.view_mut());

        assert!((single[0] - batch[[0, 0]]).abs() < 1e-6);
        assert!((single[1] - batch[[0, 1]]).abs() < 1e-6);
    }
}
