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
//! // Features on the leading axis: two features (rows), two samples (columns).
//! let mut data = array![[0.0, 5.0], [10.0, 20.0]];
//! let scaler = MinMaxScaler::default().fit(data.view());
//! scaler.apply_inplace(data.view_mut().into_dyn()).unwrap();
//! assert!(data.iter().all(|&v| v >= 0.0 && v <= 1.0 + 1e-5));
//! ```

use crate::data::scalers::{Scaler, ScalerFeatureMismatch};
use ndarray::{Array1, ArrayView, ArrayViewMutD, Axis, RemoveAxis};

/// Scaler that linearly rescales data feature-wise to a target range using the min-max method.
/// See also the [`Scaler`] trait for interface details.
///
/// # Usage
/// Create a new scaler using [`MinMaxScaler::new`] or [`MinMaxScaler::default`].
/// Use [`MinMaxScaler::fit`] to compute min/max from training data, then [`MinMaxScaler::apply`] or [`MinMaxScaler::apply_inplace`] to transform data.
#[derive(Clone, Debug)]
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

    /// Calculates the minimum and maximum per feature from samples-last input of
    /// any rank, returning a new `MinMaxScaler` instance configured for scaling.
    ///
    /// Features run along the leading axis; each is reduced over the remaining axes.
    /// This method consumes the current instance and returns a new one with fitted
    /// parameters.
    ///
    /// # Arguments
    /// * `data` - Array view of training data, features on the leading axis.
    ///
    /// # Returns
    /// A new `MinMaxScaler` configured with min/max per feature.
    ///
    pub fn fit<D: RemoveAxis>(mut self, data: ArrayView<f32, D>) -> Self {
        self.min = data
            .axis_iter(Axis(0))
            .map(|feature| feature.fold(f32::INFINITY, |a, &b| a.min(b)))
            .collect();
        self.max = data
            .axis_iter(Axis(0))
            .map(|feature| feature.fold(f32::NEG_INFINITY, |a, &b| a.max(b)))
            .collect();
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
    /// # Errors
    ///
    /// [`ScalerFeatureMismatch`] when the leading axis of `inputs` does not match the
    /// number of features in the scaler (`min` and `max` length).
    ///
    /// # Arguments
    ///
    /// * `inputs` - Mutable samples-last array to scale in-place, features on the leading axis.
    ///
    fn apply_inplace(&self, mut inputs: ArrayViewMutD<f32>) -> Result<(), ScalerFeatureMismatch> {
        let (expected, found) = (self.min.len(), inputs.shape()[0]);
        (found == expected)
            .then_some(())
            .ok_or(ScalerFeatureMismatch { expected, found })?;

        let (min_range, max_range) = self.range;
        let scale = max_range - min_range;
        for (mut feature, (&min, &max)) in inputs
            .axis_iter_mut(Axis(0))
            .zip(self.min.iter().zip(self.max.iter()))
        {
            let denom = max - min + f32::EPSILON;
            for v in feature.iter_mut() {
                *v = ((*v - min) / denom) * scale + min_range;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn fit_computes_min_max_per_feature() {
        // Two features (rows), three samples each (columns).
        let data = array![[1.0, 3.0, 2.0], [10.0, 20.0, 30.0]];
        let scaler = MinMaxScaler::default().fit(data.view());
        assert_eq!(scaler.min[0], 1.0);
        assert_eq!(scaler.min[1], 10.0);
        assert_eq!(scaler.max[0], 3.0);
        assert_eq!(scaler.max[1], 30.0);
    }

    #[test]
    fn apply_rejects_feature_count_mismatch() {
        // Fitted on 2 features, applied to inputs with 3 on the leading axis → error.
        let scaler = MinMaxScaler::default().fit(array![[0.0, 0.0], [1.0, 1.0]].view());
        let mut wrong = array![[0.0], [0.0], [0.0]];
        let error = scaler
            .apply_inplace(wrong.view_mut().into_dyn())
            .unwrap_err();
        assert_eq!((error.expected, error.found), (2, 3));
    }

    #[test]
    fn min_maps_to_zero_max_maps_to_one() {
        // Feature 0 spans [0, 10], feature 1 spans [0, 100] (one feature per row).
        let data = array![[0.0, 10.0], [0.0, 100.0]];
        let scaler = MinMaxScaler::default().fit(data.view());
        let mut scaled = data.clone();
        scaler.apply_inplace(scaled.view_mut().into_dyn()).unwrap();
        assert!((scaled[[0, 0]] - 0.0).abs() < 1e-5);
        assert!((scaled[[0, 1]] - 1.0).abs() < 1e-5);
        assert!((scaled[[1, 0]] - 0.0).abs() < 1e-5);
        assert!((scaled[[1, 1]] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn all_values_within_zero_one_range() {
        let data = array![[1.0, 5.0], [2.0, 15.0], [3.0, 25.0]];
        let scaler = MinMaxScaler::default().fit(data.view());
        let mut scaled = data.clone();
        scaler.apply_inplace(scaled.view_mut().into_dyn()).unwrap();
        for &v in scaled.iter() {
            assert!((0.0..=1.0 + 1e-5).contains(&v), "Value {} out of [0, 1]", v);
        }
    }

    #[test]
    fn constant_feature_does_not_panic() {
        // First feature (row) is constant — denom would be zero without epsilon guard.
        let data = array![[5.0, 5.0, 5.0], [1.0, 2.0, 3.0]];
        let scaler = MinMaxScaler::default().fit(data.view());
        let mut scaled = data.clone();
        scaler.apply_inplace(scaled.view_mut().into_dyn()).unwrap();
        // Constant feature maps to range lower bound (0.0)
        assert!((scaled[[0, 0]] - 0.0).abs() < 1e-3);
    }

    #[test]
    fn scales_rank_n_inputs_per_feature() {
        use ndarray::Array4;

        // (features=2, height=1, width=2, samples=3): feature 0 spans [0, 10] and
        // feature 1 [0, 100] across all spatial cells and samples. Each feature is
        // normalized over its whole slab, not per spatial cell.
        let data = Array4::from_shape_fn((2, 1, 2, 3), |(f, _h, w, s)| {
            let scale = if f == 0 { 1.0 } else { 10.0 };
            let cell = if w == 0 { 0.0 } else { 8.0 };
            (cell + s as f32) * scale
        });
        let scaler = MinMaxScaler::default().fit(data.view());
        let mut scaled = data.clone().into_dyn();
        scaler.apply_inplace(scaled.view_mut()).unwrap();

        assert_eq!(scaler.min.len(), 2);
        assert!(scaled.iter().all(|&v| (0.0..=1.0 + 1e-5).contains(&v)));
        // An interior sample of feature 0 (value 1 in [0, 10]) lands at 0.1, not 0.5:
        // the whole feature shares one min/max, it is not scaled per spatial cell.
        assert!((scaled[[0, 0, 0, 1]] - 0.1).abs() < 1e-5);
        assert!((scaled[[0, 0, 1, 0]] - 0.8).abs() < 1e-5);
    }
}
