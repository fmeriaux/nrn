mod min_max;
mod z_score;

pub use min_max::MinMaxScaler;
pub use z_score::ZScoreScaler;

use ndarray::{ArrayViewMut1, ArrayViewMut2, Axis};

/// Scaling transforms the input data to a specific range or distribution,
/// which often improves the performance and convergence of models.
///
/// # Example scaling methods:
/// - Min-Max Scaling: rescales data to a fixed range, often [0, 1].
/// - Z-score Normalization (Standardization): centers data around zero with unit variance.
///
/// # Usage
/// Implementors of this trait provide a scaling transformation applied feature-wise
/// to a 2D array of floating-point values (e.g., features in a dataset).
pub trait Scaler: Send + Sync {
    /// Returns the canonical name of the scaler method.
    fn name(&self) -> &'static str;

    /// Applies the scaling transformation in-place to the input 2D array.
    ///
    /// The input array is modified directly; no new allocation occurs.
    ///
    /// # Parameters
    ///
    /// * `input` - A mutable 2D array view representing dataset features.
    ///   The shape must be compatible with the scaler's expected input.
    ///
    fn apply_inplace(&self, input: ArrayViewMut2<f32>);

    /// Applies the scaling transformation to a single 1D array (vector) in-place.
    ///
    /// # Arguments
    /// * `input` - A mutable 1D array (vector) of `f32` values representing a single data point.
    ///
    /// # Behavior
    /// This method expands the 1D input into a 2D view, applies the transformation in-place,
    /// and modifies the original vector without allocation.
    ///
    fn apply_single_inplace(&self, input: ArrayViewMut1<f32>) {
        let mut expanded = input.insert_axis(Axis(0));
        self.apply_inplace(expanded.view_mut());
    }
}

/// Defines the available built-in scaling methods.
///
/// This enum provides a convenient way to select among the built-in scalers,
/// mainly for CLI or storage purposes (e.g., serialization, configuration).
/// It is possible to implement and use a custom `Scaler` without relying on this enum.
///
/// # Example
/// ```
/// use nrn::data::scalers::{ScalerMethod, MinMaxScaler, Scaler};
/// use ndarray::array;
///
/// let mut data = array![[0.0, 5.0], [10.0, 20.0]];
/// let scaler = ScalerMethod::MinMax(MinMaxScaler::default().fit(data.view()));
/// scaler.apply_inplace(data.view_mut());
/// assert!(data.iter().all(|&v| v >= 0.0 && v <= 1.0 + 1e-5));
/// ```
pub enum ScalerMethod {
    MinMax(MinMaxScaler),
    ZScore(ZScoreScaler),
}

/// Delegates the `Scaler` trait methods to the specific scaler implementations.
/// This allows using `ScalerMethod` wherever a `Scaler` is expected.
impl Scaler for ScalerMethod {
    fn name(&self) -> &'static str {
        match self {
            ScalerMethod::MinMax(s) => s.name(),
            ScalerMethod::ZScore(s) => s.name(),
        }
    }

    fn apply_inplace(&self, input: ArrayViewMut2<f32>) {
        match self {
            ScalerMethod::MinMax(s) => s.apply_inplace(input),
            ScalerMethod::ZScore(s) => s.apply_inplace(input),
        }
    }
}
