mod min_max;
mod z_score;

pub use min_max::MinMaxScaler;
pub use z_score::ZScoreScaler;

use ndarray::{ArrayView2, ArrayViewMut1, ArrayViewMut2, Axis};

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
#[derive(Clone, Debug)]
pub enum ScalerMethod {
    MinMax(MinMaxScaler),
    ZScore(ZScoreScaler),
}

/// Declarative choice of scaling method, selected before any data is seen.
///
/// Where [`ScalerMethod`] carries the parameters of a *fitted* scaler,
/// `ScalerKind` names only the method; [`fit`](ScalerKind::fit) turns it into a
/// fitted [`ScalerMethod`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScalerKind {
    MinMax,
    ZScore,
}

impl ScalerKind {
    /// Fits the chosen scaler on `data` (samples along rows, features along
    /// columns), returning the fitted method.
    pub fn fit(self, data: ArrayView2<f32>) -> ScalerMethod {
        match self {
            ScalerKind::MinMax => ScalerMethod::MinMax(MinMaxScaler::default().fit(data)),
            ScalerKind::ZScore => ScalerMethod::ZScore(ZScoreScaler::default().fit(data)),
        }
    }

    /// The canonical name of this scaling method.
    pub fn name(self) -> &'static str {
        match self {
            ScalerKind::MinMax => MinMaxScaler::default().name(),
            ScalerKind::ZScore => ZScoreScaler::default().name(),
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn apply_single_inplace_scales_a_lone_vector() {
        // `apply_single_inplace` treats each element as its own feature (column),
        // so fit on three features each spanning [0, 10].
        let data = array![[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]];
        let scaler = MinMaxScaler::default().fit(data.view());

        let mut vector = array![0.0, 5.0, 10.0];
        scaler.apply_single_inplace(vector.view_mut());

        assert!((vector[0] - 0.0).abs() < 1e-5);
        assert!((vector[1] - 0.5).abs() < 1e-5);
        assert!((vector[2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn method_min_max_delegates_name_and_transform() {
        let data = array![[0.0, 5.0], [10.0, 20.0]];
        let method = ScalerMethod::MinMax(MinMaxScaler::default().fit(data.view()));

        assert_eq!(method.name(), "min-max");

        let mut to_scale = data.clone();
        method.apply_inplace(to_scale.view_mut());
        assert!(to_scale.iter().all(|&v| (0.0..=1.0 + 1e-5).contains(&v)));
    }

    #[test]
    fn kind_names_and_fits_each_method() {
        let data = array![[0.0, 0.0], [10.0, 10.0]];

        assert_eq!(ScalerKind::MinMax.name(), "min-max");
        assert_eq!(ScalerKind::ZScore.name(), "z-score");

        // The fitted method reports the inner scaler's name, confirming the variant.
        assert_eq!(ScalerKind::MinMax.fit(data.view()).name(), "min-max");
        assert_eq!(ScalerKind::ZScore.fit(data.view()).name(), "z-score");
    }

    #[test]
    fn method_z_score_delegates_name_and_transform() {
        let data = array![[0.0, 5.0], [10.0, 20.0]];
        let method = ScalerMethod::ZScore(ZScoreScaler::default().fit(data.view()));

        assert_eq!(method.name(), "z-score");

        // Features run along columns (fit averages over Axis(0)); each should be
        // centred near zero mean after standardization.
        let mut to_scale = data.clone();
        method.apply_inplace(to_scale.view_mut());
        for col in to_scale.columns() {
            let mean: f32 = col.sum() / col.len() as f32;
            assert!(mean.abs() < 1e-5, "column mean was {}", mean);
        }
    }
}
