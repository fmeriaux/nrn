mod min_max;
mod z_score;

pub use min_max::MinMaxScaler;
pub use z_score::ZScoreScaler;

use ndarray::{ArrayD, ArrayView, ArrayViewD, ArrayViewMutD, RemoveAxis};

/// An input's feature count did not match the scaler's fitted feature count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScalerFeatureMismatch {
    /// The number of features the scaler was fitted on.
    pub expected: usize,
    /// The number of features the input carries.
    pub found: usize,
}

impl std::fmt::Display for ScalerFeatureMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "input has {} features but the scaler expects {}",
            self.found, self.expected
        )
    }
}

impl std::error::Error for ScalerFeatureMismatch {}

/// Scaling transforms the input data to a specific range or distribution,
/// which often improves the performance and convergence of models.
///
/// # Example scaling methods:
/// - Min-Max Scaling: rescales data to a fixed range, often [0, 1].
/// - Z-score Normalization (Standardization): centers data around zero with unit variance.
///
/// # Usage
/// Implementors of this trait provide a scaling transformation applied per feature
/// along the leading axis of a samples-last array (e.g., features in a dataset).
pub trait Scaler: Send + Sync {
    /// Returns the canonical name of the scaler method.
    fn name(&self) -> &'static str;

    /// Applies the scaling transformation in-place to samples-last inputs of any rank.
    ///
    /// Features run along the leading axis; each is scaled over the remaining axes.
    /// The input array is modified directly; no new allocation occurs.
    ///
    /// # Parameters
    ///
    /// * `inputs` - A mutable samples-last array view, features on the leading axis.
    ///
    /// # Errors
    /// [`ScalerFeatureMismatch`] when the leading axis does not match the scaler's
    /// fitted feature count.
    fn apply_inplace(&self, inputs: ArrayViewMutD<f32>) -> Result<(), ScalerFeatureMismatch>;

    /// Scales samples-last inputs and returns them owned, features on the leading axis.
    ///
    /// # Errors
    /// [`ScalerFeatureMismatch`] when the leading axis does not match the scaler's
    /// fitted feature count.
    fn apply(&self, inputs: ArrayViewD<f32>) -> Result<ArrayD<f32>, ScalerFeatureMismatch> {
        let mut owned = inputs.to_owned();
        self.apply_inplace(owned.view_mut())?;
        Ok(owned)
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
/// // Features on the leading axis: two features (rows), two samples (columns).
/// let mut data = array![[0.0, 5.0], [10.0, 20.0]];
/// let scaler = ScalerMethod::MinMax(MinMaxScaler::default().fit(data.view()));
/// scaler.apply_inplace(data.view_mut().into_dyn()).unwrap();
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
    /// Fits the chosen scaler on samples-last `data` (features along the leading
    /// axis), returning the fitted method.
    pub fn fit<D: RemoveAxis>(self, data: ArrayView<f32, D>) -> ScalerMethod {
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

    fn apply_inplace(&self, inputs: ArrayViewMutD<f32>) -> Result<(), ScalerFeatureMismatch> {
        match self {
            ScalerMethod::MinMax(s) => s.apply_inplace(inputs),
            ScalerMethod::ZScore(s) => s.apply_inplace(inputs),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn method_min_max_delegates_name_and_transform() {
        let data = array![[0.0, 5.0], [10.0, 20.0]];
        let method = ScalerMethod::MinMax(MinMaxScaler::default().fit(data.view()));

        assert_eq!(method.name(), "min-max");

        let mut to_scale = data.clone();
        method
            .apply_inplace(to_scale.view_mut().into_dyn())
            .unwrap();
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

        // Features run along rows (the leading axis); each should be centred near
        // zero mean over its samples after standardization.
        let mut to_scale = data.clone();
        method
            .apply_inplace(to_scale.view_mut().into_dyn())
            .unwrap();
        for row in to_scale.rows() {
            let mean: f32 = row.sum() / row.len() as f32;
            assert!(mean.abs() < 1e-5, "row mean was {}", mean);
        }
    }
}
