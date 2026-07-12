//! [`Tensors`]: a layer's named tensors keyed by their canonical parameter names, with typed
//! rank-checked extraction, plus the [`TensorError`] that extraction raises.

use ndarray::{Array, Array1, ArrayD, Dimension, Ix1};
use std::collections::HashMap;
use std::fmt;

/// A layer's named tensors keyed by their canonical parameter names. It exposes those names
/// ([`WEIGHT`](Tensors::WEIGHT), [`BIAS`](Tensors::BIAS)) and casts each tensor to the rank a
/// layer expects as it is taken.
#[derive(Debug, Clone, Default)]
pub struct Tensors(HashMap<String, ArrayD<f32>>);

impl Tensors {
    /// Canonical name of a layer's weight tensor.
    pub const WEIGHT: &'static str = "weight";
    /// Canonical name of a layer's bias tensor.
    pub const BIAS: &'static str = "bias";

    /// Removes the tensor named `name`, casting it to rank `D`. Errors with
    /// [`TensorError::Missing`] when it is absent, or [`TensorError::WrongRank`] when its rank is
    /// not `D`.
    pub fn take<D: Dimension>(&mut self, name: &str) -> Result<Array<f32, D>, TensorError> {
        let tensor = self
            .0
            .remove(name)
            .ok_or_else(|| TensorError::Missing(name.to_string()))?;
        let got = tensor.ndim();
        tensor
            .into_dimensionality::<D>()
            .map_err(|_| TensorError::WrongRank {
                name: name.to_string(),
                expected: D::NDIM.unwrap_or(got),
                got,
            })
    }

    /// Removes the [`WEIGHT`](Tensors::WEIGHT) tensor, casting it to rank `D` (rank-2 for a dense
    /// matrix, rank-4 for a convolution kernel).
    pub fn take_weight<D: Dimension>(&mut self) -> Result<Array<f32, D>, TensorError> {
        self.take(Self::WEIGHT)
    }

    /// Removes the [`BIAS`](Tensors::BIAS) tensor, a rank-1 vector.
    pub fn take_bias(&mut self) -> Result<Array1<f32>, TensorError> {
        self.take::<Ix1>(Self::BIAS)
    }
}

/// Error raised when a required tensor is missing or has the wrong rank.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorError {
    /// No tensor was found under the given name.
    Missing(String),
    /// A tensor did not have the rank the layer requires.
    WrongRank {
        /// The tensor's name.
        name: String,
        /// The rank the layer requires.
        expected: usize,
        /// The rank the tensor actually had.
        got: usize,
    },
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorError::Missing(name) => write!(f, "missing tensor `{name}`"),
            TensorError::WrongRank {
                name,
                expected,
                got,
            } => write!(f, "tensor `{name}` has rank {got}, expected {expected}"),
        }
    }
}

impl std::error::Error for TensorError {}

impl From<HashMap<String, ArrayD<f32>>> for Tensors {
    fn from(map: HashMap<String, ArrayD<f32>>) -> Self {
        Tensors(map)
    }
}

impl FromIterator<(String, ArrayD<f32>)> for Tensors {
    fn from_iter<I: IntoIterator<Item = (String, ArrayD<f32>)>>(iter: I) -> Self {
        Tensors(iter.into_iter().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, IxDyn};

    fn sample() -> Tensors {
        Tensors::from_iter([
            (
                Tensors::WEIGHT.to_string(),
                Array2::zeros((2, 3)).into_dyn(),
            ),
            (Tensors::BIAS.to_string(), Array1::zeros(2).into_dyn()),
        ])
    }

    #[test]
    fn take_weight_and_bias_cast_to_the_expected_rank() {
        let mut tensors = sample();
        assert_eq!(tensors.take_weight::<ndarray::Ix2>().unwrap().dim(), (2, 3));
        assert_eq!(tensors.take_bias().unwrap().len(), 2);
    }

    #[test]
    fn take_reports_a_missing_tensor() {
        let mut tensors = Tensors::default();
        let err = tensors.take::<Ix1>("bias").unwrap_err();
        assert_eq!(err, TensorError::Missing("bias".to_string()));
    }

    #[test]
    fn take_reports_a_wrong_rank() {
        // The weight is stored rank-2 but taken as rank-1.
        let mut tensors = Tensors::from_iter([(
            Tensors::WEIGHT.to_string(),
            Array2::<f32>::zeros((2, 3)).into_dyn(),
        )]);
        let err = tensors.take::<Ix1>("weight").unwrap_err();
        assert_eq!(
            err,
            TensorError::WrongRank {
                name: "weight".to_string(),
                expected: 1,
                got: 2,
            }
        );
    }

    #[test]
    fn take_removes_the_tensor() {
        let mut tensors = sample();
        tensors.take_bias().unwrap();
        // Taking the same tensor twice fails: the first take removed it.
        assert!(tensors.take_bias().is_err());
        // A leftover-agnostic check that the array still round-trips.
        let weight = tensors.take_weight::<ndarray::Ix2>().unwrap();
        assert_eq!(weight, Array2::<f32>::zeros((2, 3)));
    }

    #[test]
    fn take_ignores_a_dynamic_shape_placeholder() {
        // A rank-3 tensor taken as rank-3 succeeds (generic `take`, not weight/bias).
        let mut tensors =
            Tensors::from_iter([("x".to_string(), ArrayD::<f32>::zeros(IxDyn(&[1, 2, 3])))]);
        let x = tensors.take::<ndarray::Ix3>("x").unwrap();
        assert_eq!(x.dim(), (1, 2, 3));
    }
}
