use ndarray::ArrayD;
use std::fmt;
use std::ops::{Deref, DerefMut};

/// Small constant to prevent division by zero in gradient clipping.
/// This value was chosen to be sufficiently small to avoid affecting the clipping behavior
/// while ensuring numerical stability during calculations.
const EPSILON: f32 = 1e-6;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GradientClipping {
    /// No gradient clipping is applied.
    None,
    /// Gradients are clipped to a maximum norm using the L2 norm.
    Norm { max_norm: f32 },
    /// Gradients are clipped to a maximum value element-wise.
    Value { min: f32, max: f32 },
}

/// Returned by [`GradientClipping::norm`] / [`GradientClipping::value`]
/// when the given bounds are invalid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GradientClippingError {
    /// The max norm for [`GradientClipping::Norm`] was not a positive value.
    NonPositiveNorm(f32),
    /// The `(min, max)` range for [`GradientClipping::Value`] was not `min < max`.
    InvalidRange { min: f32, max: f32 },
}

impl fmt::Display for GradientClippingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GradientClippingError::NonPositiveNorm(max_norm) => write!(
                f,
                "the gradient clipping norm must be a positive value, got {max_norm}"
            ),
            GradientClippingError::InvalidRange { min, max } => write!(
                f,
                "the gradient clipping range must satisfy min < max, got min={min}, max={max}"
            ),
        }
    }
}

impl std::error::Error for GradientClippingError {}

impl GradientClipping {
    /// Creates a [`GradientClipping::Norm`] clipping by the L2 norm.
    /// # Errors
    /// Returns [`GradientClippingError::NonPositiveNorm`] when `max_norm` is not positive.
    pub fn norm(max_norm: f32) -> Result<Self, GradientClippingError> {
        if max_norm > 0.0 {
            Ok(GradientClipping::Norm { max_norm })
        } else {
            Err(GradientClippingError::NonPositiveNorm(max_norm))
        }
    }

    /// Creates a [`GradientClipping::Value`] element-wise clipping over `[min, max]`.
    /// # Errors
    /// Returns [`GradientClippingError::InvalidRange`] when `min >= max`.
    pub fn value(min: f32, max: f32) -> Result<Self, GradientClippingError> {
        if min < max {
            Ok(GradientClipping::Value { min, max })
        } else {
            Err(GradientClippingError::InvalidRange { min, max })
        }
    }
}

/// The gradients computed during backpropagation for a single layer: one tensor per
/// trainable parameter, in the layer's parameter order (see
/// [`Layer::parameters_mut`](crate::layers::Layer::parameters_mut)). Ranks are dynamic,
/// so parameters of any shape share one gradient type.
pub struct LayerGradients(pub Vec<ArrayD<f32>>);

impl Deref for LayerGradients {
    type Target = Vec<ArrayD<f32>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for LayerGradients {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl LayerGradients {
    /// Clips the gradients to a maximum norm, using the L2 norm over every parameter
    /// tensor jointly (the concatenation), so the layer's whole gradient is rescaled by
    /// a single factor.
    /// # Arguments
    /// - `max_norm`: The maximum norm to clip the gradients to.
    pub fn clip(&mut self, max_norm: f32) {
        let sum_sq: f32 = self
            .iter()
            .map(|g| g.iter().map(|x| x * x).sum::<f32>())
            .sum();
        let norm = sum_sq.sqrt();

        if norm > max_norm {
            let scale = max_norm / (norm + EPSILON);
            for grad in self.iter_mut() {
                grad.mapv_inplace(|x| x * scale);
            }
        }
    }

    /// Clips the gradients to a specified range element-wise, across every parameter tensor.
    /// # Arguments
    /// - `min`: The minimum value to clip the gradients to.
    /// - `max`: The maximum value to clip the gradients to.
    pub fn clip_value(&mut self, min: f32, max: f32) {
        for grad in self.iter_mut() {
            grad.mapv_inplace(|x| x.clamp(min, max));
        }
    }

    /// Clips the gradients based on the specified `GradientClipping` strategy.
    /// # Arguments
    /// - `clipping`: The `GradientClipping` strategy to apply.
    pub fn clip_by(&mut self, clipping: &GradientClipping) {
        match clipping {
            GradientClipping::None => {}
            GradientClipping::Norm { max_norm } => self.clip(*max_norm),
            GradientClipping::Value { min, max } => self.clip_value(*min, *max),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Builds layer gradients from a rank-2 weight gradient and a rank-1 bias gradient,
    /// the shape a dense layer produces.
    fn dense_grads(dw: ndarray::Array2<f32>, db: ndarray::Array1<f32>) -> LayerGradients {
        LayerGradients(vec![dw.into_dyn(), db.into_dyn()])
    }

    fn total_norm(grads: &LayerGradients) -> f32 {
        grads
            .iter()
            .map(|g| g.iter().map(|x| x * x).sum::<f32>())
            .sum::<f32>()
            .sqrt()
    }

    #[test]
    fn clip_rescales_gradients_above_the_max_norm() {
        // dw norm = sqrt(3^2 + 4^2) = 5, db = 0 → total norm 5, clipped to 1.0.
        let mut grads = dense_grads(array![[3.0, 4.0]], array![0.0]);
        grads.clip(1.0);
        let norm = total_norm(&grads);
        assert!((norm - 1.0).abs() < 1e-4, "norm was {}", norm);
    }

    #[test]
    fn clip_uses_the_joint_norm_across_parameters() {
        // Norm spread across two tensors: sqrt(3^2 + 4^2) = 5 jointly, clipped to 1.0.
        let mut grads = dense_grads(array![[3.0]], array![4.0]);
        grads.clip(1.0);
        assert!((total_norm(&grads) - 1.0).abs() < 1e-4);
    }

    #[test]
    fn clip_leaves_gradients_below_the_max_norm_untouched() {
        let mut grads = dense_grads(array![[0.3, 0.4]], array![0.0]);
        grads.clip(10.0);
        assert_eq!(grads[0], array![[0.3, 0.4]].into_dyn());
    }

    #[test]
    fn clip_value_clamps_element_wise() {
        let mut grads = dense_grads(array![[-5.0, 0.5, 5.0]], array![-3.0, 3.0]);
        grads.clip_value(-1.0, 1.0);
        assert_eq!(grads[0], array![[-1.0, 0.5, 1.0]].into_dyn());
        assert_eq!(grads[1], array![-1.0, 1.0].into_dyn());
    }

    #[test]
    fn clip_by_dispatches_to_the_selected_strategy() {
        // None leaves gradients unchanged.
        let mut none = dense_grads(array![[5.0]], array![5.0]);
        none.clip_by(&GradientClipping::None);
        assert_eq!(none[0], array![[5.0]].into_dyn());

        // Value clamps element-wise.
        let mut value = dense_grads(array![[5.0]], array![5.0]);
        value.clip_by(&GradientClipping::Value {
            min: -1.0,
            max: 1.0,
        });
        assert_eq!(value[0], array![[1.0]].into_dyn());

        // Norm rescales to the max norm.
        let mut norm = dense_grads(array![[3.0, 4.0]], array![0.0]);
        norm.clip_by(&GradientClipping::Norm { max_norm: 1.0 });
        assert!((total_norm(&norm) - 1.0).abs() < 1e-4);
    }
}
