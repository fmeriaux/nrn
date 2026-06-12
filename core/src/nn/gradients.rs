use ndarray::{Array1, Array2};

/// Small constant to prevent division by zero in gradient clipping.
/// This value was chosen to be sufficiently small to avoid affecting the clipping behavior
/// while ensuring numerical stability during calculations.
const EPSILON: f32 = 1e-6;

pub enum GradientClipping {
    /// No gradient clipping is applied.
    None,
    /// Gradients are clipped to a maximum norm using the L2 norm.
    Norm { max_norm: f32 },
    /// Gradients are clipped to a maximum value element-wise.
    Value { min: f32, max: f32 },
}

/// Represents the gradients computed during backpropagation for a single layer.
pub struct Gradients {
    /// A 2D array where each element represents the gradient of the corresponding weight.
    pub dw: Array2<f32>,
    /// A 1D array where each element represents the gradient of the corresponding bias.
    pub db: Array1<f32>,
}

impl Gradients {
    /// Clips the gradients to a maximum norm, using the L2 norm.
    /// # Arguments
    /// - `max_norm`: The maximum norm to clip the gradients to.
    pub fn clip(&mut self, max_norm: f32) {
        let dw_norm = self.dw.mapv(|x| x.powi(2)).sum();
        let db_norm = self.db.mapv(|x| x.powi(2)).sum();
        let norm = (dw_norm + db_norm).sqrt();

        if norm > max_norm {
            let scale = max_norm / (norm + EPSILON);
            self.dw.mapv_inplace(|x| x * scale);
            self.db.mapv_inplace(|x| x * scale);
        }
    }

    /// Clips the gradients to a specified range element-wise.
    /// # Arguments
    /// - `min`: The minimum value to clip the gradients to.
    /// - `max`: The maximum value to clip the gradients to.
    pub fn clip_value(&mut self, min: f32, max: f32) {
        self.dw.mapv_inplace(|x| x.clamp(min, max));
        self.db.mapv_inplace(|x| x.clamp(min, max));
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

    #[test]
    fn clip_rescales_gradients_above_the_max_norm() {
        // dw norm = sqrt(3^2 + 4^2) = 5, db = 0 → total norm 5, clipped to 1.0.
        let mut grads = Gradients {
            dw: array![[3.0, 4.0]],
            db: array![0.0],
        };
        grads.clip(1.0);
        let norm = (grads.dw.mapv(|x| x.powi(2)).sum() + grads.db.mapv(|x| x.powi(2)).sum()).sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "norm was {}", norm);
    }

    #[test]
    fn clip_leaves_gradients_below_the_max_norm_untouched() {
        let mut grads = Gradients {
            dw: array![[0.3, 0.4]],
            db: array![0.0],
        };
        grads.clip(10.0);
        assert_eq!(grads.dw, array![[0.3, 0.4]]);
    }

    #[test]
    fn clip_value_clamps_element_wise() {
        let mut grads = Gradients {
            dw: array![[-5.0, 0.5, 5.0]],
            db: array![-3.0, 3.0],
        };
        grads.clip_value(-1.0, 1.0);
        assert_eq!(grads.dw, array![[-1.0, 0.5, 1.0]]);
        assert_eq!(grads.db, array![-1.0, 1.0]);
    }

    #[test]
    fn clip_by_dispatches_to_the_selected_strategy() {
        // None leaves gradients unchanged.
        let mut none = Gradients {
            dw: array![[5.0]],
            db: array![5.0],
        };
        none.clip_by(&GradientClipping::None);
        assert_eq!(none.dw, array![[5.0]]);

        // Value clamps element-wise.
        let mut value = Gradients {
            dw: array![[5.0]],
            db: array![5.0],
        };
        value.clip_by(&GradientClipping::Value {
            min: -1.0,
            max: 1.0,
        });
        assert_eq!(value.dw, array![[1.0]]);

        // Norm rescales to the max norm.
        let mut norm = Gradients {
            dw: array![[3.0, 4.0]],
            db: array![0.0],
        };
        norm.clip_by(&GradientClipping::Norm { max_norm: 1.0 });
        let total = (norm.dw.mapv(|x| x.powi(2)).sum() + norm.db.mapv(|x| x.powi(2)).sum()).sqrt();
        assert!((total - 1.0).abs() < 1e-4);
    }
}
