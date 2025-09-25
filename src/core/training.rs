use ndarray::{Array1, Array2};

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
        let dw_norm = self.dw.iter().map(|x| x.powi(2)).sum::<f32>();
        let db_norm = self.db.iter().map(|x| x.powi(2)).sum::<f32>();
        let norm = (dw_norm + db_norm).sqrt();
        if norm > max_norm {
            let scale = max_norm / norm;
            self.dw.mapv_inplace(|x| x * scale);
            self.db.mapv_inplace(|x| x * scale);
        }
    }
}