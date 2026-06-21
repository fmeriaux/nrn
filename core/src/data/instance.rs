use ndarray::{Array1, ArrayView1};

/// A single unlabelled feature vector fed to a trained model for inference.
///
/// Unlike [`Dataset`](crate::data::Dataset), which pairs features with labels, an
/// instance carries only the features of one sample.
#[derive(Clone, Debug, PartialEq)]
pub struct Instance(Array1<f32>);

impl Instance {
    /// Wraps a feature vector as an instance.
    pub fn new(features: Array1<f32>) -> Self {
        Self(features)
    }

    /// The feature vector.
    pub fn values(&self) -> &Array1<f32> {
        &self.0
    }

    /// A read-only view of the feature vector.
    pub fn view(&self) -> ArrayView1<'_, f32> {
        self.0.view()
    }

    /// The number of features.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the instance has no features.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl From<Array1<f32>> for Instance {
    fn from(features: Array1<f32>) -> Self {
        Self::new(features)
    }
}

#[cfg(test)]
mod tests {
    use super::Instance;
    use ndarray::{Array1, array};

    #[test]
    fn exposes_its_feature_vector() {
        let instance = Instance::from(array![0.1, -0.2, 3.5]);

        assert_eq!(instance.len(), 3);
        assert!(!instance.is_empty());
        assert_eq!(instance.values(), &array![0.1, -0.2, 3.5]);
        assert_eq!(instance.view(), array![0.1, -0.2, 3.5].view());
    }

    #[test]
    fn reports_an_empty_instance() {
        assert!(Instance::new(Array1::<f32>::zeros(0)).is_empty());
    }
}
