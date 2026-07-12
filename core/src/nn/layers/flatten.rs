use crate::gradients::LayerGradients;
use crate::layers::{BackwardPass, Layer, Parameter};
use crate::model::LayerSpec;
use ndarray::{ArrayD, ArrayView2, ArrayViewD};

/// A reshape layer collapsing each sample's feature dimensions into a single axis,
/// mapping a spatial batch `(feature dims…, samples)` to the flat `(features, samples)`
/// a [`Dense`](crate::layers::Dense) head consumes. It is the boundary between the
/// spatial stages of a network and its dense classifier.
///
/// The sample axis stays last, so flattening is a pure reshape: no data moves and the
/// layer holds no parameters. The backward pass reshapes the incoming gradient back to
/// the input's spatial shape.
#[derive(Clone, Debug)]
pub struct Flatten {
    /// The per-sample feature dimensions this layer flattens, sample axis excluded
    /// (e.g. `[channels, height, width]`).
    shape: Vec<usize>,
}

impl Flatten {
    /// Builds a `Flatten` for inputs whose per-sample shape is `shape` (the sample axis
    /// excluded).
    /// # Panics
    /// - When `shape` is empty or any dimension is zero.
    /// # Arguments
    /// - `shape`: The per-sample feature dimensions, such as `[channels, height, width]`.
    pub fn new(shape: impl Into<Vec<usize>>) -> Self {
        let shape = shape.into();
        assert!(
            !shape.is_empty(),
            "Flatten needs at least one feature dimension."
        );
        assert!(
            shape.iter().all(|&d| d > 0),
            "Flatten dimensions must be greater than zero."
        );

        Flatten { shape }
    }

    /// The number of features produced by flattening one sample.
    fn features(&self) -> usize {
        self.shape.iter().product()
    }
}

impl Layer for Flatten {
    fn forward(&self, input: ArrayViewD<f32>) -> ArrayD<f32> {
        let ndim = input.ndim();
        assert_eq!(
            ndim,
            self.shape.len() + 1,
            "Flatten expected a rank-{} input (feature dims + samples).",
            self.shape.len() + 1
        );
        assert_eq!(
            &input.shape()[..ndim - 1],
            self.shape.as_slice(),
            "Flatten input feature dimensions do not match its shape."
        );

        let samples = input.shape()[ndim - 1];
        input
            .to_shape((self.features(), samples))
            .expect("collapsing the leading axes of a batch preserves its element count")
            .into_owned()
            .into_dyn()
    }

    fn backward(
        &self,
        da: ArrayViewD<f32>,
        input: ArrayViewD<f32>,
        _output: ArrayViewD<f32>,
        compute_input_gradient: bool,
    ) -> BackwardPass {
        // Flatten carries no parameters; the incoming gradient only needs reshaping
        // back to the input's spatial shape for the upstream layer.
        let input_gradient = compute_input_gradient.then(|| {
            da.to_shape(input.raw_dim())
                .expect("the output gradient reshapes back to the input's spatial shape")
                .into_owned()
        });

        BackwardPass {
            gradients: LayerGradients(vec![]),
            input_gradient,
        }
    }

    fn parameters_mut(&mut self) -> Vec<Parameter<'_>> {
        vec![]
    }

    fn input_shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    fn output_shape(&self) -> Vec<usize> {
        vec![self.features()]
    }

    fn is_finite(&self) -> bool {
        true
    }

    fn spec(&self) -> LayerSpec {
        LayerSpec::Flatten
    }

    fn named_tensors(&self) -> Vec<(String, ArrayD<f32>)> {
        vec![]
    }

    fn activation_name(&self) -> Option<&str> {
        None
    }

    fn weight_matrix(&self) -> Option<ArrayView2<'_, f32>> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array, Array2, IxDyn};

    #[test]
    fn forward_collapses_feature_dims_keeping_samples_last() {
        // (channels=2, height=2, width=1, samples=3) → (4, 3). Filling the input with
        // 0..12 in C order makes the value at (feature, sample) equal feature*3 + sample,
        // so the flattened result must equal 0..12 reshaped to (4, 3).
        let input =
            Array::from_shape_vec(IxDyn(&[2, 2, 1, 3]), (0..12).map(|v| v as f32).collect())
                .unwrap();
        let flatten = Flatten::new(vec![2, 2, 1]);

        let output = flatten.forward(input.view());

        assert_eq!(output.shape(), &[4, 3]);
        let expected = Array2::from_shape_vec((4, 3), (0..12).map(|v| v as f32).collect()).unwrap();
        assert_eq!(output, expected.into_dyn());
    }

    #[test]
    fn accessors_report_flattened_size_and_no_parameters() {
        let mut flatten = Flatten::new(vec![3, 4, 5]);
        assert_eq!(flatten.input_size(), 60);
        assert_eq!(flatten.output_size(), 60);
        assert_eq!(flatten.activation_name(), None);
        assert!(flatten.weight_matrix().is_none());
        assert!(flatten.named_tensors().is_empty());
        assert!(flatten.parameters_mut().is_empty());
        assert!(flatten.is_finite());
    }

    #[test]
    fn backward_reshapes_gradient_back_to_input_shape() {
        let input =
            Array::from_shape_vec(IxDyn(&[2, 3, 4]), (0..24).map(|v| v as f32).collect()).unwrap();
        let flatten = Flatten::new(vec![2, 3]);
        let output = flatten.forward(input.view());

        // Distinct gradient values so the reshape order is verifiable.
        let da =
            ArrayD::from_shape_vec(output.raw_dim(), (0..24).map(|v| v as f32 * 0.5).collect())
                .unwrap();
        let pass = flatten.backward(da.view(), input.view(), output.view(), true);

        assert!(pass.gradients.is_empty(), "Flatten has no parameters");
        let input_gradient = pass.input_gradient.expect("input gradient requested");
        assert_eq!(input_gradient.shape(), input.shape());
        // Flattening the reshaped gradient recovers `da`: the two reshapes are inverses.
        assert_eq!(flatten.forward(input_gradient.view()), da);
    }

    #[test]
    fn backward_skips_input_gradient_when_not_requested() {
        let input = ArrayD::<f32>::zeros(IxDyn(&[2, 3, 5]));
        let flatten = Flatten::new(vec![2, 3]);
        let output = flatten.forward(input.view());

        let da = ArrayD::<f32>::ones(output.raw_dim());
        let pass = flatten.backward(da.view(), input.view(), output.view(), false);

        assert!(pass.input_gradient.is_none());
        assert!(pass.gradients.is_empty());
    }

    #[test]
    #[should_panic(expected = "Flatten input feature dimensions")]
    fn forward_rejects_mismatched_feature_dims() {
        let flatten = Flatten::new(vec![2, 3]);
        // Second feature dimension is 4, not the expected 3.
        let input = ArrayD::<f32>::zeros(IxDyn(&[2, 4, 5]));
        flatten.forward(input.view());
    }

    #[test]
    #[should_panic(expected = "feature dims + samples")]
    fn forward_rejects_wrong_rank() {
        let flatten = Flatten::new(vec![2, 3]);
        // Rank 2 where rank 3 (two feature dims + samples) is expected.
        let input = ArrayD::<f32>::zeros(IxDyn(&[6, 5]));
        flatten.forward(input.view());
    }
}
