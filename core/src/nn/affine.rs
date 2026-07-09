use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

/// The affine map `weights · input + bias` and its backward pass — the linear building block a
/// layer wraps with an activation. [`Dense`](crate::layers::Dense) and
/// [`Conv2d`](crate::layers::Conv2d) both build on it.
#[derive(Clone, Debug)]
pub struct Affine {
    /// A `(outputs, inputs)` array, one row per output.
    weights: Array2<f32>,
    /// A `(outputs)` array, one bias per output.
    biases: Array1<f32>,
}

impl Affine {
    /// Assembles an `Affine` from explicit weights and biases.
    /// # Panics
    /// - When `weights` is empty.
    /// - When the number of weight rows does not match the number of biases.
    /// # Arguments
    /// - `weights`: A `(outputs, inputs)` array, one row per output.
    /// - `biases`: A `(outputs)` array, one bias per output.
    pub fn new(weights: Array2<f32>, biases: Array1<f32>) -> Self {
        assert!(
            weights.nrows() > 0 && weights.ncols() > 0,
            "Affine weights must be non-empty."
        );
        assert_eq!(
            weights.nrows(),
            biases.len(),
            "Affine needs one bias per output."
        );

        Affine { weights, biases }
    }

    /// The pre-activation `weights · input + bias` for one batch.
    /// # Arguments
    /// - `input`: An `(inputs, samples)` array; the bias broadcasts across the samples.
    /// # Returns
    /// - An `(outputs, samples)` array.
    pub fn forward(&self, input: ArrayView2<f32>) -> Array2<f32> {
        let biases = self.biases.view().insert_axis(Axis(1)).to_owned();
        self.weights.dot(&input) + &biases
    }

    /// The backward pass of the affine map for one batch.
    /// # Arguments
    /// - `dz`: The gradient of the loss with respect to the pre-activation, `(outputs, samples)`.
    /// - `input`: The batch fed to [`forward`](Affine::forward), `(inputs, samples)`.
    /// - `m`: The number of samples the parameter gradients average over. For a convolution
    ///   this differs from `input.ncols()`, whose columns span spatial positions as well as
    ///   samples.
    /// - `compute_input_gradient`: Whether to also compute the gradient with respect to `input`.
    /// # Returns
    /// - `(dw, db, dinput)`: the weight gradient `(outputs, inputs)`, the bias gradient
    ///   `(outputs)`, and — when requested — the input gradient `(inputs, samples)`.
    pub fn backward(
        &self,
        dz: ArrayView2<f32>,
        input: ArrayView2<f32>,
        m: f32,
        compute_input_gradient: bool,
    ) -> (Array2<f32>, Array1<f32>, Option<Array2<f32>>) {
        let dw = dz.dot(&input.t()) / m;
        let db = dz.sum_axis(Axis(1)) / m;
        let dinput = compute_input_gradient.then(|| self.weights.t().dot(&dz));
        (dw, db, dinput)
    }

    /// The number of inputs this map takes — the width of each weight row.
    pub fn inputs(&self) -> usize {
        self.weights.ncols()
    }

    /// The number of outputs this map produces — one weight row and one bias per output.
    pub fn outputs(&self) -> usize {
        self.weights.nrows()
    }

    /// This map's weight matrix `(outputs, inputs)`.
    pub fn weights(&self) -> ArrayView2<'_, f32> {
        self.weights.view()
    }

    /// This map's biases, one per output.
    pub fn biases(&self) -> ArrayView1<'_, f32> {
        self.biases.view()
    }

    /// Mutable access to the weight matrix.
    #[cfg(test)]
    pub(crate) fn weights_mut(&mut self) -> &mut Array2<f32> {
        &mut self.weights
    }

    /// Mutable access to the biases.
    #[cfg(test)]
    pub(crate) fn biases_mut(&mut self) -> &mut Array1<f32> {
        &mut self.biases
    }

    /// Mutable access to both parameters at once — weights then biases — so a layer can hand
    /// them to an optimizer together without borrowing `self` twice.
    pub fn parameters_mut(&mut self) -> (&mut Array2<f32>, &mut Array1<f32>) {
        (&mut self.weights, &mut self.biases)
    }

    /// Whether every weight and bias is finite (no NaN or Inf).
    pub fn is_finite(&self) -> bool {
        self.weights.iter().all(|v| v.is_finite()) && self.biases.iter().all(|v| v.is_finite())
    }
}
