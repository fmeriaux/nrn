use crate::activations::Activation;
use crate::affine::Affine;
use crate::gradients::LayerGradients;
use crate::layers::{BackwardPass, Layer, LayerConfigError, LayerKind, Parameter};
use ndarray::{Array1, Array2, Array4, ArrayD, ArrayView2, ArrayView4, ArrayViewD, Ix1, Ix2, Ix4};
use ndarray_rand::rand::RngCore;
use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

/// A 2D convolution layer. It slides a set of small kernels across the height and width of
/// each sample; at every position it takes the weighted sum of the patch the kernel covers,
/// so each kernel sweeps out one feature map. An activation is then applied to the result.
///
/// Arrays are `(channels, height, width, samples)` — the sample axis stays last, the layout
/// [`Flatten`](crate::layers::Flatten) collapses.
///
/// The whole batch is convolved with a single matrix multiplication: `im2col` lays every
/// kernel-sized patch out as a column, the kernels form a `(out_channels, in_channels·kh·kw)`
/// matrix, and one matmul produces all feature maps at once — the same affine math as
/// [`Dense`](crate::layers::Dense), with `col2im` folding the gradient back on the way down.
#[derive(Clone, Debug)]
pub struct Conv2d {
    /// The affine map, its weights the kernels flattened to `(out_channels, in_channels·kh·kw)`.
    affine: Affine,
    /// The kernels' shape `(out_channels, in_channels, kernel_height, kernel_width)`.
    kernels_shape: (usize, usize, usize, usize),
    /// The per-sample input shape `(in_channels, height, width)`, sample axis excluded.
    input_shape: (usize, usize, usize),
    /// The stride applied on both spatial axes.
    stride: usize,
    /// The zero-padding added on both spatial axes before convolving.
    padding: usize,
    /// The activation applied to this layer's output.
    activation: Arc<dyn Activation>,
}

impl Conv2d {
    /// Assembles a `Conv2d` from explicit kernels, biases, input shape, and activation.
    /// # Panics
    /// - When any kernel dimension is zero.
    /// - When the number of biases does not match the number of output channels.
    /// - When the kernel's input channels do not match `input_shape`'s channels.
    /// - When `stride` is zero or the kernel does not fit the padded input.
    /// # Arguments
    /// - `kernels`: A `(out_channels, in_channels, kernel_height, kernel_width)` array.
    /// - `biases`: A `(out_channels)` array, one bias per output channel.
    /// - `input_shape`: The per-sample input shape `(in_channels, height, width)`.
    /// - `stride`: The stride applied on both spatial axes.
    /// - `padding`: The zero-padding added on both spatial axes.
    /// - `activation`: The activation applied to this layer's output.
    pub fn new(
        kernels: Array4<f32>,
        biases: Array1<f32>,
        input_shape: (usize, usize, usize),
        stride: usize,
        padding: usize,
        activation: Arc<dyn Activation>,
    ) -> Self {
        let (out_channels, in_channels, kh, kw) = kernels.dim();
        assert!(
            out_channels > 0 && in_channels > 0 && kh > 0 && kw > 0,
            "Conv2d kernels must be non-empty."
        );
        assert_eq!(
            out_channels,
            biases.len(),
            "Conv2d needs one bias per output channel."
        );
        assert_eq!(
            in_channels, input_shape.0,
            "Conv2d kernel input channels must match the input channels."
        );
        assert!(stride > 0, "Conv2d stride must be greater than zero.");
        let (_, height, width) = input_shape;
        assert!(
            height + 2 * padding >= kh && width + 2 * padding >= kw,
            "Conv2d kernel does not fit the padded input."
        );

        // Flatten the kernels to the affine matrix (out_channels, in·kh·kw); the fan-in axes are
        // contiguous, so this is a reshape, not a copy.
        let kernels_shape = (out_channels, in_channels, kh, kw);
        let weights = kernels
            .into_shape_with_order((out_channels, in_channels * kh * kw))
            .expect("(out, in, kh, kw) reshapes to (out, in·kh·kw): element counts match");

        Conv2d {
            affine: Affine::new(weights, biases),
            kernels_shape,
            input_shape,
            stride,
            padding,
            activation,
        }
    }

    /// Initializes a `Conv2d` with kernels drawn from `rng` per the activation's scheme.
    /// # Panics
    /// - When any dimension is zero (see [`Conv2d::new`]).
    /// # Arguments
    /// - `input_shape`: The per-sample input shape `(in_channels, height, width)`.
    /// - `out_channels`: The number of filters, i.e. output channels.
    /// - `kernel`: The kernel's spatial size `(kernel_height, kernel_width)`.
    /// - `stride`: The stride applied on both spatial axes.
    /// - `padding`: The zero-padding added on both spatial axes.
    /// - `activation`: The activation applied to this layer's output; it also picks the
    ///   weight-initialization scheme.
    /// - `rng`: The generator the kernels are drawn from. A seeded generator makes the
    ///   initialization reproducible.
    pub fn initialization(
        input_shape: (usize, usize, usize),
        out_channels: usize,
        kernel: (usize, usize),
        stride: usize,
        padding: usize,
        activation: Arc<dyn Activation>,
        rng: &mut dyn RngCore,
    ) -> Self {
        let (in_channels, _, _) = input_shape;
        let (kh, kw) = kernel;
        assert!(
            out_channels > 0 && in_channels > 0 && kh > 0 && kw > 0,
            "Channels and kernel size must be greater than zero."
        );

        // A convolution's fan-in is one receptive field: in_channels · kh · kw. Initializing
        // the flat (out_channels, fan_in) matrix reuses the dense initializers unchanged; the
        // kernels are that same matrix folded to rank 4.
        let fan_in = in_channels * kh * kw;
        let (weights, biases) = activation
            .initialization()
            .apply((out_channels, fan_in), rng);
        let kernels = weights
            .into_shape_with_order((out_channels, in_channels, kh, kw))
            .expect("(out, in·kh·kw) reshapes to (out, in, kh, kw): element counts match");

        Self::new(kernels, biases, input_shape, stride, padding, activation)
    }

    /// This layer's kernels `(out_channels, in_channels, kernel_height, kernel_width)`.
    pub fn kernels(&self) -> Array4<f32> {
        self.affine
            .weights()
            .to_owned()
            .into_shape_with_order(self.kernels_shape)
            .expect("the affine weights fold back to the kernel shape")
    }

    /// This layer's biases, one per output channel.
    pub fn biases(&self) -> ndarray::ArrayView1<'_, f32> {
        self.affine.biases()
    }

    /// The activation applied to this layer's output.
    pub fn activation(&self) -> &Arc<dyn Activation> {
        &self.activation
    }

    /// Builds a `Conv2d` layer from its configuration and tensors.
    /// # Arguments
    /// - `config`: Carries the `"activation"` name, the `"input_shape"` `(channels, height,
    ///   width)`, the `"stride"`, and the `"padding"`.
    /// - `tensors`: Carries the `"kernels"` (rank-4) and `"biases"` (rank-1) tensors.
    pub(super) fn from_config(
        config: &HashMap<String, String>,
        mut tensors: HashMap<String, ArrayD<f32>>,
    ) -> Result<Self, LayerConfigError> {
        let kernels = super::take_tensor::<Ix4>(&mut tensors, "kernels")?;
        let biases = super::take_tensor::<Ix1>(&mut tensors, "biases")?;

        let dims = super::config_dims(config, "input_shape")?;
        let [channels, height, width] = dims[..] else {
            return Err(LayerConfigError::InvalidConfig {
                key: "input_shape".to_string(),
                reason: format!("expected 3 dimensions, got {}", dims.len()),
            });
        };
        let stride = super::config_usize(config, "stride")?;
        let padding = super::config_usize(config, "padding")?;
        let activation = super::config_activation(config)?;

        Ok(Conv2d::new(
            kernels,
            biases,
            (channels, height, width),
            stride,
            padding,
            activation,
        ))
    }
}

impl Layer for Conv2d {
    fn forward(&self, input: ArrayViewD<f32>) -> ArrayD<f32> {
        let input = input
            .into_dimensionality::<Ix4>()
            .expect("Conv2d expects a 4D (channels, height, width, samples) input");
        let (in_channels, height, width, samples) = input.dim();
        assert_eq!(
            (in_channels, height, width),
            self.input_shape,
            "Conv2d input shape does not match its configured input shape."
        );

        let (out_channels, _, kh, kw) = self.kernels_shape;

        // im2col → affine matmul → fold: unfold each patch into a column, then the shared
        // affine map turns the whole batch into feature maps in one `weights · cols + bias`.
        let cols = im2col(input, kh, kw, self.stride, self.padding);
        let pre_activation = self.affine.forward(cols.view());

        let out_h = conv_output_dim(height, kh, self.stride, self.padding);
        let out_w = conv_output_dim(width, kw, self.stride, self.padding);
        self.activation
            .apply(pre_activation.into_dyn().view())
            .into_shape_with_order((out_channels, out_h, out_w, samples))
            .expect("the matmul result folds back to (out_channels, out_h, out_w, samples)")
            .into_dyn()
    }

    fn backward(
        &self,
        da: ArrayViewD<f32>,
        input: ArrayViewD<f32>,
        output: ArrayViewD<f32>,
        compute_input_gradient: bool,
    ) -> BackwardPass {
        let da = da
            .into_dimensionality::<Ix4>()
            .expect("Conv2d expects a 4D da");
        let input = input
            .into_dimensionality::<Ix4>()
            .expect("Conv2d expects a 4D input");
        let output = output
            .into_dimensionality::<Ix4>()
            .expect("Conv2d expects a 4D output");

        let (in_channels, height, width, samples) = input.dim();
        let (out_channels, _, kh, kw) = self.kernels_shape;
        let out_h = conv_output_dim(height, kh, self.stride, self.padding);
        let out_w = conv_output_dim(width, kw, self.stride, self.padding);
        let positions = out_h * out_w * samples;

        // Collapse the spatial output to the 2D (out_channels, positions) the activation VJP
        // and the affine math operate on, mirroring Dense.
        let da = da
            .to_shape((out_channels, positions))
            .expect("da folds to (out_channels, positions)");
        let output = output
            .to_shape((out_channels, positions))
            .expect("output folds to (out_channels, positions)");

        // The columns span spatial positions as well as samples, so the affine backward is told
        // to average over the sample count alone; col2im then folds the input gradient back.
        let dz = self
            .activation
            .vjp(da.view().into_dyn(), output.view().into_dyn())
            .into_dimensionality::<Ix2>()
            .expect("the VJP preserves the rank-2 (outputs, columns) shape");

        let cols = im2col(input, kh, kw, self.stride, self.padding);
        let (dw, db, dcols) = self.affine.backward(
            dz.view(),
            cols.view(),
            samples as f32,
            compute_input_gradient,
        );

        let input_gradient = dcols.map(|dcols| {
            col2im(
                dcols.view(),
                (in_channels, height, width, samples),
                kh,
                kw,
                self.stride,
                self.padding,
            )
            .into_dyn()
        });

        BackwardPass {
            gradients: LayerGradients(vec![dw.into_dyn(), db.into_dyn()]),
            input_gradient,
        }
    }

    fn parameters_mut(&mut self) -> Vec<Parameter<'_>> {
        let (weights, biases) = self.affine.parameters_mut();
        vec![
            Parameter {
                value: weights.view_mut().into_dyn(),
                decays: true,
            },
            Parameter {
                value: biases.view_mut().into_dyn(),
                decays: false,
            },
        ]
    }

    fn input_shape(&self) -> Vec<usize> {
        let (channels, height, width) = self.input_shape;
        vec![channels, height, width]
    }

    fn output_shape(&self) -> Vec<usize> {
        let (out_channels, _, kh, kw) = self.kernels_shape;
        let (_, height, width) = self.input_shape;
        vec![
            out_channels,
            conv_output_dim(height, kh, self.stride, self.padding),
            conv_output_dim(width, kw, self.stride, self.padding),
        ]
    }

    fn is_finite(&self) -> bool {
        self.affine.is_finite()
    }

    fn kind(&self) -> LayerKind {
        LayerKind::Conv2d
    }

    fn config(&self) -> Vec<(String, String)> {
        let (channels, height, width) = self.input_shape;
        vec![
            ("activation".to_string(), self.activation.name().to_string()),
            (
                "input_shape".to_string(),
                format!("{channels},{height},{width}"),
            ),
            ("stride".to_string(), self.stride.to_string()),
            ("padding".to_string(), self.padding.to_string()),
        ]
    }

    fn named_tensors(&self) -> Vec<(String, ArrayD<f32>)> {
        vec![
            ("kernels".to_string(), self.kernels().into_dyn()),
            (
                "biases".to_string(),
                self.affine.biases().to_owned().into_dyn(),
            ),
        ]
    }

    fn activation_name(&self) -> Option<&str> {
        Some(self.activation.name())
    }

    fn weight_matrix(&self) -> Option<ArrayView2<'_, f32>> {
        // A convolution has no single (output_size, input_size) weight matrix.
        None
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// The output extent along one spatial axis for the given kernel, stride, and padding.
fn conv_output_dim(input: usize, kernel: usize, stride: usize, padding: usize) -> usize {
    (input + 2 * padding - kernel) / stride + 1
}

/// Unfolds a `(in_channels, height, width, samples)` batch into the column matrix
/// `(in_channels·kh·kw, out_height·out_width·samples)` that turns a convolution into a matmul.
///
/// Each column is one receptive-field patch; row `(c·kh + u)·kw + v` is the tap at kernel
/// offset `(u, v)` of channel `c`, and taps falling in the zero-padding stay zero. Columns are
/// ordered `(out_height, out_width, samples)` in C-order, so reshaping a `(out_channels, cols)`
/// product back to `(out_channels, out_height, out_width, samples)` is a plain reshape.
fn im2col(
    input: ArrayView4<f32>,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
) -> Array2<f32> {
    let (in_channels, height, width, samples) = input.dim();
    let out_h = conv_output_dim(height, kh, stride, padding);
    let out_w = conv_output_dim(width, kw, stride, padding);
    let mut cols = Array2::zeros((in_channels * kh * kw, out_h * out_w * samples));

    for c in 0..in_channels {
        for u in 0..kh {
            for v in 0..kw {
                let row = (c * kh + u) * kw + v;
                for oh in 0..out_h {
                    let ih = (oh * stride + u) as isize - padding as isize;
                    if ih < 0 || ih >= height as isize {
                        continue;
                    }
                    for ow in 0..out_w {
                        let iw = (ow * stride + v) as isize - padding as isize;
                        if iw < 0 || iw >= width as isize {
                            continue;
                        }
                        for s in 0..samples {
                            let col = (oh * out_w + ow) * samples + s;
                            cols[[row, col]] = input[[c, ih as usize, iw as usize, s]];
                        }
                    }
                }
            }
        }
    }

    cols
}

/// Folds a column matrix `(in_channels·kh·kw, out_height·out_width·samples)` back into an input
/// `(in_channels, height, width, samples)`, summing the contributions of taps that share an
/// input position. The gradient counterpart of [`im2col`]: overlapping receptive fields
/// accumulate, which is exactly the gradient of the shared reads in the forward pass.
fn col2im(
    cols: ArrayView2<f32>,
    input_shape: (usize, usize, usize, usize),
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
) -> Array4<f32> {
    let (in_channels, height, width, samples) = input_shape;
    let out_h = conv_output_dim(height, kh, stride, padding);
    let out_w = conv_output_dim(width, kw, stride, padding);
    let mut input = Array4::zeros((in_channels, height, width, samples));

    for c in 0..in_channels {
        for u in 0..kh {
            for v in 0..kw {
                let row = (c * kh + u) * kw + v;
                for oh in 0..out_h {
                    let ih = (oh * stride + u) as isize - padding as isize;
                    if ih < 0 || ih >= height as isize {
                        continue;
                    }
                    for ow in 0..out_w {
                        let iw = (ow * stride + v) as isize - padding as isize;
                        if iw < 0 || iw >= width as isize {
                            continue;
                        }
                        for s in 0..samples {
                            let col = (oh * out_w + ow) * samples + s;
                            input[[c, ih as usize, iw as usize, s]] += cols[[row, col]];
                        }
                    }
                }
            }
        }
    }

    input
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::{RELU, SIGMOID};
    use ndarray::{Array, Array1, IxDyn, array};
    use ndarray_rand::rand::SeedableRng;
    use ndarray_rand::rand::rngs::StdRng;

    #[test]
    fn accessors_report_shapes_activation_and_parameters() {
        // 3 filters over a 2-channel 5×5 input, 3×3 kernel, stride 2, padding 1.
        // out_h = out_w = (5 + 2 - 3) / 2 + 1 = 3.
        let mut layer = Conv2d::new(
            Array4::zeros((3, 2, 3, 3)),
            Array1::zeros(3),
            (2, 5, 5),
            2,
            1,
            RELU.clone(),
        );
        assert_eq!(layer.input_shape(), vec![2, 5, 5]);
        assert_eq!(layer.output_shape(), vec![3, 3, 3]);
        assert_eq!(layer.input_size(), 2 * 5 * 5);
        assert_eq!(layer.output_size(), 3 * 3 * 3);
        assert_eq!(layer.kind(), LayerKind::Conv2d);
        assert_eq!(layer.activation_name(), Some("relu"));
        assert!(layer.weight_matrix().is_none());
        assert!(layer.is_finite());

        let tensors = layer.named_tensors();
        assert_eq!(tensors[0].0, "kernels");
        assert_eq!(tensors[0].1.shape(), &[3, 2, 3, 3]);
        assert_eq!(tensors[1].0, "biases");
        assert_eq!(tensors[1].1.shape(), &[3]);

        let params = layer.parameters_mut();
        assert_eq!(params.len(), 2);
        // Weights decay, biases do not; order matches the gradient order [dw, db]. The weight
        // parameter is the flat affine matrix (out_channels, in·kh·kw) = (3, 18), not the rank-4
        // kernels the named tensor exposes.
        assert!(params[0].decays);
        assert_eq!(params[0].value.shape(), &[3, 2 * 3 * 3]);
        assert!(!params[1].decays);
        assert_eq!(params[1].value.shape(), &[3]);
    }

    #[test]
    fn forward_matches_hand_computed_convolution() {
        // 1 channel, 3×3 input, a single all-ones 2×2 kernel, stride 1, no padding, zero
        // bias. ReLU is the identity on the (positive) window sums, so each output is the
        // sum of its 2×2 receptive field.
        let input = Array::from_shape_vec(
            IxDyn(&[1, 3, 3, 1]),
            vec![1., 2., 3., 4., 5., 6., 7., 8., 9.],
        )
        .unwrap();
        let layer = Conv2d::new(
            Array4::ones((1, 1, 2, 2)),
            array![0.0],
            (1, 3, 3),
            1,
            0,
            RELU.clone(),
        );

        let output = layer.forward(input.view());

        assert_eq!(output.shape(), &[1, 2, 2, 1]);
        // windows: [1,2,4,5]=12, [2,3,5,6]=16, [4,5,7,8]=24, [5,6,8,9]=28.
        let expected =
            Array::from_shape_vec(IxDyn(&[1, 2, 2, 1]), vec![12., 16., 24., 28.]).unwrap();
        assert_eq!(output, expected);
    }

    #[test]
    fn backward_gradients_match_numerical_approximation() {
        // Isolated conv layer exercised with stride 2 and padding 1 over a 2-channel 4×4
        // input — the configuration that stresses the im2col/col2im index arithmetic
        // (out_h = out_w = (4 + 2 - 3) / 2 + 1 = 2). With an upstream gradient of all ones,
        // the loss is L = sum(output); finite differences of that sum recover the analytical
        // gradients. backward divides the parameter gradients by the sample count, so the
        // numerical estimate (which does not) is compared against `grad * m`.
        let mut rng = StdRng::seed_from_u64(7);
        let layer = Conv2d::initialization((2, 4, 4), 3, (3, 3), 2, 1, SIGMOID.clone(), &mut rng);
        let input = Array::from_shape_fn(IxDyn(&[2, 4, 4, 2]), |idx| {
            // Deterministic spread of values in a sensible range.
            let flat = idx[0] * 100 + idx[1] * 10 + idx[2] + idx[3];
            ((flat % 13) as f32 - 6.0) * 0.1
        });
        let m = input.shape()[3] as f32;

        let output = layer.forward(input.view());
        let da = ArrayD::<f32>::ones(output.raw_dim());
        let pass = layer.backward(da.view(), input.view(), output.view(), true);
        let grads = pass.gradients;
        let da_prev = pass.input_gradient.expect("input gradient requested");

        let loss = |layer: &Conv2d| layer.forward(input.view()).sum();
        // The summed loss is ~O(10), so a smaller step drowns the central difference in f32
        // cancellation noise; 1e-2 keeps the signal well above it while truncation stays tiny.
        let eps = 1e-2_f32;
        let tolerance = 5e-2_f32;
        let check = |analytical: f32, numerical: f32, label: String| {
            let rel = (numerical - analytical).abs()
                / (numerical.abs().max(analytical.abs()).max(1e-2) + 1e-8);
            assert!(
                rel < tolerance,
                "{label}: analytical={analytical:.6}, numerical={numerical:.6}"
            );
        };

        // Weight gradients: perturb each entry of the flat affine matrix (out_channels, in·kh·kw),
        // central-difference the loss, compare to `grad * m`.
        let (out_c, fan_in) = layer.affine.weights().dim();
        for o in 0..out_c {
            for j in 0..fan_in {
                let mut plus = layer.clone();
                plus.affine.weights_mut()[[o, j]] += eps;
                let mut minus = layer.clone();
                minus.affine.weights_mut()[[o, j]] -= eps;
                let numerical = (loss(&plus) - loss(&minus)) / (2.0 * eps);
                check(grads[0][[o, j]] * m, numerical, format!("dw[{o},{j}]"));
            }
            let mut plus = layer.clone();
            plus.affine.biases_mut()[o] += eps;
            let mut minus = layer.clone();
            minus.affine.biases_mut()[o] -= eps;
            let numerical = (loss(&plus) - loss(&minus)) / (2.0 * eps);
            check(grads[1][o] * m, numerical, format!("db[{o}]"));
        }

        // Input gradient: perturb each input entry (no `* m` — it chains straight upstream).
        for idx in ndarray::indices(input.shape()) {
            let idx = [idx[0], idx[1], idx[2], idx[3]];
            let mut plus = input.clone();
            plus[idx] += eps;
            let mut minus = input.clone();
            minus[idx] -= eps;
            let numerical = (layer.forward(plus.view()).sum() - layer.forward(minus.view()).sum())
                / (2.0 * eps);
            check(da_prev[IxDyn(&idx)], numerical, format!("da_prev{idx:?}"));
        }
    }

    #[test]
    fn backward_skips_input_gradient_when_not_requested() {
        let mut rng = StdRng::seed_from_u64(1);
        let layer = Conv2d::initialization((1, 4, 4), 2, (2, 2), 1, 0, RELU.clone(), &mut rng);
        let input = ArrayD::<f32>::ones(IxDyn(&[1, 4, 4, 3]));
        let output = layer.forward(input.view());

        let da = ArrayD::<f32>::ones(output.raw_dim());
        let pass = layer.backward(da.view(), input.view(), output.view(), false);

        assert!(pass.input_gradient.is_none());
        // Kernels and biases both produce a gradient.
        assert_eq!(pass.gradients.len(), 2);
    }

    #[test]
    fn initialization_is_reproducible_and_correctly_shaped() {
        let build = || {
            let mut rng = StdRng::seed_from_u64(42);
            Conv2d::initialization((3, 8, 8), 5, (3, 3), 1, 1, RELU.clone(), &mut rng)
        };
        let a = build();
        let b = build();

        assert_eq!(a.kernels().dim(), (5, 3, 3, 3));
        assert_eq!(a.biases().len(), 5);
        // Same seed and architecture yield identical kernels.
        assert_eq!(a.kernels(), b.kernels());
        // He initialization leaves the biases at zero.
        assert!(a.biases().iter().all(|&b| b == 0.0));
    }

    #[test]
    #[should_panic(expected = "Conv2d expects a 4D")]
    fn forward_rejects_non_4d_input() {
        let layer = Conv2d::new(
            Array4::ones((1, 1, 2, 2)),
            array![0.0],
            (1, 3, 3),
            1,
            0,
            RELU.clone(),
        );
        let input = ArrayD::<f32>::zeros(IxDyn(&[1, 3, 3]));
        layer.forward(input.view());
    }

    #[test]
    #[should_panic(expected = "input shape does not match")]
    fn forward_rejects_mismatched_input_shape() {
        let layer = Conv2d::new(
            Array4::ones((1, 1, 2, 2)),
            array![0.0],
            (1, 3, 3),
            1,
            0,
            RELU.clone(),
        );
        // Height 4 where the layer was configured for 3.
        let input = ArrayD::<f32>::zeros(IxDyn(&[1, 4, 3, 2]));
        layer.forward(input.view());
    }

    #[test]
    #[should_panic(expected = "does not fit the padded input")]
    fn new_rejects_kernel_larger_than_padded_input() {
        // 4×4 kernel over a 3×3 input with no padding does not fit.
        Conv2d::new(
            Array4::ones((1, 1, 4, 4)),
            array![0.0],
            (1, 3, 3),
            1,
            0,
            RELU.clone(),
        );
    }

    #[test]
    fn config_and_tensors_round_trip_through_from_config() {
        // 3 filters over a 2-channel 5×5 input, 3×3 kernel, stride 2, padding 1.
        let mut rng = StdRng::seed_from_u64(7);
        let layer = Conv2d::initialization((2, 5, 5), 3, (3, 3), 2, 1, RELU.clone(), &mut rng);

        let config: HashMap<String, String> = layer.config().into_iter().collect();
        let tensors: HashMap<String, ArrayD<f32>> = layer.named_tensors().into_iter().collect();
        let rebuilt = Conv2d::from_config(&config, tensors).unwrap();

        // Same geometry and parameters: forward on an arbitrary batch matches bit-for-bit.
        let input = Array::from_shape_fn(IxDyn(&[2, 5, 5, 4]), |d| {
            (d[0] + d[1] + d[2] + d[3]) as f32 * 0.1
        });
        assert_eq!(layer.forward(input.view()), rebuilt.forward(input.view()));
        assert_eq!(layer.config(), rebuilt.config());
    }

    #[test]
    fn from_config_rejects_missing_input_shape() {
        let mut rng = StdRng::seed_from_u64(1);
        let layer = Conv2d::initialization((1, 4, 4), 2, (3, 3), 1, 0, RELU.clone(), &mut rng);
        let tensors: HashMap<String, ArrayD<f32>> = layer.named_tensors().into_iter().collect();
        // Drop "input_shape" from an otherwise valid config.
        let config: HashMap<String, String> = layer
            .config()
            .into_iter()
            .filter(|(key, _)| key != "input_shape")
            .collect();

        assert_eq!(
            Conv2d::from_config(&config, tensors).unwrap_err(),
            LayerConfigError::MissingConfig("input_shape".to_string())
        );
    }

    #[test]
    fn from_config_rejects_input_shape_without_three_dimensions() {
        let mut rng = StdRng::seed_from_u64(2);
        let layer = Conv2d::initialization((1, 4, 4), 2, (3, 3), 1, 0, RELU.clone(), &mut rng);
        let tensors: HashMap<String, ArrayD<f32>> = layer.named_tensors().into_iter().collect();
        // A 2-dimension input_shape where a convolution needs (channels, height, width).
        let mut config: HashMap<String, String> = layer.config().into_iter().collect();
        config.insert("input_shape".to_string(), "4,4".to_string());

        assert_eq!(
            Conv2d::from_config(&config, tensors).unwrap_err(),
            LayerConfigError::InvalidConfig {
                key: "input_shape".to_string(),
                reason: "expected 3 dimensions, got 2".to_string(),
            }
        );
    }
}
