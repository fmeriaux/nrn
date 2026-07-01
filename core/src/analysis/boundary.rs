//! Decision boundary analysis for neural networks.
//!
//! This module provides functionality to compute decision boundaries of trained neural networks.
//! Decision boundaries are the regions in input space where a classifier changes its prediction from one class to another.
//! They are essential for understanding how a model partitions the feature space and can reveal insights about
//! model behavior, overfitting, and generalization capabilities.
//!
//! ## Data Layout Convention
//! Matrices are organized with the primary dimension (features/outputs) along rows and samples along columns.
//!
//! ## Classification Support
//!
//! - **Binary Classification**: Detects edges where the prediction crosses 0.5
//! - **Multi-class Classification**: Detects edges where the winning class changes
//!
//! Note: This module only computes coordinate points. For visualization, the computed points
//! can be used with external plotting libraries.

use crate::model::Predictor;
use ndarray::{Array2, ArrayView1};

impl Predictor {
    /// Computes decision boundary points within the given bounds.
    ///
    /// Samples a grid over the bounds and detects, on each axis-aligned edge between
    /// neighboring grid points, where the predicted class flips. The crossing point is
    /// located by linear interpolation along the edge, so the boundary stays a crisp,
    /// one-cell-wide curve regardless of how sharp the model's transition is.
    ///
    /// # Arguments
    ///
    /// * `mins` - Minimum values for each input dimension
    /// * `maxs` - Maximum values for each input dimension
    /// * `resolution` - Number of grid points per dimension (higher = smoother curve)
    ///
    /// # Returns
    ///
    /// An `Array2<f32>` where each row is a point on the decision boundary, in raw
    /// (unscaled) input coordinates.
    ///
    /// # Classification Behavior
    ///
    /// - **Binary**: Edges where the prediction crosses 0.5
    /// - **Multi-class**: Edges where the winning class (argmax) changes
    ///
    pub fn decision_boundary(&self, mins: &[f32], maxs: &[f32], resolution: usize) -> Array2<f32> {
        assert!(
            resolution >= 2,
            "Resolution must be at least 2 for meaningful boundary analysis. Got: {}",
            resolution
        );

        assert_eq!(
            mins.len(),
            maxs.len(),
            "Mins and maxs must have the same length"
        );

        if mins.is_empty() {
            return Array2::zeros((0, 0));
        }

        assert!(
            mins.iter().zip(maxs.iter()).all(|(&min, &max)| min < max),
            "Each min value must be less than the corresponding max value"
        );

        let n_dims = mins.len();
        let (grid_points, inputs) = make_grid_and_inputs(mins, maxs, resolution);
        let predictions = self
            .predict(inputs.view())
            .expect("grid dimensionality matches the network input size");

        // Flat-index stride to reach the next grid point along each dimension.
        let strides: Vec<usize> = (0..n_dims)
            .map(|d| resolution.pow((n_dims - 1 - d) as u32))
            .collect();

        let mut points: Vec<f32> = Vec::new();
        let mut n_points = 0;

        for idx in 0..grid_points.nrows() {
            let here = predictions.column(idx);
            for &stride in &strides {
                // Skip dimensions already at their last grid index (no forward neighbor).
                if (idx / stride) % resolution == resolution - 1 {
                    continue;
                }
                let next = idx + stride;
                if let Some(t) = edge_crossing(here, predictions.column(next)) {
                    let (a, b) = (grid_points.row(idx), grid_points.row(next));
                    points.extend((0..n_dims).map(|k| a[k] + t * (b[k] - a[k])));
                    n_points += 1;
                }
            }
        }

        Array2::from_shape_vec((n_points, n_dims), points).unwrap()
    }
}

/// The interpolation parameter `t ∈ [0, 1]` along the edge `a → b` where the decision
/// boundary crosses, or `None` when both endpoints predict the same class.
fn edge_crossing(a: ArrayView1<f32>, b: ArrayView1<f32>) -> Option<f32> {
    // Binary: crossing where the prediction passes through 0.5.
    if a.len() == 1 {
        let (ma, mb) = (a[0] - 0.5, b[0] - 0.5);
        return ((ma < 0.0) != (mb < 0.0)).then(|| ma / (ma - mb));
    }

    // Multi-class: crossing where the winning class changes. The margin is the lead of
    // `a`'s winner over `b`'s winner, which is non-negative at `a` and non-positive at `b`.
    let (wa, wb) = (argmax(a), argmax(b));
    if wa == wb {
        return None;
    }
    // `a[wa] >= a[wb]` and `b[wb] >= b[wa]` (each winner leads its own column), and the
    // winners differ, so the margins span zero with a strictly positive denominator.
    let (ma, mb) = (a[wa] - a[wb], b[wa] - b[wb]);
    Some(ma / (ma - mb))
}

/// The index of the largest element, picking the first on ties.
fn argmax(values: ArrayView1<f32>) -> usize {
    values
        .iter()
        .enumerate()
        .fold(0, |best, (i, &v)| if v > values[best] { i } else { best })
}

/// Creates a grid of points and corresponding inputs for neural network evaluation.
///
/// This function generates a uniform grid covering the specified n-dimensional space
/// and formats the data for efficient neural network batch processing.
///
/// # Arguments
///
/// * `mins` - Minimum bounds for each dimension
/// * `maxs` - Maximum bounds for each dimension
/// * `resolution` - Number of points per dimension (minimum 2 for meaningful boundary analysis)
///
/// # Returns
///
/// A tuple containing:
/// - Array2<f32> of grid points (total_points × dimensions)
/// - Array2<f32> formatted for neural network input (dimensions × total_points)
fn make_grid_and_inputs(
    mins: &[f32],
    maxs: &[f32],
    resolution: usize,
) -> (Array2<f32>, Array2<f32>) {
    let n_dims = mins.len();

    // Calculate steps for each dimension
    // Use (resolution-1) to include both min and max bounds
    let steps: Vec<f32> = mins
        .iter()
        .zip(maxs.iter())
        .map(|(&min, &max)| (max - min) / (resolution - 1) as f32)
        .collect();

    // Generate all grid points using recursive backtracking
    let total_points = resolution
        .checked_pow(n_dims as u32)
        .expect("Grid too large: resolution^n_dims overflows usize");
    let mut nested_points = Vec::with_capacity(total_points);

    generate_points_recursive(
        mins,
        &steps,
        resolution,
        n_dims,
        &mut vec![],
        &mut nested_points,
    );

    // Reorganize grid points by dimension for model input format
    // Example: points [(x1,y1), (x2,y2)] become dims [[x1,x2], [y1,y2]]
    let mut flat = Vec::with_capacity(n_dims * total_points);
    for dim in 0..n_dims {
        for point in &nested_points {
            flat.push(point[dim]);
        }
    }

    let inputs = Array2::from_shape_vec((n_dims, total_points), flat).unwrap();
    let grid_points =
        Array2::from_shape_vec((total_points, n_dims), nested_points.concat()).unwrap();
    (grid_points, inputs)
}

/// Recursively generates all grid points using backtracking algorithm.
///
/// This function implements a systematic approach to generate all possible combinations
/// of coordinates within the specified grid. It uses recursive backtracking to build
/// each point dimension by dimension, ensuring complete coverage of the n-dimensional space.
///
/// # Algorithm
///
/// 1. **Base Case**: When a point has coordinates for all dimensions, add it to results
/// 2. **Recursive Step**: For the current dimension, try all possible values (0 to resolution-1)
/// 3. **Coordinate Calculation**: `coordinate = min + (index × step)`
/// 4. **Backtrack**: Remove the last coordinate and try the next value
///
/// # Arguments
///
/// * `mins` - Minimum bounds for each dimension
/// * `steps` - Step size for each dimension (calculated as (max-min)/resolution)
/// * `resolution` - Number of points per dimension
/// * `n_dims` - Total number of dimensions
/// * `current_point` - Mutable reference to the point being built
/// * `result` - Mutable reference to store all generated points
///
/// # Examples
///
/// For a 2D grid with resolution=2, mins=[0.0, 0.0], maxs=[1.0, 1.0], this generates points:
/// - [0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]
///
fn generate_points_recursive(
    mins: &[f32],
    steps: &[f32],
    resolution: usize,
    n_dims: usize,
    current_point: &mut Vec<f32>,
    result: &mut Vec<Vec<f32>>,
) {
    // Base case: point is complete when it has coordinates for all dimensions
    if current_point.len() == n_dims {
        result.push(current_point.clone());
        return;
    }

    let dim = current_point.len();
    for i in 0..resolution {
        // Calculate coordinate: min + (index * step)
        let value = mins[dim] + (i as f32) * steps[dim];
        current_point.push(value);
        generate_points_recursive(mins, steps, resolution, n_dims, current_point, result);
        // Backtrack: remove coordinate to try next value in current dimension
        current_point.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::{SIGMOID, SOFTMAX};
    use crate::model::{NeuralNetwork, NeuronLayer};
    use ndarray::array;

    #[test]
    #[should_panic(expected = "Grid too large")]
    #[cfg(target_pointer_width = "64")]
    fn overflow_panics_with_clear_message() {
        // 2^65 > usize::MAX on 64-bit — without checked_pow this silently wraps to 0
        let n = (usize::BITS + 1) as usize;
        let mins = vec![0.0f32; n];
        let maxs = vec![1.0f32; n];
        make_grid_and_inputs(&mins, &maxs, 2);
    }

    #[test]
    fn small_2d_grid_has_correct_shape() {
        // resolution=3, n_dims=2 → 3^2=9 points, each with 2 coords
        let (grid, inputs) = make_grid_and_inputs(&[0.0, 0.0], &[1.0, 1.0], 3);
        assert_eq!(grid.shape(), &[9, 2]);
        assert_eq!(inputs.shape(), &[2, 9]);
    }

    /// A 2-input → 1-output sigmoid predictor. With weights [1, 0] and zero bias,
    /// the prediction is `sigmoid(x0)`, which equals exactly 0.5 when x0 == 0.
    fn binary_model() -> Predictor {
        Predictor::new(
            NeuralNetwork {
                layers: vec![NeuronLayer {
                    weights: array![[1.0, 0.0]],
                    biases: array![0.0],
                    activation: SIGMOID.clone(),
                }],
            },
            None,
        )
    }

    /// A 2-input → 3-output softmax predictor with symmetric weights for two of the
    /// classes, so a tie (equal top-two probabilities) occurs along x0 == 0.
    fn multiclass_model() -> Predictor {
        Predictor::new(
            NeuralNetwork {
                layers: vec![NeuronLayer {
                    weights: array![[1.0, 0.0], [-1.0, 0.0], [0.0, 0.0]],
                    biases: array![0.0, 0.0, -10.0],
                    activation: SOFTMAX.clone(),
                }],
            },
            None,
        )
    }

    #[test]
    fn binary_boundary_returns_points_near_threshold() {
        // Odd resolution includes x0 == 0, where sigmoid(x0) == 0.5 exactly.
        let boundary = binary_model().decision_boundary(&[-1.0, -1.0], &[1.0, 1.0], 5);
        assert!(boundary.nrows() > 0);
        assert_eq!(boundary.ncols(), 2);
        // Every returned point sits on the x0 == 0 line.
        assert!(boundary.column(0).iter().all(|&x| x.abs() < 1e-5));
    }

    #[test]
    fn multiclass_boundary_returns_points_on_the_tie_line() {
        let boundary = multiclass_model().decision_boundary(&[-2.0, -2.0], &[2.0, 2.0], 5);
        assert!(boundary.nrows() > 0);
        assert_eq!(boundary.ncols(), 2);
        assert!(boundary.column(0).iter().all(|&x| x.abs() < 1e-5));
    }

    #[test]
    fn scaler_shifts_the_boundary_into_raw_coordinates() {
        use crate::data::scalers::{MinMaxScaler, ScalerMethod};

        // Network: sigmoid(scaled_x0 - 0.5) == 0.5 when scaled_x0 == 0.5.
        let network = NeuralNetwork {
            layers: vec![NeuronLayer {
                weights: array![[1.0, 0.0]],
                biases: array![-0.5],
                activation: SIGMOID.clone(),
            }],
        };
        // MinMax fitted on raw x0 ∈ [0, 2]: scaled 0.5 maps back to raw 1.0.
        let scaler = ScalerMethod::MinMax(
            MinMaxScaler::default().fit(array![[0.0, 0.0], [2.0, 2.0]].view()),
        );
        let predictor = Predictor::new(network, Some(scaler));

        // Grid x0 ∈ {-1, 0, 1, 2, 3}; the boundary sits on raw x0 == 1.0.
        let boundary = predictor.decision_boundary(&[-1.0, -1.0], &[3.0, 3.0], 5);
        assert!(boundary.nrows() > 0);
        assert!(boundary.column(0).iter().all(|&x| (x - 1.0).abs() < 1e-5));
    }

    #[test]
    fn empty_bounds_return_empty_array() {
        let boundary = binary_model().decision_boundary(&[], &[], 5);
        assert_eq!(boundary.shape(), &[0, 0]);
    }

    #[test]
    #[should_panic(expected = "Resolution must be at least 2")]
    fn resolution_below_two_panics() {
        binary_model().decision_boundary(&[0.0], &[1.0], 1);
    }

    #[test]
    #[should_panic(expected = "Mins and maxs must have the same length")]
    fn mismatched_bounds_length_panics() {
        binary_model().decision_boundary(&[0.0, 0.0], &[1.0], 3);
    }

    #[test]
    #[should_panic(expected = "Each min value must be less than")]
    fn min_not_below_max_panics() {
        binary_model().decision_boundary(&[1.0, 0.0], &[1.0, 1.0], 3);
    }
}
