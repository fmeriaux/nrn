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
//! - **Binary Classification**: Identifies points where prediction ≈ 0.5
//! - **Multi-class Classification**: Identifies points where the top two class probabilities are nearly equal
//!
//! Note: This module only computes coordinate points. For visualization, the computed points
//! can be used with external plotting libraries.

use crate::model::NeuralNetwork;
use ndarray::Array2;

/// Computes decision boundary points for a trained neural network with custom tolerance.
///
/// This function generates a grid of points within the specified bounds and identifies
/// points where the model's prediction lies near decision boundaries.
///
/// # Arguments
///
/// * `mins` - Minimum values for each input dimension
/// * `maxs` - Maximum values for each input dimension
/// * `model` - Trained neural network to analyze
/// * `resolution` - Number of grid points per dimension (higher = more precise boundary)
/// * `tolerance` - Acceptable deviation from decision threshold (e.g., 0.001 for high precision)
///
/// # Returns
///
/// A vector of points (each point is a Vec<f32>) that lie approximately on the decision boundary.
///
/// # Classification Behavior
///
/// - **Binary**: Points where `|prediction - 0.5| < tolerance`
/// - **Multi-class**: Points where `|max_prob - second_max_prob| < tolerance`
///
/// # Examples
///
/// ```rust
/// // High precision boundary detection
/// let boundary_points = decision_boundary(&mins, &maxs, &model, 100, 0.0005);
///
/// // Lower precision for faster computation
/// let boundary_points = decision_boundary(&mins, &maxs, &model, 50, 0.01);
/// ```
pub fn decision_boundary(
    mins: &[f32],
    maxs: &[f32],
    model: &NeuralNetwork,
    resolution: usize,
    tolerance: f32,
) -> Vec<Vec<f32>> {
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

    if mins.len() == 0 {
        return vec![];
    }

    assert!(
        mins.iter().zip(maxs.iter()).all(|(&min, &max)| min < max),
        "Each min value must be less than the corresponding max value"
    );

    let (grid_points, inputs) = make_grid_and_inputs(mins, maxs, resolution);
    let predictions = model.predict(inputs.view());

    let n_outputs = predictions.nrows();

    grid_points
        .iter()
        .enumerate()
        .filter(|(i, _)| {
            let pred = predictions.column(*i);
            if n_outputs == 1 {
                // Binary classification: threshold at 0.5
                (pred[0] - 0.5).abs() < tolerance
            } else {
                // Multi-class classification: difference between top two classes
                let mut probabilities: Vec<f32> = pred.to_vec();
                probabilities.sort_by(|a, b| b.partial_cmp(a).unwrap());
                (probabilities[0] - probabilities[1]).abs() < tolerance
            }
        })
        .map(|(_, pt)| pt.clone())
        .collect()
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
/// - Vector of grid points (each point as Vec<f32>)
/// - Array2<f32> formatted for neural network input (dimensions × total_points)
fn make_grid_and_inputs(
    mins: &[f32],
    maxs: &[f32],
    resolution: usize,
) -> (Vec<Vec<f32>>, Array2<f32>) {
    let n_dims = mins.len();

    // Calculate steps for each dimension
    // Use (resolution-1) to include both min and max bounds
    let steps: Vec<f32> = mins
        .iter()
        .zip(maxs.iter())
        .map(|(&min, &max)| (max - min) / (resolution - 1) as f32)
        .collect();

    // Generate all grid points using recursive backtracking
    let total_points = resolution.pow(n_dims as u32);
    let mut grid_points = Vec::with_capacity(total_points);

    generate_points_recursive(
        &mins,
        &steps,
        resolution,
        n_dims,
        &mut vec![],
        &mut grid_points,
    );

    // Reorganize grid points by dimension for model input format
    // Example: points [(x1,y1), (x2,y2)] become dims [[x1,x2], [y1,y2]]
    let mut flat = Vec::with_capacity(n_dims * total_points);
    for dim in 0..n_dims {
        for point in &grid_points {
            flat.push(point[dim]);
        }
    }

    let inputs = Array2::from_shape_vec((n_dims, total_points), flat).unwrap();
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
