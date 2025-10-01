use ndarray::Array2;
use crate::model::NeuralNetwork;

pub fn decision_boundary(mins: &[f32], maxs: &[f32], resolution: usize, model: &NeuralNetwork) -> Vec<Vec<f32>> {
    let (grid_points, inputs) = make_grid_and_inputs(mins, maxs, resolution);
    let predictions = model.predict(inputs.view());

    grid_points
        .iter()
        .enumerate()
        .filter(|(i, _)| {
            let pred = predictions.column(*i);
            pred.iter().any(|&p| (p - 0.5).abs() < 0.001)
        })
        .map(|(_, pt)| pt.clone())
        .collect()
}


/// Creates a grid of points and corresponding inputs for a neural network in n dimensions.
fn make_grid_and_inputs(mins: &[f32], maxs: &[f32], resolution: usize) -> (Vec<Vec<f32>>, Array2<f32>) {
    assert_eq!(mins.len(), maxs.len(), "mins and maxs must have the same length");
    let n_dims = mins.len();

    if n_dims == 0 {
        return (vec![], Array2::zeros((0, 0)));
    }

    // Calculate steps for each dimension
    let steps: Vec<f32> = mins.iter()
        .zip(maxs.iter())
        .map(|(&min, &max)| (max - min) / resolution as f32)
        .collect();

    // Generate all grid points using recursive backtracking
    let total_points = resolution.pow(n_dims as u32);
    let mut grid_points = Vec::with_capacity(total_points);

    generate_points_recursive(&mins, &steps, resolution, n_dims, &mut vec![], &mut grid_points);


    // Reshape data for neural network: group coordinates by dimension
    // Transform from points [[x1,y1], [x2,y2]] to dimensions [[x1,x2], [y1,y2]]
    let mut flat = Vec::with_capacity(n_dims * total_points);
    for dim in 0..n_dims {
        for point in &grid_points {
            flat.push(point[dim]);
        }
    }

    let inputs = Array2::from_shape_vec((n_dims, total_points), flat).unwrap();
    (grid_points, inputs)
}


/// Recursively generates all grid points using backtracking.
///
/// The algorithm builds each point dimension by dimension, then backtracks
/// to explore all combinations. For each dimension, it calculates:
/// `coordinate = min + (index * step)`
fn generate_points_recursive(
    mins: &[f32],
    steps: &[f32],
    resolution: usize,
    n_dims: usize,
    current_point: &mut Vec<f32>,
    result: &mut Vec<Vec<f32>>
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