use ndarray::{Array2, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand::Rng;
use ndarray_rand::rand_distr::{StandardNormal, Uniform};

/// Create a random set of points uniformly distributed on a sphere with a given radius range
/// # Arguments
/// - `rng`: A mutable reference to a random number generator.
/// - `n_points`: The number of points to generate.
/// - `centers`: A slice of f32 representing the center of the sphere.
/// - `radius_min`: The minimum radius of the sphere.
/// - `radius_max`: The maximum radius of the sphere.
pub(crate) fn random_points(
    rng: &mut impl Rng,
    n_points: usize,
    centers: &[f32],
    radius_min: f32,
    radius_max: f32,
) -> Array2<f32> {
    let n_features = centers.len();

    let mut points = Array2::<f32>::random_using((n_points, n_features), StandardNormal, rng);

    // Normalize each row to unit length, ensuring points are uniformly distributed on the sphere
    for mut row in points.axis_iter_mut(Axis(0)) {
        let norm = row.iter().map(|v| v.powi(2)).sum::<f32>().sqrt();
        if norm != 0.0 {
            row.iter_mut().for_each(|v| *v /= norm);
        }
        // For uniform distribution, we scale the points to the desired radius
        let u: f32 = rng.sample(Uniform::new(0.0, 1.0));
        let r = ((radius_max.powf(n_features as f32) - radius_min.powf(n_features as f32)) * u
            + radius_min.powf(n_features as f32))
        .powf(1.0 / n_features as f32);
        row.iter_mut().for_each(|v| *v *= r);
        // Translate the points by the centers
        for (v, c) in row.iter_mut().zip(centers.iter()) {
            *v += c;
        }
    }

    points
}
