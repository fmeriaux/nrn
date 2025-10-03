mod ring;
mod uniform;

pub use ring::RingDataset;
pub use uniform::UniformDataset;

use crate::data::dataset::Dataset;
use ndarray::{Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand::prelude::StdRng;
use ndarray_rand::rand::{Rng, RngCore, SeedableRng};
use ndarray_rand::rand_distr::{StandardNormal, Uniform};

pub trait DatasetGenerator {
    fn generate_rng(&self, rng: &mut dyn RngCore) -> Dataset;

    /// Generate a dataset using a specific seed for the random number generator.
    /// # Arguments
    /// - `seed`: A seed for the random number generator to ensure reproducibility.
    fn generate(&self, seed: u64) -> Dataset {
        let mut rng = StdRng::seed_from_u64(seed);
        self.generate_rng(&mut rng)
    }
}

/// Initializes the features and labels for a dataset.
fn init_features_and_labels(
    n_features: usize,
    n_clusters: usize,
    samples_per_cluster: usize,
) -> (Array2<f32>, Array1<f32>) {
    assert!(n_features > 0, "n_features must be greater than zero");
    assert!(n_clusters > 0, "n_clusters must be greater than zero");
    assert!(
        samples_per_cluster > 0,
        "samples_per_cluster must be greater than zero"
    );

    let total_samples = samples_per_cluster * n_clusters;
    let labels = Array1::from_shape_fn(total_samples, |i| (i / samples_per_cluster) as f32);
    let features = Array2::<f32>::zeros((total_samples, n_features));

    (features, labels)
}

/// Calculates the maximum radius from a list of radius ranges.
fn max_radius(radii: &[(f32, f32)]) -> f32 {
    assert!(!radii.is_empty(), "radii must not be empty");

    assert!(
        radii.iter().all(|&(min, max)| min >= 0.0 && max > min),
        "Each radius must be a valid range with min >= 0 and max > min"
    );

    radii.iter().map(|&(_, max)| max).fold(0.0, f32::max)
}

/// Calculates the bounds for feature values based on the provided minimum and maximum feature values
fn feature_bounds(feature_min: f32, feature_max: f32, radii: &[(f32, f32)]) -> (f32, f32) {
    assert!(
        feature_min < feature_max,
        "feature_min must be less than feature_max"
    );

    let max_radius = max_radius(radii);
    let feature_min_bound = feature_min + max_radius;
    let feature_max_bound = feature_max - max_radius;

    assert!(
        feature_max_bound > feature_min_bound,
        "feature_max_bound must be greater than feature_min_bound"
    );

    (feature_min_bound, feature_max_bound)
}

/// Calculates an optimal radius for clusters based on the feature range and number of clusters.
fn calculate_radius(feature_min: f32, feature_max: f32, n_clusters: usize) -> f32 {
    assert!(n_clusters > 0, "n_clusters must be greater than zero");
    (feature_max - feature_min) / (2.5 * n_clusters as f32)
}

/// Create a random set of points uniformly distributed on a sphere with a given radius range
/// # Arguments
/// - `rng`: A mutable reference to a random number generator.
/// - `n_points`: The number of points to generate.
/// - `centers`: A slice of f32 representing the center of the sphere.
/// - `radius_min`: The minimum radius of the sphere.
/// - `radius_max`: The maximum radius of the sphere.
pub fn random_points(
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
