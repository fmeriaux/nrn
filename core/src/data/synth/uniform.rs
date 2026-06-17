use crate::data::synth::{SynthParams, calculate_radius, feature_bounds, random_points};
use ndarray::{Array2, s};
use ndarray_rand::rand::{Rng, RngCore};

/// Fills `features` with isotropic blobs: one spherical cluster (class) per
/// cluster, each centered at a random point within the bounded region.
pub(super) fn fill(params: &SynthParams, features: &mut Array2<f32>, mut rng: &mut dyn RngCore) {
    let samples_per_cluster = params.samples_per_cluster();
    let radius = calculate_radius(params.feature_min, params.feature_max, params.n_clusters);
    let radii = vec![(0.0, radius); params.n_clusters];

    let (feature_min_bound, feature_max_bound) =
        feature_bounds(params.feature_min, params.feature_max, &radii);

    let centers = Array2::<f32>::from_shape_fn((params.n_clusters, params.n_features), |_| {
        rng.random_range(feature_min_bound..feature_max_bound)
    });

    for (cluster_idx, (center, &(radius_min, radius_max))) in
        centers.outer_iter().zip(radii.iter()).enumerate()
    {
        let points = random_points(
            &mut rng,
            samples_per_cluster,
            &center.to_vec(),
            radius_min,
            radius_max,
        );
        let start = cluster_idx * samples_per_cluster;
        let end = start + samples_per_cluster;
        features.slice_mut(s![start..end, ..]).assign(&points);
    }
}
