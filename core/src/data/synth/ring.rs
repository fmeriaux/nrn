use crate::data::synth::{SynthParams, calculate_radius, feature_bounds, random_points};
use ndarray::{Array1, Array2, s};
use ndarray_rand::rand::{Rng, RngCore};

/// Fills `features` with concentric rings sharing a common center, one ring
/// (class) per cluster. `overlap` controls how much consecutive rings overlap
/// (negative values leave a gap).
pub(super) fn fill(
    params: &SynthParams,
    overlap: f32,
    features: &mut Array2<f32>,
    mut rng: &mut dyn RngCore,
) {
    let radius = calculate_radius(params.feature_min, params.feature_max, params.n_clusters);
    let samples_per_cluster = params.samples_per_cluster();

    // Increasing radii for each cluster; `overlap` controls how much consecutive
    // rings overlap.
    let radii: Vec<(f32, f32)> = (0..params.n_clusters)
        .map(|i| {
            let idx = i as f32;
            let min = radius * idx * (1.0 - overlap);
            let max = min + radius;
            (min, max)
        })
        .collect();

    let (feature_min_bound, feature_max_bound) =
        feature_bounds(params.feature_min, params.feature_max, &radii);

    let center = Array1::<f32>::from_shape_fn(params.n_features, |_| {
        rng.random_range(feature_min_bound..feature_max_bound)
    });

    for (cluster_idx, &(r_min, r_max)) in radii.iter().enumerate() {
        let points = random_points(
            &mut rng,
            samples_per_cluster,
            &center.to_vec(),
            r_min,
            r_max,
        );
        let start = cluster_idx * samples_per_cluster;
        let end = start + samples_per_cluster;
        features.slice_mut(s![start..end, ..]).assign(&points);
    }
}
