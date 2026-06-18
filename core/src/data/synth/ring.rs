use crate::data::synth::{
    SynthError, SynthParams, calculate_radius, feature_bounds, max_radius, random_points,
};
use ndarray::{Array2, s};
use ndarray_rand::rand::{Rng, RngCore};

/// Increasing `(min, max)` radius range for each cluster. `overlap` controls how
/// much consecutive rings overlap (negative values leave a gap). Shared by
/// [`validate`] and [`fill`] so both reason about the same geometry.
fn radii(params: &SynthParams, overlap: f32) -> Vec<(f32, f32)> {
    let radius = calculate_radius(params.feature_min, params.feature_max, params.n_clusters);
    (0..params.n_clusters)
        .map(|i| {
            let min = radius * i as f32 * (1.0 - overlap);
            (min, min + radius)
        })
        .collect()
}

/// Checks that the rings fit within the feature range, so [`fill`]'s geometry
/// invariants hold and generation cannot panic.
///
/// # Errors
/// - [`SynthError::RingOverlapTooLarge`] when `overlap >= 1.0` (inner radii would
///   turn negative).
/// - [`SynthError::RingTooDenseForRange`] when the outermost ring leaves no room
///   inside the feature range for a valid center.
pub(super) fn validate(params: &SynthParams, overlap: f32) -> Result<(), SynthError> {
    if overlap >= 1.0 {
        return Err(SynthError::RingOverlapTooLarge(overlap));
    }
    // `overlap < 1.0` keeps every inner radius non-negative, so `max_radius` holds.
    let max_radius = max_radius(&radii(params, overlap));
    let span = params.feature_max - params.feature_min;
    if span <= 2.0 * max_radius {
        return Err(SynthError::RingTooDenseForRange(params.n_clusters));
    }
    Ok(())
}

/// Fills `features` with concentric rings sharing a common center, one ring
/// (class) per cluster. The geometry was checked by [`validate`] at
/// construction, so the bounds below always leave a valid center.
pub(super) fn fill(
    params: &SynthParams,
    overlap: f32,
    features: &mut Array2<f32>,
    mut rng: &mut dyn RngCore,
) {
    let samples_per_cluster = params.samples_per_cluster();
    let radii = radii(params, overlap);

    let (feature_min_bound, feature_max_bound) =
        feature_bounds(params.feature_min, params.feature_max, &radii);

    let center: Vec<f32> = (0..params.n_features)
        .map(|_| rng.random_range(feature_min_bound..feature_max_bound))
        .collect();

    for (cluster_idx, &(r_min, r_max)) in radii.iter().enumerate() {
        let points = random_points(&mut rng, samples_per_cluster, &center, r_min, r_max);
        let start = cluster_idx * samples_per_cluster;
        let end = start + samples_per_cluster;
        features.slice_mut(s![start..end, ..]).assign(&points);
    }
}
