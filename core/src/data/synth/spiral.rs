use crate::data::synth::{SynthError, SynthParams};
use ndarray::Array2;
use ndarray_rand::rand::{Rng, RngCore};
use ndarray_rand::rand_distr::StandardNormal;
use std::f32::consts::TAU;

/// Checks that the spiral is two-dimensional, the only constraint [`fill`] adds
/// (it indexes features 0 and 1 directly).
///
/// # Errors
/// - [`SynthError::SpiralRequiresTwoFeatures`] when `n_features != 2`.
pub(super) fn validate(params: &SynthParams) -> Result<(), SynthError> {
    if params.n_features != 2 {
        return Err(SynthError::SpiralRequiresTwoFeatures(params.n_features));
    }
    Ok(())
}

/// Fills `features` with interleaved spiral arms, one arm (class) per cluster.
///
/// Every arm is the same Archimedean spiral rotated by an even phase offset, so
/// the classes interleave and cannot be separated by a linear boundary — the
/// canonical benchmark for why hidden layers are needed. Spiral datasets are
/// inherently two-dimensional. `turns` is how many turns each arm makes and
/// `noise` is the Gaussian jitter as a fraction of the arm's max radius.
pub(super) fn fill(
    params: &SynthParams,
    turns: f32,
    noise: f32,
    features: &mut Array2<f32>,
    rng: &mut dyn RngCore,
) {
    let samples_per_cluster = params.samples_per_cluster();
    let center = (params.feature_min + params.feature_max) / 2.0;
    // Leave a small margin so jittered points stay inside the bounds.
    let max_radius = (params.feature_max - params.feature_min) / 2.0 * 0.9;
    let jitter_scale = max_radius * noise;

    for arm in 0..params.n_clusters {
        let phase = arm as f32 / params.n_clusters as f32 * TAU;
        for i in 0..samples_per_cluster {
            // `t` in (0, 1]: distance along the arm, from center to rim.
            let t = (i + 1) as f32 / samples_per_cluster as f32;
            let angle = phase + t * turns * TAU;
            let radius = t * max_radius;

            let jitter_x: f32 = rng.sample(StandardNormal);
            let jitter_y: f32 = rng.sample(StandardNormal);
            let x = center + radius * angle.cos() + jitter_x * jitter_scale;
            let y = center + radius * angle.sin() + jitter_y * jitter_scale;

            let row = arm * samples_per_cluster + i;
            features[[row, 0]] = x.clamp(params.feature_min, params.feature_max);
            features[[row, 1]] = y.clamp(params.feature_min, params.feature_max);
        }
    }
}
