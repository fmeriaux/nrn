mod ring;
mod spiral;
mod uniform;

use crate::data::DatasetOrigin;
use crate::data::dataset::Dataset;
use ndarray::{Array1, Array2, ArrayViewMut1, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand::prelude::StdRng;
use ndarray_rand::rand::{Rng, SeedableRng};
use ndarray_rand::rand_distr::{StandardNormal, Uniform};
use std::error::Error;
use std::fmt;

/// Errors returned when synthetic-generation parameters are inconsistent.
#[derive(Debug, PartialEq)]
pub enum SynthParamsError {
    /// No features were requested.
    NoFeatures,
    /// Fewer than two clusters: a classifier needs at least two classes.
    TooFewClusters(usize),
    /// Not enough samples to place at least one in each cluster.
    NotEnoughSamples { samples: usize, clusters: usize },
    /// The feature range is empty (`min >= max`).
    EmptyRange { min: f32, max: f32 },
}

impl fmt::Display for SynthParamsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SynthParamsError::NoFeatures => write!(f, "at least one feature is required"),
            SynthParamsError::TooFewClusters(n) => {
                write!(f, "at least 2 clusters are required, but found {n}")
            }
            SynthParamsError::NotEnoughSamples { samples, clusters } => write!(
                f,
                "need at least one sample per cluster: {samples} samples for {clusters} clusters"
            ),
            SynthParamsError::EmptyRange { min, max } => {
                write!(
                    f,
                    "feature range is empty: min {min} must be less than max {max}"
                )
            }
        }
    }
}

impl Error for SynthParamsError {}

/// Shared, validated configuration common to every synthetic generator: how many
/// samples and features to produce, how many clusters (classes) to split them
/// into, and the per-feature value range the points live in.
///
/// Built exclusively through [`SynthParams::new`] (fields are private) so an
/// inconsistent configuration cannot be represented. Requiring `n_clusters >= 2`
/// guarantees the generated dataset always has enough classes to load and train.
#[derive(Debug, Clone, PartialEq)]
pub struct SynthParams {
    n_samples: usize,
    n_features: usize,
    n_clusters: usize,
    feature_min: f32,
    feature_max: f32,
}

impl SynthParams {
    /// Builds a validated parameter set.
    ///
    /// # Errors
    /// - [`SynthParamsError::NoFeatures`] when `n_features == 0`.
    /// - [`SynthParamsError::TooFewClusters`] when `n_clusters < 2`.
    /// - [`SynthParamsError::NotEnoughSamples`] when `n_samples < n_clusters`.
    /// - [`SynthParamsError::EmptyRange`] when `feature_min >= feature_max`.
    pub fn new(
        n_samples: usize,
        n_features: usize,
        n_clusters: usize,
        feature_min: f32,
        feature_max: f32,
    ) -> Result<Self, SynthParamsError> {
        if n_features == 0 {
            return Err(SynthParamsError::NoFeatures);
        }
        if n_clusters < 2 {
            return Err(SynthParamsError::TooFewClusters(n_clusters));
        }
        if n_samples < n_clusters {
            return Err(SynthParamsError::NotEnoughSamples {
                samples: n_samples,
                clusters: n_clusters,
            });
        }
        if feature_min >= feature_max {
            return Err(SynthParamsError::EmptyRange {
                min: feature_min,
                max: feature_max,
            });
        }

        Ok(Self {
            n_samples,
            n_features,
            n_clusters,
            feature_min,
            feature_max,
        })
    }

    /// Number of samples placed in each cluster (the total is truncated to a
    /// multiple of `n_clusters`).
    fn samples_per_cluster(&self) -> usize {
        self.n_samples / self.n_clusters
    }
}

/// The point-placement strategy that distinguishes one synthetic dataset from
/// another. Doubles as the dataset's provenance "type". Each variant carries its
/// own shape knobs, while sizing stays in the shared [`SynthParams`]. Default
/// values for these knobs are a caller policy and live with the caller.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Distribution {
    /// Isotropic blobs: one randomly-placed spherical cluster per class.
    Uniform,
    /// Concentric rings sharing a common center, one ring per class. `overlap`
    /// is the fraction by which consecutive rings overlap (negative = a gap).
    Ring { overlap: f32 },
    /// Interleaved spiral arms (2D only), one arm per class — the canonical
    /// non-linearly-separable benchmark. `turns` is how many turns each arm makes
    /// and `noise` the Gaussian jitter as a fraction of the arm's max radius.
    Spiral { turns: f32, noise: f32 },
}

// `Display` is never derived in Rust (only `Debug` is), and `Debug` would yield
// the capitalized variant name with its fields. We want the lowercase canonical
// name used for filenames and provenance metadata, so it is written by hand.
impl fmt::Display for Distribution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Distribution::Uniform => "uniform",
            Distribution::Ring { .. } => "ring",
            Distribution::Spiral { .. } => "spiral",
        };
        write!(f, "{name}")
    }
}

/// Errors returned when a [`Distribution`] is incompatible with its
/// [`SynthParams`].
#[derive(Debug, PartialEq, Eq)]
pub enum SynthError {
    /// A spiral was requested with a feature count other than two (carries the
    /// count found); spirals are inherently two-dimensional.
    SpiralRequiresTwoFeatures(usize),
}

impl fmt::Display for SynthError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SynthError::SpiralRequiresTwoFeatures(n) => {
                write!(
                    f,
                    "spiral datasets require exactly two features, but found {n}"
                )
            }
        }
    }
}

impl Error for SynthError {}

/// A validated synthetic dataset generator: shared [`SynthParams`] plus the
/// chosen [`Distribution`]. The distribution selects how points are placed per
/// cluster; label assignment and shuffling are shared across all distributions.
///
/// Built through [`SynthDataset::new`] (fields are private) so a distribution
/// incompatible with the parameters cannot be represented — generation therefore
/// never fails.
pub struct SynthDataset {
    params: SynthParams,
    distribution: Distribution,
}

impl SynthDataset {
    /// Pairs validated `params` with a `distribution`, checking their cross-field
    /// invariants.
    ///
    /// # Errors
    /// - [`SynthError::SpiralRequiresTwoFeatures`] when a spiral is requested with
    ///   a feature count other than two.
    pub fn new(params: SynthParams, distribution: Distribution) -> Result<Self, SynthError> {
        if matches!(distribution, Distribution::Spiral { .. }) && params.n_features != 2 {
            return Err(SynthError::SpiralRequiresTwoFeatures(params.n_features));
        }
        Ok(Self {
            params,
            distribution,
        })
    }

    /// Generates a dataset using a specific seed for reproducibility, stamped with
    /// its [`DatasetOrigin::Synthetic`] provenance. The parameters were validated
    /// at construction, so the result is always well-formed.
    /// # Arguments
    /// - `seed`: A seed for the random number generator to ensure reproducibility.
    pub fn generate(&self, seed: u64) -> Dataset {
        let mut rng = StdRng::seed_from_u64(seed);

        let (mut features, labels) = init_features_and_labels(
            self.params.n_features,
            self.params.n_clusters,
            self.params.samples_per_cluster(),
        );

        match self.distribution {
            Distribution::Uniform => uniform::fill(&self.params, &mut features, &mut rng),
            Distribution::Ring { overlap } => {
                ring::fill(&self.params, overlap, &mut features, &mut rng)
            }
            Distribution::Spiral { turns, noise } => {
                spiral::fill(&self.params, turns, noise, &mut features, &mut rng)
            }
        }

        let origin = DatasetOrigin::Synthetic {
            distribution: self.distribution.to_string(),
            seed,
        };

        Dataset::new(features, labels, Some(origin))
            .expect("synthetic parameters guarantee a valid dataset")
            .shuffled(&mut rng)
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

/// Scales a vector to unit length in place.
///
/// A zero vector is left untouched: dividing by its (zero) norm would produce
/// `NaN`s, so the degenerate case is guarded explicitly.
fn normalize_to_unit(mut row: ArrayViewMut1<f32>) {
    let norm = row.iter().map(|v| v.powi(2)).sum::<f32>().sqrt();
    if norm != 0.0 {
        row.iter_mut().for_each(|v| *v /= norm);
    }
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
        normalize_to_unit(row.view_mut());
        // For uniform distribution, we scale the points to the desired radius
        let u: f32 = rng.sample(Uniform::new(0.0_f32, 1.0).unwrap());
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    const RING: Distribution = Distribution::Ring { overlap: -0.2 };
    const SPIRAL: Distribution = Distribution::Spiral {
        turns: 1.5,
        noise: 0.03,
    };

    /// Test fixture: a valid two-feature parameter set over `[0, 10]`.
    fn params(n_samples: usize, n_clusters: usize) -> SynthParams {
        SynthParams::new(n_samples, 2, n_clusters, 0.0, 10.0).unwrap()
    }

    /// Generates a seeded dataset from already-validated params.
    fn generate(distribution: Distribution, params: SynthParams) -> Dataset {
        SynthDataset::new(params, distribution)
            .unwrap()
            .generate(42)
    }

    #[test]
    fn labels_are_zero_indexed() {
        for distribution in [Distribution::Uniform, RING, SPIRAL] {
            let mut labels = generate(distribution, params(90, 3)).unique_labels();
            labels.sort_by(|a, b| a.partial_cmp(b).unwrap());
            assert_eq!(labels, vec![0.0, 1.0, 2.0], "{distribution}");
        }
    }

    #[test]
    fn sample_count_exact_when_divisible() {
        for distribution in [Distribution::Uniform, RING, SPIRAL] {
            assert_eq!(generate(distribution, params(100, 5)).n_samples(), 100);
        }
    }

    #[test]
    fn sample_count_truncated_when_not_divisible() {
        for distribution in [Distribution::Uniform, RING, SPIRAL] {
            assert_eq!(generate(distribution, params(100, 3)).n_samples(), 99);
        }
    }

    #[test]
    fn features_within_bounds() {
        for distribution in [Distribution::Uniform, RING, SPIRAL] {
            for &val in generate(distribution, params(200, 2)).features().iter() {
                assert!(
                    (0.0..=10.0).contains(&val),
                    "{distribution}: feature {val} out of [0, 10]"
                );
            }
        }
    }

    #[test]
    fn spiral_rejects_a_feature_count_other_than_two() {
        let params = SynthParams::new(100, 3, 2, 0.0, 10.0).unwrap();
        let err = SynthDataset::new(params, SPIRAL).err().unwrap();
        assert_eq!(err, SynthError::SpiralRequiresTwoFeatures(3));
    }

    #[test]
    fn new_params_validates_invariants() {
        assert_eq!(
            SynthParams::new(100, 0, 2, 0.0, 10.0),
            Err(SynthParamsError::NoFeatures)
        );
        assert_eq!(
            SynthParams::new(100, 2, 1, 0.0, 10.0),
            Err(SynthParamsError::TooFewClusters(1))
        );
        assert_eq!(
            SynthParams::new(2, 2, 3, 0.0, 10.0),
            Err(SynthParamsError::NotEnoughSamples {
                samples: 2,
                clusters: 3
            })
        );
        assert_eq!(
            SynthParams::new(100, 2, 2, 10.0, 10.0),
            Err(SynthParamsError::EmptyRange {
                min: 10.0,
                max: 10.0
            })
        );
        assert!(SynthParams::new(100, 2, 2, 0.0, 10.0).is_ok());
    }

    #[test]
    fn synth_params_error_messages_are_human_readable() {
        assert_eq!(
            SynthParamsError::NoFeatures.to_string(),
            "at least one feature is required"
        );
        assert_eq!(
            SynthParamsError::TooFewClusters(1).to_string(),
            "at least 2 clusters are required, but found 1"
        );
        assert_eq!(
            SynthParamsError::NotEnoughSamples {
                samples: 2,
                clusters: 3
            }
            .to_string(),
            "need at least one sample per cluster: 2 samples for 3 clusters"
        );
        assert_eq!(
            SynthParamsError::EmptyRange {
                min: 10.0,
                max: 1.0
            }
            .to_string(),
            "feature range is empty: min 10 must be less than max 1"
        );
    }

    #[test]
    fn synth_error_message_is_human_readable() {
        assert_eq!(
            SynthError::SpiralRequiresTwoFeatures(3).to_string(),
            "spiral datasets require exactly two features, but found 3"
        );
    }

    #[test]
    fn distribution_display_is_the_lowercase_name() {
        assert_eq!(Distribution::Uniform.to_string(), "uniform");
        assert_eq!(RING.to_string(), "ring");
        assert_eq!(SPIRAL.to_string(), "spiral");
    }

    #[test]
    fn normalize_to_unit_scales_nonzero_vector_to_unit_length() {
        let mut row = array![3.0, 4.0]; // norm = 5
        normalize_to_unit(row.view_mut());
        let norm = row.iter().map(|v| v.powi(2)).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
        assert!((row[0] - 0.6).abs() < 1e-6);
        assert!((row[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn normalize_to_unit_leaves_zero_vector_untouched() {
        // A zero-norm row must stay at the origin, not become NaN.
        let mut row = array![0.0, 0.0, 0.0];
        normalize_to_unit(row.view_mut());
        assert_eq!(row, array![0.0, 0.0, 0.0]);
    }
}
