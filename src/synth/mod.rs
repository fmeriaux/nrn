use ndarray::{Array1, Array2, Axis, s};
use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::rand::seq::SliceRandom;
use ndarray_rand::rand::{Rng, SeedableRng};
use nrn::data::Dataset;
use std::error::Error;
use std::fmt;

mod generators;

/// Represents the type of distribution used for generating synthetic datasets.
#[derive(Debug, Clone)]
pub enum DistributionType {
    Uniform,
    Ring,
}

impl fmt::Display for DistributionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DistributionType::Uniform => write!(f, "uniform"),
            DistributionType::Ring => write!(f, "ring"),
        }
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

pub trait DatasetExt {
    fn shuffled<R: Rng>(rng: &mut R, features: &Array2<f32>, labels: &Array1<f32>) -> Self;
    fn from_image_vec<R: Rng>(
        rng: &mut R,
        images: Vec<Array1<u8>>,
        labels: Vec<usize>,
    ) -> Result<Dataset, Box<dyn Error>>;
    fn new(
        distribution: &DistributionType,
        seed: u64,
        n_samples: usize,
        n_features: usize,
        n_clusters: usize,
        feature_min: f32,
        feature_max: f32,
    ) -> Self;
    fn uniform(
        seed: u64,
        n_samples: usize,
        n_features: usize,
        n_clusters: usize,
        feature_min: f32,
        feature_max: f32,
    ) -> Self;
    fn ring(
        seed: u64,
        n_samples: usize,
        n_features: usize,
        n_clusters: usize,
        feature_min: f32,
        feature_max: f32,
    ) -> Self;
}

impl DatasetExt for Dataset {
    /// Shuffles the dataset features and labels in unison using a random number generator.
    fn shuffled<R: Rng>(rng: &mut R, features: &Array2<f32>, labels: &Array1<f32>) -> Self {
        let mut indices: Vec<usize> = (0..features.nrows()).collect();
        indices.shuffle(rng);

        let shuffled_features = features.select(Axis(0), &indices);
        let shuffled_labels =
            Array1::from(indices.iter().map(|&i| labels[i]).collect::<Vec<f32>>());

        Dataset {
            features: shuffled_features.to_owned(),
            labels: shuffled_labels,
        }
    }

    /// Creates a new dataset from a vector of images and their corresponding labels.
    /// # Arguments
    /// - `rng`: A mutable reference to a random number generator for shuffling.
    /// - `images`: A vector of images represented as 1D arrays of pixel values.
    /// - `labels`: A vector of labels corresponding to each image.
    fn from_image_vec<R: Rng>(
        rng: &mut R,
        images: Vec<Array1<u8>>,
        labels: Vec<usize>,
    ) -> Result<Self, Box<dyn Error>> {
        assert!(
            !images.is_empty(),
            "Images vector must not be empty to create a dataset"
        );

        let features: Array2<f32> = Array2::from_shape_vec(
            (images.len(), images[0].len()),
            images.into_iter().flatten().map(|x| x as f32).collect(),
        )?;

        let labels: Array1<f32> =
            Array1::from_shape_vec(labels.len(), labels.into_iter().map(|x| x as f32).collect())?;

        Ok(Dataset::shuffled(rng, &features, &labels))
    }

    /// Creates a new synthetic dataset based on the specified distribution type.
    /// # Arguments
    /// - `distribution`: The type of distribution to use for generating the dataset.
    /// - `seed`: A seed for the random number generator to ensure reproducibility.
    /// - `n_samples`: The total number of samples to generate.
    /// - `n_features`: The number of features for each sample.
    /// - `n_clusters`: The number of clusters to generate.
    /// - `feature_min`: The minimum value for each feature.
    /// - `feature_max`: The maximum value for each feature.
    fn new(
        distribution: &DistributionType,
        seed: u64,
        n_samples: usize,
        n_features: usize,
        n_clusters: usize,
        feature_min: f32,
        feature_max: f32,
    ) -> Self {
        match distribution {
            DistributionType::Uniform => Self::uniform(
                seed,
                n_samples,
                n_features,
                n_clusters,
                feature_min,
                feature_max,
            ),
            DistributionType::Ring => Self::ring(
                seed,
                n_samples,
                n_features,
                n_clusters,
                feature_min,
                feature_max,
            ),
        }
    }

    /// Generates a synthetic dataset with clusters arranged in concentric spheres.
    /// # Arguments
    /// - `seed`: A seed for the random number generator to ensure reproducibility.
    /// - `n_samples`: The total number of samples to generate.
    /// - `n_features`: The number of features for each sample.
    /// - `n_clusters`: The number of clusters to generate.
    /// - `feature_min`: The minimum value for each feature.
    /// - `feature_max`: The maximum value for each feature.
    fn uniform(
        seed: u64,
        n_samples: usize,
        n_features: usize,
        n_clusters: usize,
        feature_min: f32,
        feature_max: f32,
    ) -> Self {
        assert!(n_samples > 0,);

        let mut rng = StdRng::seed_from_u64(seed);

        let samples_per_cluster = n_samples / n_clusters;

        let radius: f32 = calculate_radius(feature_min, feature_max, n_clusters);
        let radii = &vec![(0.0, radius); n_clusters];

        let (mut features, labels) =
            init_features_and_labels(n_features, n_clusters, samples_per_cluster);

        let (feature_min_bound, feature_max_bound) =
            feature_bounds(feature_min, feature_max, radii);

        assert!(
            feature_max_bound > feature_min_bound,
            "feature_max_bound must be greater than feature_min_bound"
        );

        let centers = Array2::<f32>::from_shape_fn((n_clusters, n_features), |_| {
            rng.gen_range(feature_min_bound..feature_max_bound)
        });

        for (cluster_idx, (center, &(radius_min, radius_max))) in
            centers.outer_iter().zip(radii.iter()).enumerate()
        {
            let points = generators::random_points(
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

        Dataset::shuffled(&mut rng, &features, &labels)
    }

    /// Generates a dataset with clusters arranged in concentric rings.
    /// # Arguments
    /// - `seed`: A seed for the random number generator to ensure reproducibility.
    /// - `samples`: The number of samples to generate for each cluster.
    /// - `n_features`: The number of features for each sample.
    /// - `n_clusters`: The number of clusters to generate.
    /// - `feature_min`: The minimum value for each feature.
    /// - `feature_max`: The maximum value for each feature.
    fn ring(
        seed: u64,
        n_samples: usize,
        n_features: usize,
        n_clusters: usize,
        feature_min: f32,
        feature_max: f32,
    ) -> Self {
        assert!(
            feature_min < feature_max,
            "feature_min must be less than feature_max"
        );

        let radius: f32 = calculate_radius(feature_min, feature_max, n_clusters);
        let samples_per_cluster = n_samples / n_clusters;

        // Generate increasing radii for each cluster with a configurable overlap_factor.
        // The overlap_factor controls the percentage of overlap between clusters.
        let overlap_factor = -0.2;
        let radii: Vec<(f32, f32)> = (0..n_clusters)
            .map(|i| {
                let idx = i as f32;
                let min = radius * idx * (1.0 - overlap_factor);
                let max = min + radius;
                (min, max)
            })
            .collect();

        let (mut features, labels) =
            init_features_and_labels(n_features, radii.len(), samples_per_cluster);

        let mut rng = StdRng::seed_from_u64(seed);

        let (feature_min_bound, feature_max_bound) =
            feature_bounds(feature_min, feature_max, &radii);

        let center = Array1::<f32>::from_shape_fn(n_features, |_| {
            rng.gen_range(feature_min_bound..feature_max_bound)
        });

        for (cluster_idx, &(r_min, r_max)) in radii.iter().enumerate() {
            let points = generators::random_points(
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

        Dataset::shuffled(&mut rng, &features, &labels)
    }
}
