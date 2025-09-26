use crate::data::Dataset;
use crate::synth::{
    DatasetGenerator, calculate_radius, feature_bounds, init_features_and_labels, random_points,
};
use ndarray::{Array1, s};
use ndarray_rand::rand::{Rng, RngCore};

/// Generates a dataset with clusters arranged in concentric rings.
pub struct RingDataset {
    /// The number of samples to generate for each cluster.
    pub n_samples: usize,
    /// The number of features for each sample.
    pub n_features: usize,
    /// The number of clusters to generate.
    pub n_clusters: usize,
    /// The minimum value for each feature.
    pub feature_min: f32,
    /// The maximum value for each feature.
    pub feature_max: f32,
}

impl DatasetGenerator for RingDataset {
    fn generate_rng(&self, mut rng: &mut dyn RngCore) -> Dataset {
        assert!(
            self.feature_min < self.feature_max,
            "feature_min must be less than feature_max"
        );

        let radius: f32 = calculate_radius(self.feature_min, self.feature_max, self.n_clusters);
        let samples_per_cluster = self.n_samples / self.n_clusters;

        // Generate increasing radii for each cluster with a configurable overlap_factor.
        // The overlap_factor controls the percentage of overlap between clusters.
        let overlap_factor = -0.2;
        let radii: Vec<(f32, f32)> = (0..self.n_clusters)
            .map(|i| {
                let idx = i as f32;
                let min = radius * idx * (1.0 - overlap_factor);
                let max = min + radius;
                (min, max)
            })
            .collect();

        let (mut features, labels) =
            init_features_and_labels(self.n_features, radii.len(), samples_per_cluster);

        let (feature_min_bound, feature_max_bound) =
            feature_bounds(self.feature_min, self.feature_max, &radii);

        let center = Array1::<f32>::from_shape_fn(self.n_features, |_| {
            rng.gen_range(feature_min_bound..feature_max_bound)
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

        Dataset::shuffled(&mut rng, &features, &labels)
    }
}
