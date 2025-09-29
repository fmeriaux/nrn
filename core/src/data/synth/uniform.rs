use crate::data::dataset::Dataset;
use crate::data::synth::{
    DatasetGenerator, calculate_radius, feature_bounds, init_features_and_labels, random_points,
};
use ndarray::{Array2, s};
use ndarray_rand::rand::Rng;
use ndarray_rand::rand::RngCore;

/// Generates a synthetic dataset with clusters arranged in concentric spheres.
pub struct UniformDataset {
    /// The total number of samples to generate.
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

impl DatasetGenerator for UniformDataset {
    fn generate_rng(&self, mut rng: &mut dyn RngCore) -> Dataset {
        assert!(self.n_samples > 0,);

        let samples_per_cluster = self.n_samples / self.n_clusters;

        let radius: f32 = calculate_radius(self.feature_min, self.feature_max, self.n_clusters);
        let radii = &vec![(0.0, radius); self.n_clusters];

        let (mut features, labels) =
            init_features_and_labels(self.n_features, self.n_clusters, samples_per_cluster);

        let (feature_min_bound, feature_max_bound) =
            feature_bounds(self.feature_min, self.feature_max, radii);

        assert!(
            feature_max_bound > feature_min_bound,
            "feature_max_bound must be greater than feature_min_bound"
        );

        let centers = Array2::<f32>::from_shape_fn((self.n_clusters, self.n_features), |_| {
            rng.gen_range(feature_min_bound..feature_max_bound)
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

        Dataset::shuffled(&mut rng, &features, &labels)
    }
}
