//! The ranked [`Classification`] decision read from a trained classifier's final outputs.

use crate::model::Activations;
use ndarray::{ArrayView1, Ix2};
use std::cmp::Ordering::Equal;

/// Class probabilities for one instance, ordered by descending probability.
///
/// Each entry pairs a class id with its probability. A binary network emits a
/// single output, which is expanded into the two complementary class
/// probabilities; a multi-class network's outputs are taken as-is.
#[derive(Clone, Debug, PartialEq)]
pub struct Classification(Vec<(usize, f32)>);

impl Classification {
    /// Builds a ranking from a network's class probabilities for one instance.
    ///
    /// A single output is the binary positive-class probability, expanded into
    /// `[(0, 1 - p), (1, p)]`; otherwise each output is the probability of its
    /// class. Entries are sorted by descending probability.
    pub fn from_probabilities(probabilities: ArrayView1<f32>) -> Self {
        let mut ranking: Vec<(usize, f32)> = if probabilities.len() == 1 {
            vec![(0, 1.0 - probabilities[0]), (1, probabilities[0])]
        } else {
            probabilities.iter().copied().enumerate().collect()
        };

        ranking.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(Equal));
        Self(ranking)
    }

    /// Builds the ranking from a single instance's finalized [`Activations`]: the output stage,
    /// read as one flat class vector.
    pub fn from_activations(activations: &Activations) -> Self {
        let outputs = activations
            .output()
            .into_dimensionality::<Ix2>()
            .expect("Classification ranks a flat class vector, not a spatial output");
        Self::from_probabilities(outputs.column(0))
    }

    /// The most likely class and its probability.
    pub fn top(&self) -> (usize, f32) {
        self.ranking()[0]
    }

    /// The ranked `(class, probability)` entries, most likely first.
    pub fn ranking(&self) -> &[(usize, f32)] {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::{IDENTITY, RELU};
    use crate::data::scalers::{Scaler, ScalerKind};
    use crate::model::{NetworkConfig, NeuralNetwork, PredictionError, Predictor};
    use crate::task::Task;
    use ndarray::{Array, Array4, ArrayD, IxDyn, array, s};

    #[test]
    fn from_activations_ranks_a_binary_networks_outputs() {
        let config = NetworkConfig::builder(vec![2])
            .dense(4, &RELU)
            .dense(1, &IDENTITY)
            .build();
        let predictor = Predictor::new(
            NeuralNetwork::from_config(config, 0).unwrap(),
            Task::Binary,
            None,
        );

        let activations = predictor.infer_instance(array![0.3, 0.7].view()).unwrap();
        let classification = Classification::from_activations(&activations);

        // A binary network yields the two complementary class probabilities,
        // ranked by descending probability and summing to one.
        let ranking = classification.ranking();
        assert_eq!(ranking.len(), 2);
        assert!(ranking[0].1 >= ranking[1].1);
        let sum: f32 = ranking.iter().map(|(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn binary_output_expands_to_two_complementary_classes() {
        let c = Classification::from_probabilities(array![0.7].view());

        assert_eq!(c.ranking().len(), 2);
        // Sorted desc: positive class (0.7) first, then its complement (0.3).
        assert_eq!(c.top().0, 1);
        assert!((c.top().1 - 0.7).abs() < 1e-6);
        assert_eq!(c.ranking()[1].0, 0);
        assert!((c.ranking()[1].1 - 0.3).abs() < 1e-6);
        let sum: f32 = c.ranking().iter().map(|(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn binary_output_below_half_ranks_negative_class_first() {
        let c = Classification::from_probabilities(array![0.2].view());

        assert_eq!(c.top().0, 0);
        assert!((c.top().1 - 0.8).abs() < 1e-6);
    }

    #[test]
    fn multiclass_outputs_keep_class_indices_and_sort_desc() {
        let c = Classification::from_probabilities(array![0.1, 0.6, 0.3].view());

        assert_eq!(
            c.ranking().iter().map(|(i, _)| *i).collect::<Vec<_>>(),
            vec![1, 2, 0]
        );
        assert_eq!(c.top().0, 1);
    }

    #[test]
    fn convolutional_network_predicts_end_to_end_on_a_spatial_batch() {
        // Conv2d (1×4×4 → 2×2×2) → Flatten (8) → Dense (1 identity logit, binary). The rank-4
        // spatial batch threads through the network unchanged; the Flatten collapses it to the
        // rank-2 the Dense head consumes, and infer yields (classes, samples) outputs.
        let config = NetworkConfig::builder(vec![1, 4, 4])
            .conv2d(2, (3, 3), 1, 0, &RELU)
            .flatten()
            .dense(1, &IDENTITY)
            .build();
        let model = NeuralNetwork::from_config(config, 0).unwrap();

        // A spatial input the dense path could never accept: (channels, height, width, samples).
        let inputs = Array::from_shape_fn(IxDyn(&[1, 4, 4, 3]), |idx| {
            ((idx[1] + idx[2] + idx[3]) as f32).sin()
        });
        let output = Predictor::new(model, Task::Binary, None)
            .output(inputs.view())
            .unwrap();
        assert_eq!(output.shape(), &[1, 3]); // (classes, samples)
        assert!(output.iter().all(|&v| (0.0..=1.0).contains(&v)));
    }

    #[test]
    fn predictor_applies_a_per_feature_scaler_to_a_spatial_batch() {
        // Flatten (2×2×2 → 8) → Dense (1 identity logit, binary): a network that accepts a
        // rank-4 spatial batch. Two identical builds (same seed) let one go into the
        // predictor and the other check the bare network on manually scaled inputs.
        let network = |seed| {
            let config = NetworkConfig::builder(vec![2, 2, 2])
                .flatten()
                .dense(1, &IDENTITY)
                .build();
            NeuralNetwork::from_config(config, seed).unwrap()
        };

        // (features=2, height=2, width=2, samples=4), the two features on different scales.
        let inputs = Array4::from_shape_fn((2, 2, 2, 4), |(f, h, w, s)| {
            (f as f32 + 1.0) * 10.0 * (h + w + s) as f32
        });
        let scaler = ScalerKind::MinMax.fit(inputs.view());
        let predictor = Predictor::new(network(1), Task::Binary, Some(scaler.clone()));

        // The predictor scales the rank-4 batch per feature before the network: its
        // output matches the bare network run on the manually scaled inputs.
        let via_predictor = predictor.output(inputs.view()).unwrap();
        let mut scaled = inputs.clone().into_dyn();
        scaler.apply_inplace(scaled.view_mut()).unwrap();
        let via_network = Predictor::new(network(1), Task::Binary, None)
            .output(scaled.view())
            .unwrap();
        assert_eq!(via_predictor, via_network);

        // A single rank-3 instance is scaled with the same parameters, so its classification
        // matches the first column of the batch probabilities.
        let instance = inputs.slice(s![.., .., .., 0]).to_owned();
        let activations = predictor.infer_instance(instance.view()).unwrap();
        let single = Classification::from_activations(&activations);
        let batch = via_predictor.into_dimensionality::<Ix2>().unwrap();
        let expected = Classification::from_probabilities(batch.column(0));
        assert_eq!(single, expected);
    }

    #[test]
    fn output_surfaces_a_scaler_feature_mismatch_as_a_scaling_error() {
        // A scaler fitted on 2 leading-axis features, then applied to a batch with 3:
        // the scaler's mismatch surfaces through `?` as PredictionError::Scaling.
        let config = NetworkConfig::builder(vec![2, 2, 2])
            .flatten()
            .dense(1, &IDENTITY)
            .build();
        let network = NeuralNetwork::from_config(config, 0).unwrap();
        let fit_batch = Array::from_shape_fn(IxDyn(&[2, 4]), |d| (d[0] + d[1]) as f32);
        let scaler = ScalerKind::MinMax.fit(fit_batch.view());
        let predictor = Predictor::new(network, Task::Binary, Some(scaler));

        let wrong = ArrayD::<f32>::ones(IxDyn(&[3, 2, 2, 5]));
        let error = predictor.output(wrong.view()).unwrap_err();

        assert!(
            matches!(error, PredictionError::Scaling(_)),
            "unexpected error: {error}"
        );
        // The Scaling arm delegates its Display to the underlying scaler error.
        assert!(error.to_string().contains("features"));
    }

    #[test]
    fn probabilities_always_in_zero_one() {
        let config = NetworkConfig::builder(vec![4]).dense(1, &IDENTITY).build();
        let model = NeuralNetwork::from_config(config, 0).unwrap();
        // f32 saturates to exactly 0.0 or 1.0 for large logits, so use closed interval
        let inputs = array![
            [100.0, -100.0],
            [100.0, -100.0],
            [100.0, -100.0],
            [100.0, -100.0]
        ];
        let output = Predictor::new(model, Task::Binary, None)
            .output(inputs.view())
            .unwrap();
        for &v in output.iter() {
            assert!(
                (0.0..=1.0).contains(&v),
                "Sigmoid output {} not in [0, 1]",
                v
            );
        }
    }
}
