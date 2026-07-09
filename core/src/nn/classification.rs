//! The outcome of running a trained classifier on a single instance: the class
//! probabilities, ranked most likely first.

use crate::activations::probabilities_from_logits;
use crate::data::scalers::Scaler;
use crate::model::{PredictionError, Predictor};
use ndarray::{ArrayD, ArrayView, ArrayView1, Axis, Dimension, Ix2};
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

    /// The most likely class and its probability.
    pub fn top(&self) -> (usize, f32) {
        self.ranking()[0]
    }

    /// The ranked `(class, probability)` entries, most likely first.
    pub fn ranking(&self) -> &[(usize, f32)] {
        &self.0
    }
}

impl Predictor {
    /// Classifies a single raw instance of a flat classifier, returning its class
    /// probabilities ranked most likely first. The scaler is applied first when present.
    ///
    /// A [`Classification`] is a single ranking, so it applies to a flat class vector:
    /// a spatial per-position output has no single ranking.
    ///
    /// # Errors
    /// [`PredictionError`](crate::model::PredictionError) when the instance does not match
    /// the scaler's fitted feature count or the network's input shape.
    pub fn classify_instance<D: Dimension>(
        &self,
        input: ArrayView<f32, D>,
    ) -> Result<Classification, PredictionError> {
        let sample_axis = input.ndim();
        let inputs = input.insert_axis(Axis(sample_axis));
        let probabilities = self
            .class_probabilities(inputs)?
            .into_dimensionality::<Ix2>()
            .expect("Classification ranks a flat class vector, not a spatial output");
        Ok(Classification::from_probabilities(probabilities.column(0)))
    }

    /// Predicts class probabilities for a batch of raw inputs of any rank (samples on the
    /// trailing axis): applies the scaler first when present, then maps the network's logits
    /// through the inferred output activation: sigmoid for a single output (binary), softmax
    /// otherwise.
    ///
    /// # Errors
    /// [`PredictionError::Scaling`] when a scaler is present and the inputs do not match
    /// its fitted feature count, or [`PredictionError::Network`] when they do not match the
    /// network's input shape.
    pub fn class_probabilities<D: Dimension>(
        &self,
        inputs: ArrayView<f32, D>,
    ) -> Result<ArrayD<f32>, PredictionError> {
        let logits = match &self.scaler {
            Some(scaler) => {
                let mut owned = inputs.to_owned().into_dyn();
                scaler.apply_inplace(owned.view_mut())?;
                self.network.output(owned.view())?
            }
            None => self.network.output(inputs)?,
        };

        Ok(probabilities_from_logits(logits.view()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::{RELU, SIGMOID};
    use crate::data::scalers::ScalerKind;
    use crate::layers::{Conv2d, Dense, Flatten};
    use crate::model::{InputShapeMismatch, LayerPlan, NeuralNetwork, NeuronLayerSpec};
    use ndarray::{Array, Array4, IxDyn, array, s};
    use ndarray_rand::rand::SeedableRng;
    use ndarray_rand::rand::rngs::StdRng;

    #[test]
    fn classify_single_ranks_the_networks_outputs() {
        let specs = NeuronLayerSpec::plan(LayerPlan::Explicit(vec![4]), 2, &*RELU).unwrap();
        let predictor = Predictor::new(NeuralNetwork::initialization(2, &specs, 0), None);

        let classification = predictor
            .classify_instance(array![0.3, 0.7].view())
            .unwrap();

        // A binary network yields the two complementary class probabilities,
        // ranked by descending probability and summing to one.
        let ranking = classification.ranking();
        assert_eq!(ranking.len(), 2);
        assert!(ranking[0].1 >= ranking[1].1);
        let sum: f32 = ranking.iter().map(|(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn classify_single_rejects_an_instance_of_the_wrong_size() {
        let specs = NeuronLayerSpec::plan(LayerPlan::Explicit(vec![4]), 2, &*RELU).unwrap();
        let predictor = Predictor::new(NeuralNetwork::initialization(2, &specs, 0), None);

        // The network expects two features; a three-feature instance is rejected.
        // Without a scaler the mismatch surfaces from the network.
        let error = predictor
            .classify_instance(array![0.1, 0.2, 0.3].view())
            .unwrap_err();
        assert_eq!(
            error,
            PredictionError::Network(InputShapeMismatch {
                expected: vec![2],
                found: vec![3]
            })
        );
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
        // rank-2 the Dense head consumes, and class_probabilities yields (classes, samples).
        let conv = Conv2d::initialization(
            (1, 4, 4),
            2,
            (3, 3),
            1,
            0,
            RELU.clone(),
            &mut StdRng::seed_from_u64(0),
        );
        let head = Dense::initialization(
            8,
            &NeuronLayerSpec::output_for(2),
            &mut StdRng::seed_from_u64(1),
        );
        let model = NeuralNetwork::single(conv)
            .with_layer(Flatten::new(vec![2, 2, 2]))
            .with_layer(head);

        // A spatial input the dense path could never accept: (channels, height, width, samples).
        let inputs = Array::from_shape_fn(IxDyn(&[1, 4, 4, 3]), |idx| {
            ((idx[1] + idx[2] + idx[3]) as f32).sin()
        });
        let output = Predictor::new(model, None)
            .class_probabilities(inputs.view())
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
            NeuralNetwork::single(Flatten::new(vec![2, 2, 2])).with_layer(Dense::initialization(
                8,
                &NeuronLayerSpec::output_for(2),
                &mut StdRng::seed_from_u64(seed),
            ))
        };

        // (features=2, height=2, width=2, samples=4), the two features on different scales.
        let inputs = Array4::from_shape_fn((2, 2, 2, 4), |(f, h, w, s)| {
            (f as f32 + 1.0) * 10.0 * (h + w + s) as f32
        });
        let scaler = ScalerKind::MinMax.fit(inputs.view());
        let predictor = Predictor::new(network(1), Some(scaler.clone()));

        // The predictor scales the rank-4 batch per feature before the network: its
        // output matches the bare network run on the manually scaled inputs.
        let via_predictor = predictor.class_probabilities(inputs.view()).unwrap();
        let mut scaled = inputs.clone().into_dyn();
        scaler.apply_inplace(scaled.view_mut()).unwrap();
        let via_network = Predictor::new(network(1), None)
            .class_probabilities(scaled.view())
            .unwrap();
        assert_eq!(via_predictor, via_network);

        // A single rank-3 instance is scaled with the same parameters, so its classification
        // matches the first column of the batch probabilities.
        let instance = inputs.slice(s![.., .., .., 0]).to_owned();
        let single = predictor.classify_instance(instance.view()).unwrap();
        let batch = via_predictor.into_dimensionality::<Ix2>().unwrap();
        let expected = Classification::from_probabilities(batch.column(0));
        assert_eq!(single, expected);
    }

    #[test]
    fn class_probabilities_surfaces_a_scaler_feature_mismatch_as_a_scaling_error() {
        // A scaler fitted on 2 leading-axis features, then applied to a batch with 3:
        // the scaler's mismatch surfaces through `?` as PredictionError::Scaling.
        let network =
            NeuralNetwork::single(Flatten::new(vec![2, 2, 2])).with_layer(Dense::initialization(
                8,
                &NeuronLayerSpec::output_for(2),
                &mut StdRng::seed_from_u64(0),
            ));
        let fit_batch = Array::from_shape_fn(IxDyn(&[2, 4]), |d| (d[0] + d[1]) as f32);
        let scaler = ScalerKind::MinMax.fit(fit_batch.view());
        let predictor = Predictor::new(network, Some(scaler));

        let wrong = ArrayD::<f32>::ones(IxDyn(&[3, 2, 2, 5]));
        let error = predictor.class_probabilities(wrong.view()).unwrap_err();

        assert!(
            matches!(error, PredictionError::Scaling(_)),
            "unexpected error: {error}"
        );
        // The Scaling arm delegates its Display to the underlying scaler error.
        assert!(error.to_string().contains("features"));
    }

    #[test]
    fn probabilities_always_in_zero_one() {
        let specs = NeuronLayerSpec::network_for(vec![], &*SIGMOID, 2);
        let model = NeuralNetwork::initialization(4, &specs, 0);
        // f32 saturates to exactly 0.0 or 1.0 for large logits, so use closed interval
        let inputs = array![
            [100.0, -100.0],
            [100.0, -100.0],
            [100.0, -100.0],
            [100.0, -100.0]
        ];
        let output = Predictor::new(model, None)
            .class_probabilities(inputs.view())
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
