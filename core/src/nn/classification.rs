//! The outcome of running a trained classifier on a single instance: the class
//! probabilities, ranked most likely first.

use crate::model::Predictor;
use ndarray::ArrayView1;
use std::cmp::Ordering::Equal;

/// Class probabilities for one instance, ordered by descending probability.
///
/// Each entry pairs a class id with its probability. A binary network emits a
/// single output, which is expanded into the two complementary class
/// probabilities; a multi-class network's outputs are taken as-is.
#[derive(Clone, Debug, PartialEq)]
pub struct Classification(Vec<(usize, f32)>);

impl Classification {
    /// Builds a ranking from a network's raw output activations for one instance.
    ///
    /// A single output is the binary positive-class probability, expanded into
    /// `[(0, 1 - p), (1, p)]`; otherwise each output is the probability of its
    /// class. Entries are sorted by descending probability.
    pub fn from_outputs(outputs: ArrayView1<f32>) -> Self {
        let mut ranking: Vec<(usize, f32)> = if outputs.len() == 1 {
            vec![(0, 1.0 - outputs[0]), (1, outputs[0])]
        } else {
            outputs.iter().copied().enumerate().collect()
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
    /// Classifies a single raw input vector, returning the class probabilities
    /// ranked most likely first. The scaler is applied first when present.
    pub fn classify_single(&self, input: ArrayView1<f32>) -> Classification {
        Classification::from_outputs(self.predict_single(input).view())
    }
}

#[cfg(test)]
mod tests {
    use super::Classification;
    use crate::activations::RELU;
    use crate::model::{LayerPlan, NeuralNetwork, NeuronLayerSpec, Predictor};
    use ndarray::array;

    #[test]
    fn classify_single_ranks_the_networks_outputs() {
        let specs = NeuronLayerSpec::plan(LayerPlan::Explicit(vec![4]), 2, &*RELU).unwrap();
        let predictor = Predictor::new(NeuralNetwork::initialization(2, &specs, 0), None);

        let classification = predictor.classify_single(array![0.3, 0.7].view());

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
        let c = Classification::from_outputs(array![0.7].view());

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
        let c = Classification::from_outputs(array![0.2].view());

        assert_eq!(c.top().0, 0);
        assert!((c.top().1 - 0.8).abs() < 1e-6);
    }

    #[test]
    fn multiclass_outputs_keep_class_indices_and_sort_desc() {
        let c = Classification::from_outputs(array![0.1, 0.6, 0.3].view());

        assert_eq!(
            c.ranking().iter().map(|(i, _)| *i).collect::<Vec<_>>(),
            vec![1, 2, 0]
        );
        assert_eq!(c.top().0, 1);
    }
}
