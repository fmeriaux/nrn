//! The [`Predictor`]: a trained network paired with the scaler fitted alongside it, plus the
//! error it raises when an instance's shape does not match.

use crate::activations::{SIGMOID, SOFTMAX};
use crate::data::scalers::{Scaler, ScalerFeatureMismatch, ScalerMethod};
use crate::model::{Activations, InputShapeMismatch, Labels, ModelConfig, NeuralNetwork};
use crate::task::Task;
use ndarray::{ArrayD, ArrayView, ArrayView1, Axis, Dimension, Ix2};
use std::cmp::Ordering::Equal;
use std::fmt;

/// A trained [`NeuralNetwork`] paired with the [`ModelConfig`] it was trained for and the
/// scaler fitted alongside it.
#[derive(Clone, Debug)]
pub struct Predictor {
    /// The trained network.
    pub network: NeuralNetwork,
    /// The task the network was trained for, and its labels when known.
    pub config: ModelConfig,
    /// The scaler applied to raw inputs before prediction, when one is present.
    pub scaler: Option<ScalerMethod>,
}

impl Predictor {
    /// Pairs a network and its config with an optional scaler.
    pub fn new(network: NeuralNetwork, config: ModelConfig, scaler: Option<ScalerMethod>) -> Self {
        Self {
            network,
            config,
            scaler,
        }
    }

    /// Runs raw `inputs` (any rank, samples on the trailing axis) through the model: scales them
    /// when a scaler is present, then finalizes the output stage for the task. A binary or
    /// multi-class task yields [`Inference::Classification`]; a multi-label or regression task
    /// yields [`Inference::Values`].
    ///
    /// # Errors
    /// [`PredictionError::Scaling`] when a scaler is present and `inputs` do not match its fitted
    /// feature count, or [`PredictionError::Network`] when they do not match the network's input shape.
    pub fn infer<D: Dimension>(
        &self,
        inputs: ArrayView<f32, D>,
    ) -> Result<Inference, PredictionError> {
        let mut activations = match &self.scaler {
            Some(scaler) => {
                let scaled = scaler.apply(inputs.into_dyn())?;
                self.network.forward(scaled.view())?
            }
            None => self.network.forward(inputs)?,
        };

        Ok(match self.config.task() {
            Task::Binary => {
                activations.finalize(&**SIGMOID);
                Inference::classification(activations, self.config.labels().cloned())
            }
            Task::MultiClass { .. } => {
                activations.finalize(&**SOFTMAX);
                Inference::classification(activations, self.config.labels().cloned())
            }
            Task::MultiLabel { .. } => {
                activations.finalize(&**SIGMOID);
                Inference::Values(activations)
            }
            Task::Regression { .. } => Inference::Values(activations),
        })
    }

    /// Runs a single raw `instance` (any rank, with no sample axis) through the model, inserting
    /// the trailing sample axis it lacks before delegating to [`infer`](Predictor::infer).
    ///
    /// # Errors
    /// As [`infer`](Predictor::infer).
    pub fn infer_instance<D: Dimension>(
        &self,
        instance: ArrayView<f32, D>,
    ) -> Result<Inference, PredictionError> {
        let sample_axis = instance.ndim();
        self.infer(instance.insert_axis(Axis(sample_axis)))
    }

    /// The model's final outputs for raw `inputs`: [`infer`](Predictor::infer) keeping only the
    /// finalized output stage — the probabilities or values, outputs on the leading axis and
    /// samples trailing.
    ///
    /// # Errors
    /// As [`infer`](Predictor::infer).
    pub fn output<D: Dimension>(
        &self,
        inputs: ArrayView<f32, D>,
    ) -> Result<ArrayD<f32>, PredictionError> {
        Ok(self.infer(inputs)?.into_output())
    }
}

/// A [`Predictor`]'s finalized output for one [`infer`](Predictor::infer) call, read according to
/// its task.
#[derive(Clone, Debug)]
pub enum Inference {
    /// A binary or multi-class task's output.
    Classification {
        /// The finalized activations the ranking was read from.
        activations: Activations,
        /// Class ids ranked by descending probability.
        ranking: Vec<(usize, f32)>,
        /// The label naming each class id, when known.
        labels: Option<Labels>,
    },
    /// A multi-label or regression task's output, with no ranking.
    Values(Activations),
}

impl Inference {
    /// Ranks `activations`' output by descending probability, naming classes with `labels` when
    /// known. A single sigmoid output is expanded into the two complementary class probabilities.
    fn classification(activations: Activations, labels: Option<Labels>) -> Self {
        let outputs = activations
            .output()
            .into_dimensionality::<Ix2>()
            .expect("a classification ranks a flat class vector, not a spatial output");
        let ranking = rank(outputs.column(0));

        Inference::Classification {
            activations,
            ranking,
            labels,
        }
    }

    /// The finalized activations behind this inference, whichever task produced it.
    pub fn activations(&self) -> &Activations {
        match self {
            Inference::Classification { activations, .. } | Inference::Values(activations) => {
                activations
            }
        }
    }

    /// Consumes the inference, keeping only its finalized output.
    pub fn into_output(self) -> ArrayD<f32> {
        match self {
            Inference::Classification { activations, .. } | Inference::Values(activations) => {
                activations.into_output()
            }
        }
    }
}

/// Ranks `probabilities` by descending value. A single probability is read as the positive
/// class's, expanded into `[(0, 1 - p), (1, p)]`; otherwise each entry is the probability of its
/// own class index.
fn rank(probabilities: ArrayView1<f32>) -> Vec<(usize, f32)> {
    let mut ranking: Vec<(usize, f32)> = if probabilities.len() == 1 {
        vec![(0, 1.0 - probabilities[0]), (1, probabilities[0])]
    } else {
        probabilities.iter().copied().enumerate().collect()
    };

    ranking.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(Equal));
    ranking
}

/// A [`Predictor`] rejected an instance: either its scaler or its network found the
/// wrong number of features.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PredictionError {
    /// The scaler's fitted feature count did not match the input.
    Scaling(ScalerFeatureMismatch),
    /// The network's input shape did not match the input.
    Network(InputShapeMismatch),
}

impl fmt::Display for PredictionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PredictionError::Scaling(e) => write!(f, "{e}"),
            PredictionError::Network(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for PredictionError {}

impl From<ScalerFeatureMismatch> for PredictionError {
    fn from(error: ScalerFeatureMismatch) -> Self {
        PredictionError::Scaling(error)
    }
}

impl From<InputShapeMismatch> for PredictionError {
    fn from(error: InputShapeMismatch) -> Self {
        PredictionError::Network(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::{IDENTITY, RELU};
    use crate::data::scalers::{Scaler, ScalerKind};
    use crate::model::{Labels, ModelConfig, NetworkConfig, NeuralNetwork};
    use ndarray::{Array, Array4, ArrayD, IxDyn, array, s};

    #[test]
    fn infer_instance_matches_infer_with_a_manual_sample_axis() {
        let predictor = Predictor::binary();

        let instance = array![0.3_f32, 0.7];
        let from_instance = predictor
            .infer_instance(instance.view())
            .unwrap()
            .into_output();
        let from_batch = predictor
            .infer(instance.view().insert_axis(Axis(1)))
            .unwrap()
            .into_output();

        assert_eq!(from_instance, from_batch);
    }

    #[test]
    fn infer_instance_rejects_an_instance_of_the_wrong_size() {
        let predictor = Predictor::binary();

        // The network expects two features; a three-feature instance is rejected by the network.
        let error = predictor
            .infer_instance(array![0.1_f32, 0.2, 0.3].view())
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
    fn infer_ranks_a_binary_networks_output_into_two_classes() {
        let predictor = Predictor::binary();

        let inference = predictor
            .infer_instance(array![0.3_f32, 0.7].view())
            .unwrap();
        let Inference::Classification { ranking, .. } = inference else {
            panic!("a binary task must yield Inference::Classification");
        };

        // A binary network yields the two complementary class probabilities, ranked by
        // descending probability and summing to one.
        assert_eq!(ranking.len(), 2);
        assert!(ranking[0].1 >= ranking[1].1);
        let sum: f32 = ranking.iter().map(|(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn rank_expands_a_single_probability_into_two_complementary_classes() {
        let ranking = rank(array![0.7].view());

        assert_eq!(ranking.len(), 2);
        // Sorted desc: positive class (0.7) first, then its complement (0.3).
        assert_eq!(ranking[0], (1, 0.7));
        let (class, probability) = ranking[1];
        assert_eq!(class, 0);
        assert!((probability - 0.3).abs() < 1e-6);
    }

    #[test]
    fn rank_puts_the_negative_class_first_below_half() {
        let ranking = rank(array![0.2].view());

        assert_eq!(ranking[0].0, 0);
        assert!((ranking[0].1 - 0.8).abs() < 1e-6);
    }

    #[test]
    fn rank_keeps_class_indices_and_sorts_multiclass_outputs_desc() {
        let ranking = rank(array![0.1, 0.6, 0.3].view());

        assert_eq!(
            ranking.iter().map(|(i, _)| *i).collect::<Vec<_>>(),
            vec![1, 2, 0]
        );
    }

    #[test]
    fn infer_yields_classification_for_binary_and_multi_class_tasks() {
        assert!(matches!(
            Predictor::binary()
                .infer_instance(array![0.3_f32, 0.7].view())
                .unwrap(),
            Inference::Classification { .. }
        ));

        assert!(matches!(
            Predictor::multiclass()
                .infer_instance(array![0.3_f32, 0.7].view())
                .unwrap(),
            Inference::Classification { .. }
        ));
    }

    #[test]
    fn infer_yields_values_for_multi_label_and_regression_tasks() {
        let config = NetworkConfig::builder(vec![2]).dense(3, &IDENTITY).build();
        let multi_label = Predictor::unlabeled(
            NeuralNetwork::from_config(config, 0).unwrap(),
            Task::MultiLabel { label_count: 3 },
            None,
        );
        assert!(matches!(
            multi_label
                .infer_instance(array![0.3_f32, 0.7].view())
                .unwrap(),
            Inference::Values(_)
        ));

        let config = NetworkConfig::builder(vec![2]).dense(1, &IDENTITY).build();
        let regression = Predictor::unlabeled(
            NeuralNetwork::from_config(config, 0).unwrap(),
            Task::Regression {
                target_shape: vec![1],
            },
            None,
        );
        assert!(matches!(
            regression
                .infer_instance(array![0.3_f32, 0.7].view())
                .unwrap(),
            Inference::Values(_)
        ));
    }

    #[test]
    fn infer_carries_the_configs_labels_into_the_classification() {
        let config = NetworkConfig::builder(vec![2]).dense(2, &IDENTITY).build();
        let labels = Labels::new(vec!["cat".to_string(), "dog".to_string()]);
        let model_config =
            ModelConfig::new(Task::MultiClass { class_count: 2 }, Some(labels.clone())).unwrap();
        let predictor = Predictor::new(
            NeuralNetwork::from_config(config, 0).unwrap(),
            model_config,
            None,
        );

        let Inference::Classification { labels: got, .. } = predictor
            .infer_instance(array![0.3_f32, 0.7].view())
            .unwrap()
        else {
            panic!("a multi-class task must yield Inference::Classification");
        };
        assert_eq!(got, Some(labels));
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
        let output = Predictor::unlabeled(model, Task::Binary, None)
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
        let predictor = Predictor::unlabeled(network(1), Task::Binary, Some(scaler.clone()));

        // The predictor scales the rank-4 batch per feature before the network: its
        // output matches the bare network run on the manually scaled inputs.
        let via_predictor = predictor.output(inputs.view()).unwrap();
        let mut scaled = inputs.clone().into_dyn();
        scaler.apply_inplace(scaled.view_mut()).unwrap();
        let via_network = Predictor::unlabeled(network(1), Task::Binary, None)
            .output(scaled.view())
            .unwrap();
        assert_eq!(via_predictor, via_network);

        // A single rank-3 instance is scaled with the same parameters, so its classification
        // matches the first column of the batch probabilities.
        let instance = inputs.slice(s![.., .., .., 0]).to_owned();
        let inference = predictor.infer_instance(instance.view()).unwrap();
        let Inference::Classification {
            ranking: single, ..
        } = inference
        else {
            panic!("a binary task must yield Inference::Classification");
        };
        let batch = via_predictor.into_dimensionality::<Ix2>().unwrap();
        let expected = rank(batch.column(0));
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
        let predictor = Predictor::unlabeled(network, Task::Binary, Some(scaler));

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
        let output = Predictor::unlabeled(model, Task::Binary, None)
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
