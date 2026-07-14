//! The [`Predictor`]: a trained network paired with the scaler fitted alongside it, plus the
//! error it raises when an instance's shape does not match.

use crate::activations::{Activation, IDENTITY, SIGMOID, SOFTMAX};
use crate::data::scalers::{Scaler, ScalerFeatureMismatch, ScalerMethod};
use crate::model::{Activations, InputShapeMismatch, NeuralNetwork};
use crate::task::Task;
use ndarray::{ArrayD, ArrayView, Axis, Dimension};
use std::fmt;

/// A trained [`NeuralNetwork`] paired with the [`Task`] it was trained for and the
/// scaler fitted alongside it.
#[derive(Clone, Debug)]
pub struct Predictor {
    /// The trained network.
    pub network: NeuralNetwork,
    /// The learning task the network was trained for; drives how its logits are read as outputs.
    pub task: Task,
    /// The scaler applied to raw inputs before prediction, when one is present.
    pub scaler: Option<ScalerMethod>,
}

impl Predictor {
    /// Pairs a network and its task with an optional scaler.
    pub fn new(network: NeuralNetwork, task: Task, scaler: Option<ScalerMethod>) -> Self {
        Self {
            network,
            task,
            scaler,
        }
    }

    /// Runs raw `inputs` (any rank, samples on the trailing axis) through the model: scales them
    /// when a scaler is present, then finalizes the output stage for the task. The returned
    /// activations' [`output`](Activations::output) holds the final probabilities or values.
    ///
    /// # Errors
    /// [`PredictionError::Scaling`] when a scaler is present and `inputs` do not match its fitted
    /// feature count, or [`PredictionError::Network`] when they do not match the network's input shape.
    pub fn infer<D: Dimension>(
        &self,
        inputs: ArrayView<f32, D>,
    ) -> Result<Activations, PredictionError> {
        let mut activations = match &self.scaler {
            Some(scaler) => {
                let scaled = scaler.apply(inputs.into_dyn())?;
                self.network.forward(scaled.view())?
            }
            None => self.network.forward(inputs)?,
        };

        activations.finalize(output_activation(&self.task));
        Ok(activations)
    }

    /// Runs a single raw `instance` (any rank, with no sample axis) through the model, inserting
    /// the trailing sample axis it lacks before delegating to [`infer`](Predictor::infer).
    ///
    /// # Errors
    /// As [`infer`](Predictor::infer).
    pub fn infer_instance<D: Dimension>(
        &self,
        instance: ArrayView<f32, D>,
    ) -> Result<Activations, PredictionError> {
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

/// The output activation that reads a task's logits as final outputs: sigmoid for the
/// independent-probability tasks (binary / multi-label), softmax for multi-class, and identity
/// for regression.
fn output_activation(task: &Task) -> &'static dyn Activation {
    match task {
        Task::Binary | Task::MultiLabel { .. } => &**SIGMOID,
        Task::MultiClass { .. } => &**SOFTMAX,
        Task::Regression { .. } => &**IDENTITY,
    }
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
    use crate::model::NetworkConfig;
    use ndarray::array;

    #[test]
    fn infer_instance_matches_infer_with_a_manual_sample_axis() {
        let config = NetworkConfig::builder(vec![2])
            .dense(4, &RELU)
            .dense(1, &IDENTITY)
            .build();
        let predictor = Predictor::new(
            NeuralNetwork::from_config(config, 0).unwrap(),
            Task::Binary,
            None,
        );

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
        let config = NetworkConfig::builder(vec![2])
            .dense(4, &RELU)
            .dense(1, &IDENTITY)
            .build();
        let predictor = Predictor::new(
            NeuralNetwork::from_config(config, 0).unwrap(),
            Task::Binary,
            None,
        );

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
}
