use crate::accuracies::Accuracy;
use crate::data::{ModelDataset, ModelSplit};
use crate::evaluation::{Evaluation, EvaluationSet};
use crate::loss_functions::LossFunction;
use crate::model::{FeatureCountMismatch, NeuralNetwork};
use ndarray::ArrayView2;
use std::sync::Arc;

/// Computes [`Evaluation`]s and [`EvaluationSet`]s for a model, given a fixed
/// loss function and accuracy metric.
pub struct Evaluator {
    loss_fn: Arc<dyn LossFunction>,
    accuracy: Arc<dyn Accuracy>,
}

impl Evaluator {
    pub fn new(loss_fn: Arc<dyn LossFunction>, accuracy: Arc<dyn Accuracy>) -> Self {
        Self { loss_fn, accuracy }
    }

    /// Evaluates the model on the training, validation (if available), and test datasets.
    ///
    /// # Errors
    /// [`FeatureCountMismatch`] when `model`'s input size does not match the split's features.
    pub fn eval_set(
        &self,
        model: &NeuralNetwork,
        split: &ModelSplit,
    ) -> Result<EvaluationSet, FeatureCountMismatch> {
        Ok(EvaluationSet {
            train: self.eval_dataset(model, &split.train)?,
            validation: split
                .validation
                .as_ref()
                .map(|v| self.eval_dataset(model, v))
                .transpose()?,
            test: self.eval_dataset(model, &split.test)?,
        })
    }

    /// Evaluates the model on a single dataset.
    ///
    /// # Errors
    /// [`FeatureCountMismatch`] when `model`'s input size does not match the dataset's features.
    pub fn eval_dataset(
        &self,
        model: &NeuralNetwork,
        dataset: &ModelDataset,
    ) -> Result<Evaluation, FeatureCountMismatch> {
        let predictions = model.predict(dataset.inputs.view())?;
        Ok(self.eval_predictions(predictions.view(), dataset.targets.view()))
    }

    /// Evaluates precomputed predictions against the true targets.
    pub fn eval_predictions(
        &self,
        predictions: ArrayView2<f32>,
        targets: ArrayView2<f32>,
    ) -> Evaluation {
        Evaluation {
            loss: self.loss_fn.compute(predictions, targets),
            accuracy: self.accuracy.compute(predictions, targets),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accuracies::BINARY_ACCURACY;
    use crate::activations::SIGMOID;
    use crate::loss_functions::CROSS_ENTROPY_LOSS;
    use crate::model::NeuronLayer;
    use ndarray::array;

    /// A 2-input → 1-output sigmoid network whose weights/bias are zeroed,
    /// so every prediction is exactly 0.5 regardless of the input.
    fn constant_half_model() -> NeuralNetwork {
        NeuralNetwork {
            layers: vec![NeuronLayer {
                weights: array![[0.0, 0.0]],
                biases: array![0.0],
                activation: SIGMOID.clone(),
            }],
        }
    }

    fn dataset(inputs: ndarray::Array2<f32>, targets: ndarray::Array2<f32>) -> ModelDataset {
        ModelDataset { inputs, targets }
    }

    fn evaluator() -> Evaluator {
        Evaluator::new(CROSS_ENTROPY_LOSS.clone(), BINARY_ACCURACY.clone())
    }

    #[test]
    fn eval_predictions_scores_perfect_classification() {
        let predictions = array![[0.99, 0.01]];
        let targets = array![[1.0, 0.0]];

        let eval = evaluator().eval_predictions(predictions.view(), targets.view());

        assert!(
            eval.loss >= 0.0 && eval.loss < 0.1,
            "loss was {}",
            eval.loss
        );
        assert_eq!(eval.accuracy, 100.0);
    }

    #[test]
    fn eval_dataset_matches_eval_predictions() {
        let model = constant_half_model();
        let data = dataset(array![[0.2, 0.8], [0.3, 0.7]], array![[1.0, 0.0]]);

        let from_dataset = evaluator().eval_dataset(&model, &data).unwrap();
        let from_preds = evaluator().eval_predictions(
            model.predict(data.inputs.view()).unwrap().view(),
            data.targets.view(),
        );

        assert_eq!(from_dataset.loss, from_preds.loss);
        assert_eq!(from_dataset.accuracy, from_preds.accuracy);
    }

    #[test]
    fn eval_set_without_validation() {
        let model = constant_half_model();
        let split = ModelSplit {
            train: dataset(array![[0.1, 0.9], [0.2, 0.8]], array![[1.0, 0.0]]),
            validation: None,
            test: dataset(array![[0.4, 0.6], [0.5, 0.5]], array![[0.0, 1.0]]),
        };

        let set = evaluator().eval_set(&model, &split).unwrap();

        assert!(set.validation.is_none());
        assert!(set.train.loss.is_finite());
        assert_eq!(set.train.loss, set.test.loss);
    }

    #[test]
    fn eval_set_with_validation() {
        let model = constant_half_model();
        let split = ModelSplit {
            train: dataset(array![[0.1, 0.9], [0.2, 0.8]], array![[1.0, 0.0]]),
            validation: Some(dataset(array![[0.3, 0.7], [0.6, 0.4]], array![[1.0, 0.0]])),
            test: dataset(array![[0.4, 0.6], [0.5, 0.5]], array![[0.0, 1.0]]),
        };

        let set = evaluator().eval_set(&model, &split).unwrap();

        assert!(set.validation.is_some());
    }
}
