use crate::accuracies::Accuracy;
use crate::data::{ModelDataset, ModelSplit};
use crate::loss_functions::LossFunction;
use crate::model::NeuralNetwork;
use ndarray::ArrayView2;
use std::sync::Arc;

/// Represents the evaluation metrics of a model after training or validation.
#[derive(Clone, Copy, Debug)]
pub struct Evaluation {
    /// The computed loss value of the model.
    pub loss: f32,
    /// The computed accuracy of the model.
    pub accuracy: f32,
}

impl Evaluation {
    /// Evaluates the model on the provided dataset using the specified loss function and accuracy metric.
    /// # Arguments
    /// - `model`: The neural network model to be evaluated.
    /// - `loss_function`: The loss function used to compute the loss.
    /// - `accuracy`: The accuracy metric used to compute the accuracy.
    /// - `dataset`: The dataset containing inputs and targets for evaluation.
    pub fn using_model(
        model: &NeuralNetwork,
        loss_function: &Arc<dyn LossFunction>,
        accuracy: &Arc<dyn Accuracy>,
        dataset: &ModelDataset,
    ) -> Self {
        Self::from_predictions(
            loss_function,
            accuracy,
            model.predict(dataset.inputs.view()).view(),
            dataset.targets.view(),
        )
    }

    /// Evaluates the model using precomputed predictions and targets.
    /// # Arguments
    /// - `loss_function`: The loss function used to compute the loss.
    /// - `accuracy`: The accuracy metric used to compute the accuracy.
    /// - `predictions`: A 2D array representing the model's predictions.
    /// - `targets`: A 2D array representing the true labels for the predictions.
    pub fn from_predictions(
        loss_function: &Arc<dyn LossFunction>,
        accuracy: &Arc<dyn Accuracy>,
        predictions: ArrayView2<f32>,
        targets: ArrayView2<f32>,
    ) -> Self {
        let loss = loss_function.compute(predictions.view(), targets.view());
        let accuracy = accuracy.compute(predictions.view(), targets.view());

        Evaluation { loss, accuracy }
    }
}

/// Represents the evaluation results of a model on training, validation, and test datasets.
#[derive(Clone, Copy, Debug)]
pub struct EvaluationSet {
    pub train: Evaluation,
    pub validation: Option<Evaluation>,
    pub test: Evaluation,
}

impl EvaluationSet {
    /// Evaluates the model on the training, validation (if available), and test datasets.
    /// # Arguments
    /// - `model`: The neural network model to be evaluated.
    /// - `loss_function`: The loss function used to compute the loss.
    /// - `accuracy`: The accuracy metric used to compute the accuracy.
    /// - `split`: The dataset split containing training, validation, and test datasets.
    /// - `train_predictions`: Optional precomputed predictions for the training dataset to avoid redundant computation.
    pub fn using_model(
        model: &NeuralNetwork,
        loss_function: &Arc<dyn LossFunction>,
        accuracy: &Arc<dyn Accuracy>,
        split: &ModelSplit,
        train_predictions: Option<ArrayView2<f32>>,
    ) -> Self {
        EvaluationSet {
            train: match train_predictions {
                Some(predictions) => Evaluation::from_predictions(
                    loss_function,
                    accuracy,
                    predictions,
                    split.train.targets.view(),
                ),
                None => Evaluation::using_model(model, loss_function, accuracy, &split.train),
            },
            validation: split
                .validation
                .as_ref()
                .map(|val| Evaluation::using_model(model, loss_function, accuracy, val)),
            test: Evaluation::using_model(model, loss_function, accuracy, &split.test),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accuracies::BINARY_ACCURACY;
    use crate::activations::SIGMOID;
    use crate::data::ModelDataset;
    use crate::loss_functions::CROSS_ENTROPY_LOSS;
    use crate::model::{NeuralNetwork, NeuronLayer};
    use ndarray::array;

    fn loss() -> Arc<dyn LossFunction> {
        CROSS_ENTROPY_LOSS.clone()
    }

    fn accuracy() -> Arc<dyn Accuracy> {
        BINARY_ACCURACY.clone()
    }

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

    #[test]
    fn from_predictions_scores_perfect_classification() {
        // Confident, correct binary predictions → ~0 loss, 100% accuracy.
        let predictions = array![[0.99, 0.01]];
        let targets = array![[1.0, 0.0]];

        let eval = Evaluation::from_predictions(
            &loss(),
            &accuracy(),
            predictions.view(),
            targets.view(),
        );

        assert!(eval.loss >= 0.0 && eval.loss < 0.1, "loss was {}", eval.loss);
        assert_eq!(eval.accuracy, 100.0);
    }

    #[test]
    fn using_model_matches_from_predictions() {
        let model = constant_half_model();
        let data = dataset(array![[0.2, 0.8], [0.3, 0.7]], array![[1.0, 0.0]]);

        let from_model = Evaluation::using_model(&model, &loss(), &accuracy(), &data);
        let from_preds = Evaluation::from_predictions(
            &loss(),
            &accuracy(),
            model.predict(data.inputs.view()).view(),
            data.targets.view(),
        );

        assert_eq!(from_model.loss, from_preds.loss);
        assert_eq!(from_model.accuracy, from_preds.accuracy);
    }

    #[test]
    fn evaluation_set_without_validation_or_cached_predictions() {
        let model = constant_half_model();
        let split = ModelSplit {
            train: dataset(array![[0.1, 0.9], [0.2, 0.8]], array![[1.0, 0.0]]),
            validation: None,
            test: dataset(array![[0.4, 0.6], [0.5, 0.5]], array![[0.0, 1.0]]),
        };

        let set = EvaluationSet::using_model(&model, &loss(), &accuracy(), &split, None);

        assert!(set.validation.is_none());
        // Constant-0.5 predictions: train/test losses are equal and finite.
        assert!(set.train.loss.is_finite());
        assert_eq!(set.train.loss, set.test.loss);
    }

    #[test]
    fn evaluation_set_uses_cached_train_predictions_and_validation() {
        let model = constant_half_model();
        let split = ModelSplit {
            train: dataset(array![[0.1, 0.9], [0.2, 0.8]], array![[1.0, 0.0]]),
            validation: Some(dataset(array![[0.3, 0.7], [0.6, 0.4]], array![[1.0, 0.0]])),
            test: dataset(array![[0.4, 0.6], [0.5, 0.5]], array![[0.0, 1.0]]),
        };

        // Provide perfect cached predictions for the train split.
        let cached = array![[0.99, 0.01]];
        let set =
            EvaluationSet::using_model(&model, &loss(), &accuracy(), &split, Some(cached.view()));

        // Train metrics come from the cached predictions, not the constant-0.5 model.
        assert_eq!(set.train.accuracy, 100.0);
        assert!(set.validation.is_some());
    }
}
