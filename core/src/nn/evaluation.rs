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
