/// Represents the evaluation metrics of a model after training or validation.
#[derive(Clone, Copy, Debug)]
pub struct Evaluation {
    /// The computed loss value of the model.
    pub loss: f32,
    /// The computed accuracy of the model.
    pub accuracy: f32,
}

/// Represents the evaluation results of a model on training, validation, and test datasets.
#[derive(Clone, Copy, Debug)]
pub struct EvaluationSet {
    pub train: Evaluation,
    pub validation: Option<Evaluation>,
    pub test: Evaluation,
}
