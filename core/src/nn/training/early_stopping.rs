use super::evaluator::Evaluator;
use crate::data::ModelDataset;
use crate::model::NeuralNetwork;
use std::fmt;

/// Declarative early-stopping settings, part of a [`crate::training::HyperParameters`] spec.
/// Constructs the runtime [`EarlyStopping`] via [`EarlyStopping::new`].
#[derive(Clone, Debug, PartialEq)]
pub struct EarlyStoppingConfig {
    /// The number of consecutive epochs without improvement after which training will be stopped.
    patience: usize,
    /// Whether to restore the model to the state with the best observed loss when stopping.
    restore_best_model: bool,
}

/// Returned by [`EarlyStoppingConfig::new`] when `patience` is zero.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EarlyStoppingConfigError;

impl fmt::Display for EarlyStoppingConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "early stopping patience must be greater than zero")
    }
}

impl std::error::Error for EarlyStoppingConfigError {}

impl EarlyStoppingConfig {
    /// Creates a new early-stopping config.
    /// # Errors
    /// Returns [`EarlyStoppingConfigError`] when `patience` is zero.
    pub fn new(
        patience: usize,
        restore_best_model: bool,
    ) -> Result<Self, EarlyStoppingConfigError> {
        if patience > 0 {
            Ok(EarlyStoppingConfig {
                patience,
                restore_best_model,
            })
        } else {
            Err(EarlyStoppingConfigError)
        }
    }

    pub fn patience(&self) -> usize {
        self.patience
    }

    pub fn restore_best_model(&self) -> bool {
        self.restore_best_model
    }
}

/// Early stopping mechanism is used to halt training when the model's performance on a validation set
/// stops improving, thereby preventing overfitting.
pub struct EarlyStopping {
    /// The number of consecutive epochs without improvement after which training will be stopped.
    patience: usize,
    /// The best loss observed so far during training.
    best_loss: f32,
    /// The number of consecutive epochs without improvement.
    epochs_without_improvement: usize,
    /// Whether to restore the model to the state with the best observed loss when stopping.
    restore_best_model: bool,
    /// The best model observed during training, if `restore_best_model` is true.
    pub best_model: Option<NeuralNetwork>,
}

impl EarlyStopping {
    /// Creates a new `EarlyStopping` instance from the given config, seeding
    /// `best_model` with `init_model` so a divergence at the first epoch
    /// (before any `check` call) can still recover. No-op when
    /// `restore_best_model` is false.
    pub fn new(config: EarlyStoppingConfig, init_model: &NeuralNetwork) -> Self {
        EarlyStopping {
            patience: config.patience,
            best_loss: f32::INFINITY,
            epochs_without_improvement: 0,
            restore_best_model: config.restore_best_model,
            best_model: config.restore_best_model.then(|| init_model.clone()),
        }
    }

    /// Evaluates `model` on `validation` and checks whether training should
    /// be stopped based on the resulting loss.
    pub fn check(
        &mut self,
        validation: &ModelDataset,
        model: &NeuralNetwork,
        evaluator: &Evaluator,
    ) -> bool {
        let loss = evaluator.eval_dataset(model, validation).loss;

        self.observe(loss, model)
    }

    /// Checks if training should be stopped based on the current loss.
    fn observe(&mut self, loss: f32, model: &NeuralNetwork) -> bool {
        if loss < self.best_loss {
            self.best_loss = loss;
            self.epochs_without_improvement = 0;

            if self.restore_best_model {
                self.best_model = Some(model.clone());
            }
        } else {
            self.epochs_without_improvement += 1;
        }

        self.epochs_without_improvement >= self.patience
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::SIGMOID;
    use crate::model::NeuronLayerSpec;
    use ndarray::Array1;

    #[test]
    fn early_stopping_restores_best_model() {
        // Sequence: loss 1.0 → 0.5 (best, saved) → 0.8 → 0.9 (patience=2 exhausted)
        // After stop, best_model must reflect the state at loss=0.5, not the final state.
        let specs = NeuronLayerSpec::network_for(vec![2], &*SIGMOID, 2);

        let model_initial = NeuralNetwork::initialization(2, &specs);
        let mut es = EarlyStopping::new(EarlyStoppingConfig::new(2, true).unwrap(), &model_initial);

        let mut model_at_best = model_initial.clone();
        // Give model_at_best a distinctive bias so we can tell it apart
        model_at_best.layers[0].biases = Array1::from_vec(vec![42.0, 42.0]);

        assert!(!es.observe(1.0, &model_initial)); // improvement: saved
        assert!(!es.observe(0.5, &model_at_best)); // new best: overwrites saved
        assert!(!es.observe(0.8, &model_initial)); // regression: 1 epoch without improvement
        assert!(es.observe(0.9, &model_initial)); // regression: patience=2 exhausted → true

        let best = es
            .best_model
            .expect("best_model should be Some after early stopping");
        assert_eq!(
            best.layers[0].biases, model_at_best.layers[0].biases,
            "best_model should be the epoch with loss=0.5, not the final state"
        );
    }

    #[test]
    fn early_stopping_skips_snapshot_when_restore_disabled() {
        // With restore_best_model = false, an improvement must NOT clone the model.
        let specs = NeuronLayerSpec::network_for(vec![2], &*SIGMOID, 2);
        let model = NeuralNetwork::initialization(2, &specs);
        let mut es = EarlyStopping::new(EarlyStoppingConfig::new(2, false).unwrap(), &model);

        assert!(!es.observe(1.0, &model)); // improvement, but no snapshot is taken
        assert!(es.best_model.is_none());
    }

    #[test]
    fn new_seeds_best_model_when_restore_enabled() {
        let specs = NeuronLayerSpec::network_for(vec![2], &*SIGMOID, 2);
        let model = NeuralNetwork::initialization(2, &specs);
        let es = EarlyStopping::new(EarlyStoppingConfig::new(2, true).unwrap(), &model);

        assert!(es.best_model.is_some());
    }

    #[test]
    fn new_does_not_seed_best_model_when_restore_disabled() {
        let specs = NeuronLayerSpec::network_for(vec![2], &*SIGMOID, 2);
        let model = NeuralNetwork::initialization(2, &specs);
        let es = EarlyStopping::new(EarlyStoppingConfig::new(2, false).unwrap(), &model);

        assert!(es.best_model.is_none());
    }

    #[test]
    fn rejects_zero_patience() {
        match EarlyStoppingConfig::new(0, false) {
            Err(EarlyStoppingConfigError) => {}
            Ok(_) => panic!("expected EarlyStoppingConfigError"),
        }
    }
}
