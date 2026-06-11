use crate::model::NeuralNetwork;

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
    /// Creates a new `EarlyStopping` instance with the specified patience and restore_best_model flag.
    /// # Panics
    /// - When the `patience` is zero.
    pub fn new(patience: usize, restore_best_model: bool) -> Self {
        assert!(patience > 0, "Patience must be greater than zero.");

        EarlyStopping {
            patience,
            best_loss: f32::INFINITY,
            epochs_without_improvement: 0,
            restore_best_model,
            best_model: None,
        }
    }

    /// Seeds `best_model` with the given model so divergence at epoch 1 (before any
    /// `check` call) can still recover. No-op when `restore_best_model` is false.
    pub fn seed_best_model(&mut self, model: &NeuralNetwork) {
        if self.restore_best_model {
            self.best_model = Some(model.clone());
        }
    }

    /// Checks if training should be stopped based on the current loss.
    pub fn check(&mut self, current_loss: f32, model: &NeuralNetwork) -> bool {
        if current_loss < self.best_loss {
            self.best_loss = current_loss;
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
        let mut es = EarlyStopping::new(2, true);

        let model_initial = NeuralNetwork::initialization(2, &specs);
        let mut model_at_best = model_initial.clone();
        // Give model_at_best a distinctive bias so we can tell it apart
        model_at_best.layers[0].biases = Array1::from_vec(vec![42.0, 42.0]);

        assert!(!es.check(1.0, &model_initial)); // improvement: saved
        assert!(!es.check(0.5, &model_at_best)); // new best: overwrites saved
        assert!(!es.check(0.8, &model_initial)); // regression: 1 epoch without improvement
        assert!(es.check(0.9, &model_initial)); // regression: patience=2 exhausted → true

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
        let mut es = EarlyStopping::new(2, false);

        assert!(!es.check(1.0, &model)); // improvement, but no snapshot is taken
        assert!(es.best_model.is_none());
    }

    #[test]
    fn seed_best_model_seeds_when_restore_enabled() {
        let specs = NeuronLayerSpec::network_for(vec![2], &*SIGMOID, 2);
        let model = NeuralNetwork::initialization(2, &specs);
        let mut es = EarlyStopping::new(2, true);

        es.seed_best_model(&model);

        assert!(es.best_model.is_some());
    }

    #[test]
    fn seed_best_model_is_noop_when_restore_disabled() {
        let specs = NeuronLayerSpec::network_for(vec![2], &*SIGMOID, 2);
        let model = NeuralNetwork::initialization(2, &specs);
        let mut es = EarlyStopping::new(2, false);

        es.seed_best_model(&model);

        assert!(es.best_model.is_none());
    }
}
