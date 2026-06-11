use crate::callbacks::{Callbacks, TrainingCallback};
use crate::data::ModelSplit;
use crate::evaluation::EvaluationSet;
use crate::evaluator::Evaluator;
use crate::model::NeuralNetwork;
use crate::training::{EarlyStopping, TrainingConfig};
use crate::training_outcome::TrainingOutcome;
use std::io::Result;

/// The result of a completed [`TrainingLoop::run`].
pub struct TrainingReport {
    pub outcome: TrainingOutcome,
    /// The final model. On a fatal divergence, these are the diverged weights —
    /// the caller should check `outcome` before using or persisting it.
    pub model: NeuralNetwork,
    /// `None` only when `outcome` is `Diverged { recovered: false }`.
    pub final_evaluation: Option<EvaluationSet>,
    pub final_epoch: usize,
}

/// Orchestrates a training run: forward/backward passes, scheduled evaluation,
/// early stopping, and divergence handling. Performs no I/O of its own — all
/// side effects (persistence, display) go through `callbacks`.
pub struct TrainingLoop {
    pub model: NeuralNetwork,
    pub callbacks: Callbacks,
    pub split: ModelSplit,
    pub evaluator: Evaluator,
    pub config: TrainingConfig,
    pub early_stopping: Option<EarlyStopping>,
    pub epoch_start: usize,
    pub restore_best_model: bool,
}

impl TrainingLoop {
    /// Whether `epoch` should trigger an evaluation + snapshot, given
    /// `eval_interval == 0` means "no checkpoints at all".
    pub fn is_checkpoint(eval_interval: usize, epoch: usize) -> bool {
        eval_interval != 0 && epoch.is_multiple_of(eval_interval)
    }

    pub fn run(mut self) -> Result<TrainingReport> {
        self.callbacks.on_train_start(&self.config)?;

        let epochs = self.config.epochs;
        let eval_interval = self.config.eval_interval;

        if self.epoch_start == 0 && eval_interval != 0 {
            self.evaluate(0)?;
        }

        let mut early_stopping = self.early_stopping.take();
        // Seed best model before epoch 1 so divergence at the first epoch can recover.
        if let Some(ref mut es) = early_stopping
            && self.split.validation.is_some()
        {
            es.seed_best_model(&self.model);
        }

        let mut outcome = TrainingOutcome::Completed;
        let mut final_epoch = self.epoch_start;
        let mut final_evaluation = None;

        for epoch in (self.epoch_start + 1)..=(self.epoch_start + epochs) {
            self.model.train(
                &self.split.train,
                &self.config.loss,
                self.config.optimizer.as_mut(),
                self.config.scheduler.as_mut(),
                &self.config.clipping,
                self.config.batch_size,
            );

            final_epoch = epoch;
            final_evaluation = None;

            if !self.model.is_finite() {
                let recovered = early_stopping.as_mut().and_then(|es| es.best_model.take());

                outcome = match recovered {
                    Some(best) => {
                        self.model = best;
                        TrainingOutcome::Diverged { recovered: true }
                    }
                    None => TrainingOutcome::Diverged { recovered: false },
                };
                break;
            }

            self.callbacks.on_epoch_end(epoch)?;

            if let Some(ref mut es) = early_stopping
                && let Some(validation) = &self.split.validation
            {
                let preds = self.model.predict(validation.inputs.view());
                let loss = self
                    .evaluator
                    .eval_predictions(preds.view(), validation.targets.view())
                    .loss;

                if es.check(loss, &self.model) {
                    if self.restore_best_model {
                        self.model = es
                            .best_model
                            .as_ref()
                            .expect("Best model should be available")
                            .clone();
                    }
                    outcome = TrainingOutcome::EarlyStopped {
                        restored: self.restore_best_model,
                    };
                    break;
                }
            }

            if Self::is_checkpoint(eval_interval, epoch) {
                final_evaluation = Some(self.evaluate(epoch)?);
            }
        }

        let final_evaluation = match final_evaluation {
            Some(eval) => Some(eval),
            // A diverged model with no recovered fallback is non-finite:
            // evaluating it would panic, so there is nothing to report.
            None if !self.model.is_finite() => None,
            None => Some(self.evaluate(final_epoch)?),
        };

        self.callbacks
            .on_train_end(outcome, final_evaluation.as_ref(), final_epoch)?;

        Ok(TrainingReport {
            outcome,
            model: self.model,
            final_evaluation,
            final_epoch,
        })
    }

    /// Computes an evaluation for the current model and reports it via
    /// [`Callbacks::on_evaluate`].
    fn evaluate(&mut self, epoch: usize) -> Result<EvaluationSet> {
        let eval = self.evaluator.eval_set(&self.model, &self.split);
        self.callbacks.on_evaluate(&self.model, &eval, epoch)?;
        Ok(eval)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accuracies::accuracy_for;
    use crate::activations::SIGMOID;
    use crate::data::ModelDataset;
    use crate::loss_functions::CROSS_ENTROPY_LOSS;
    use crate::model::NeuronLayer;
    use crate::optimizers::Adam;
    use crate::schedulers::ConstantScheduler;
    use crate::training::{GradientClipping, LearningRate};
    use ndarray::array;
    use std::cell::RefCell;
    use std::rc::Rc;

    fn sample_dataset() -> ModelDataset {
        ModelDataset {
            inputs: array![[0.1, 0.9, 0.2, 0.8], [0.2, 0.8, 0.3, 0.7]],
            targets: array![[1.0, 0.0, 1.0, 0.0]],
        }
    }

    fn sample_split(with_validation: bool) -> ModelSplit {
        ModelSplit {
            train: sample_dataset(),
            validation: with_validation.then(sample_dataset),
            test: sample_dataset(),
        }
    }

    /// A fixed (non-random) 2-input -> 1-output sigmoid network, so loss
    /// sequences are fully deterministic across runs.
    fn sample_model() -> NeuralNetwork {
        NeuralNetwork {
            layers: vec![NeuronLayer {
                weights: array![[0.1, -0.2]],
                biases: array![0.05],
                activation: SIGMOID.clone(),
            }],
        }
    }

    fn sample_evaluator() -> Evaluator {
        Evaluator::new(CROSS_ENTROPY_LOSS.clone(), accuracy_for(2))
    }

    fn sample_config(epochs: usize, eval_interval: usize, lr: f32) -> TrainingConfig {
        TrainingConfig {
            epochs,
            eval_interval,
            batch_size: None,
            loss: CROSS_ENTROPY_LOSS.clone(),
            optimizer: Box::new(Adam::with_defaults(LearningRate::new(lr))),
            scheduler: Box::new(ConstantScheduler::new(LearningRate::new(lr))),
            clipping: GradientClipping::None,
        }
    }

    #[derive(Default)]
    struct Counts {
        train_starts: usize,
        epoch_ends: usize,
        evaluated_epochs: Vec<usize>,
        train_ends: usize,
        last_outcome: Option<TrainingOutcome>,
    }

    struct CountingCallback(Rc<RefCell<Counts>>);

    impl TrainingCallback for CountingCallback {
        fn on_train_start(&mut self, _config: &TrainingConfig) -> Result<()> {
            self.0.borrow_mut().train_starts += 1;
            Ok(())
        }

        fn on_epoch_end(&mut self, _epoch: usize) -> Result<()> {
            self.0.borrow_mut().epoch_ends += 1;
            Ok(())
        }

        fn on_evaluate(
            &mut self,
            _model: &NeuralNetwork,
            _eval: &EvaluationSet,
            epoch: usize,
        ) -> Result<()> {
            self.0.borrow_mut().evaluated_epochs.push(epoch);
            Ok(())
        }

        fn on_train_end(
            &mut self,
            outcome: TrainingOutcome,
            _eval: Option<&EvaluationSet>,
            _epoch: usize,
        ) -> Result<()> {
            let mut counts = self.0.borrow_mut();
            counts.train_ends += 1;
            counts.last_outcome = Some(outcome);
            Ok(())
        }
    }

    fn training_loop(
        config: TrainingConfig,
        with_validation: bool,
        early_stopping: Option<EarlyStopping>,
        restore_best_model: bool,
        counts: Rc<RefCell<Counts>>,
    ) -> TrainingLoop {
        TrainingLoop {
            model: sample_model(),
            callbacks: Callbacks::new(vec![Box::new(CountingCallback(counts))]),
            split: sample_split(with_validation),
            evaluator: sample_evaluator(),
            config,
            early_stopping,
            epoch_start: 0,
            restore_best_model,
        }
    }

    #[test]
    fn is_checkpoint_cases() {
        assert!(TrainingLoop::is_checkpoint(3, 0));
        assert!(TrainingLoop::is_checkpoint(3, 3));
        assert!(!TrainingLoop::is_checkpoint(3, 1));
        assert!(!TrainingLoop::is_checkpoint(3, 4));
        assert!(!TrainingLoop::is_checkpoint(0, 0));
        assert!(!TrainingLoop::is_checkpoint(0, 3));
    }

    #[test]
    fn evaluation_schedule_includes_epoch_zero_multiples_and_final_epoch() {
        let counts = Rc::new(RefCell::new(Counts::default()));
        let config = sample_config(7, 3, 0.01);
        let training_loop = training_loop(config, false, None, false, counts.clone());

        let report = training_loop.run().unwrap();

        assert_eq!(report.final_epoch, 7);
        let counts = counts.borrow();
        assert_eq!(counts.train_starts, 1);
        assert_eq!(counts.train_ends, 1);
        assert_eq!(counts.epoch_ends, 7);
        // epoch 0 + multiples of 3 (3, 6) + final epoch 7 (not a multiple of 3)
        assert_eq!(counts.evaluated_epochs, vec![0, 3, 6, 7]);
    }

    #[test]
    fn eval_interval_zero_only_emits_the_final_evaluation() {
        let counts = Rc::new(RefCell::new(Counts::default()));
        let config = sample_config(3, 0, 0.01);
        let training_loop = training_loop(config, false, None, false, counts.clone());

        let report = training_loop.run().unwrap();

        assert!(report.final_evaluation.is_some());
        assert_eq!(counts.borrow().evaluated_epochs, vec![3]);
    }

    #[test]
    fn early_stopping_halts_training_and_restores_best_model() {
        let counts = Rc::new(RefCell::new(Counts::default()));
        let config = sample_config(50, 1, 5.0);
        let early_stopping = Some(EarlyStopping::new(1, true));
        let training_loop = training_loop(config, true, early_stopping, true, counts.clone());

        let report = training_loop.run().unwrap();

        assert_eq!(
            report.outcome,
            TrainingOutcome::EarlyStopped { restored: true }
        );
        assert!(report.final_epoch < 50);
        assert_eq!(
            counts.borrow().last_outcome,
            Some(TrainingOutcome::EarlyStopped { restored: true })
        );
    }

    #[test]
    fn fatal_divergence_without_recovery_is_reported_as_data() {
        let counts = Rc::new(RefCell::new(Counts::default()));
        let config = sample_config(5, 1, 1e30);
        let training_loop = training_loop(config, false, None, false, counts.clone());

        let report = training_loop.run().unwrap();

        assert_eq!(
            report.outcome,
            TrainingOutcome::Diverged { recovered: false }
        );
        assert!(report.final_evaluation.is_none());
        assert_eq!(
            counts.borrow().last_outcome,
            Some(TrainingOutcome::Diverged { recovered: false })
        );
    }

    #[test]
    fn divergence_recovers_seeded_best_model_when_restore_enabled() {
        let counts = Rc::new(RefCell::new(Counts::default()));
        let config = sample_config(5, 1, 1e30);
        let early_stopping = Some(EarlyStopping::new(10, true));
        let training_loop = training_loop(config, true, early_stopping, true, counts.clone());

        let report = training_loop.run().unwrap();

        assert_eq!(
            report.outcome,
            TrainingOutcome::Diverged { recovered: true }
        );
        assert!(report.model.is_finite());
        assert!(report.final_evaluation.is_some());
    }
}
