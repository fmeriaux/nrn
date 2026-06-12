use super::callbacks::{Callbacks, TrainingCallback};
use super::config::TrainingConfig;
use super::early_stopping::EarlyStopping;
use super::evaluator::Evaluator;
use super::outcome::TrainingOutcome;
use crate::accuracies::accuracy_for;
use crate::data::ModelSplit;
use crate::evaluation::EvaluationSet;
use crate::model::NeuralNetwork;
use std::io::Result as IoResult;

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

/// A fatal divergence: the model's weights became non-finite and no
/// recovered (early-stopping) fallback was available.
#[derive(Debug)]
pub struct FatalDivergence {
    pub final_epoch: usize,
}

impl std::fmt::Display for FatalDivergence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "model diverged at epoch {}: non-finite weights",
            self.final_epoch
        )
    }
}

impl std::error::Error for FatalDivergence {}

impl TrainingReport {
    /// Turns a `Diverged { recovered: false }` outcome into an error;
    /// every other outcome (including a recovered divergence) is a success.
    pub fn into_result(self) -> Result<Self, FatalDivergence> {
        match self.outcome {
            TrainingOutcome::Diverged { recovered: false } => Err(FatalDivergence {
                final_epoch: self.final_epoch,
            }),
            _ => Ok(self),
        }
    }
}

/// Orchestrates a training run: forward/backward passes, scheduled evaluation,
/// early stopping, and divergence handling. Performs no I/O of its own — all
/// side effects (persistence, display) go through `callbacks`.
pub struct TrainingLoop {
    pub model: NeuralNetwork,
    pub callbacks: Callbacks,
    pub split: ModelSplit,
    pub config: TrainingConfig,
    pub early_stopping: Option<EarlyStopping>,
    pub epoch_start: usize,
}

impl TrainingLoop {
    /// Whether `epoch` should trigger an evaluation + checkpoint, given
    /// `eval_interval == 0` means "no checkpoints at all".
    pub fn is_checkpoint(eval_interval: usize, epoch: usize) -> bool {
        eval_interval != 0 && epoch.is_multiple_of(eval_interval)
    }

    pub fn run(mut self) -> IoResult<TrainingReport> {
        self.callbacks.on_train_start(&self.config)?;

        // Accuracy is strictly determined by the number of classes, itself encoded
        // in the output layer — derive it from the model rather than taking it as config.
        let evaluator = Evaluator::new(
            self.config.loss.clone(),
            accuracy_for(self.model.n_classes()),
        );

        let epochs = self.config.epochs;
        let eval_interval = self.config.eval_interval;

        if self.epoch_start == 0 && eval_interval != 0 {
            self.evaluate(&evaluator, 0)?;
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
                let loss = evaluator
                    .eval_predictions(preds.view(), validation.targets.view())
                    .loss;

                if es.check(loss, &self.model) {
                    let restored = es.best_model.is_some();
                    if let Some(best) = es.best_model.take() {
                        self.model = best;
                    }
                    outcome = TrainingOutcome::EarlyStopped { restored };
                    break;
                }
            }

            if Self::is_checkpoint(eval_interval, epoch) {
                final_evaluation = Some(self.evaluate(&evaluator, epoch)?);
            }
        }

        let final_evaluation = match final_evaluation {
            Some(eval) => Some(eval),
            // A diverged model with no recovered fallback is non-finite:
            // evaluating it would panic, so there is nothing to report.
            None if !self.model.is_finite() => None,
            None => Some(self.evaluate(&evaluator, final_epoch)?),
        };

        // `None` exactly when there is nothing safe to persist (fatal divergence).
        let final_model = self.model.is_finite().then_some(&self.model);
        self.callbacks.on_train_end(
            outcome,
            final_model,
            final_evaluation.as_ref(),
            final_epoch,
        )?;

        Ok(TrainingReport {
            outcome,
            model: self.model,
            final_evaluation,
            final_epoch,
        })
    }

    /// Computes an evaluation for the current model and reports it via
    /// [`Callbacks::on_evaluate`].
    fn evaluate(&mut self, evaluator: &Evaluator, epoch: usize) -> IoResult<EvaluationSet> {
        let eval = evaluator.eval_set(&self.model, &self.split);
        self.callbacks.on_evaluate(&self.model, &eval, epoch)?;
        Ok(eval)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::SIGMOID;
    use crate::data::ModelDataset;
    use crate::loss_functions::CROSS_ENTROPY_LOSS;
    use crate::model::NeuronLayer;
    use crate::optimizers::Adam;
    use crate::schedulers::ConstantScheduler;
    use crate::training::{GradientClipping, LearningRate};
    use ndarray::array;
    use std::cell::RefCell;
    use std::io::Error;
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
        last_model_was_some: Option<bool>,
    }

    struct CountingCallback(Rc<RefCell<Counts>>);

    impl TrainingCallback for CountingCallback {
        fn on_train_start(&mut self, _config: &TrainingConfig) -> IoResult<()> {
            self.0.borrow_mut().train_starts += 1;
            Ok(())
        }

        fn on_epoch_end(&mut self, _epoch: usize) -> IoResult<()> {
            self.0.borrow_mut().epoch_ends += 1;
            Ok(())
        }

        fn on_evaluate(
            &mut self,
            _model: &NeuralNetwork,
            _eval: &EvaluationSet,
            epoch: usize,
        ) -> IoResult<()> {
            self.0.borrow_mut().evaluated_epochs.push(epoch);
            Ok(())
        }

        fn on_train_end(
            &mut self,
            outcome: TrainingOutcome,
            model: Option<&NeuralNetwork>,
            _eval: Option<&EvaluationSet>,
            _epoch: usize,
        ) -> IoResult<()> {
            let mut counts = self.0.borrow_mut();
            counts.train_ends += 1;
            counts.last_outcome = Some(outcome);
            counts.last_model_was_some = Some(model.is_some());
            Ok(())
        }
    }

    /// A callback whose `on_evaluate` always fails, used to verify that
    /// `TrainingLoop::run` propagates errors from the orchestration loop.
    struct FailingOnEvaluate;

    impl TrainingCallback for FailingOnEvaluate {
        fn on_evaluate(
            &mut self,
            _model: &NeuralNetwork,
            _eval: &EvaluationSet,
            _epoch: usize,
        ) -> IoResult<()> {
            Err(Error::other("boom"))
        }
    }

    /// A callback whose `on_train_end` always fails, used to verify that
    /// `TrainingLoop::run` propagates errors raised at the end of training.
    struct FailingOnTrainEnd;

    impl TrainingCallback for FailingOnTrainEnd {
        fn on_train_end(
            &mut self,
            _outcome: TrainingOutcome,
            _model: Option<&NeuralNetwork>,
            _eval: Option<&EvaluationSet>,
            _epoch: usize,
        ) -> IoResult<()> {
            Err(Error::other("boom"))
        }
    }

    /// A callback whose `on_train_start` always fails, used to verify that
    /// `TrainingLoop::run` propagates errors raised before the training loop.
    struct FailingOnTrainStart;

    impl TrainingCallback for FailingOnTrainStart {
        fn on_train_start(&mut self, _config: &TrainingConfig) -> IoResult<()> {
            Err(Error::other("boom"))
        }
    }

    /// A callback whose `on_epoch_end` always fails, used to verify that
    /// `TrainingLoop::run` propagates errors raised inside the training loop.
    struct FailingOnEpochEnd;

    impl TrainingCallback for FailingOnEpochEnd {
        fn on_epoch_end(&mut self, _epoch: usize) -> IoResult<()> {
            Err(Error::other("boom"))
        }
    }

    /// A callback whose `on_evaluate` succeeds at epoch 0 but fails afterwards,
    /// so the failure surfaces from an in-loop checkpoint or the final fallback
    /// evaluation rather than from the epoch-0 pre-evaluation.
    struct FailingOnEvaluateAfterFirst;

    impl TrainingCallback for FailingOnEvaluateAfterFirst {
        fn on_evaluate(
            &mut self,
            _model: &NeuralNetwork,
            _eval: &EvaluationSet,
            epoch: usize,
        ) -> IoResult<()> {
            if epoch == 0 {
                Ok(())
            } else {
                Err(Error::other("boom"))
            }
        }
    }

    fn training_loop(
        config: TrainingConfig,
        with_validation: bool,
        early_stopping: Option<EarlyStopping>,
        counts: Rc<RefCell<Counts>>,
    ) -> TrainingLoop {
        TrainingLoop {
            model: sample_model(),
            callbacks: Callbacks::new(vec![Box::new(CountingCallback(counts))]),
            split: sample_split(with_validation),
            config,
            early_stopping,
            epoch_start: 0,
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
        let training_loop = training_loop(config, false, None, counts.clone());

        let report = training_loop.run().unwrap();

        assert_eq!(report.final_epoch, 7);
        let counts = counts.borrow();
        assert_eq!(counts.train_starts, 1);
        assert_eq!(counts.train_ends, 1);
        assert_eq!(counts.epoch_ends, 7);
        // epoch 0 + multiples of 3 (3, 6) + final epoch 7 (not a multiple of 3)
        assert_eq!(counts.evaluated_epochs, vec![0, 3, 6, 7]);
        assert_eq!(counts.last_model_was_some, Some(true));
    }

    #[test]
    fn eval_interval_zero_only_emits_the_final_evaluation() {
        let counts = Rc::new(RefCell::new(Counts::default()));
        let config = sample_config(3, 0, 0.01);
        let training_loop = training_loop(config, false, None, counts.clone());

        let report = training_loop.run().unwrap();

        assert!(report.final_evaluation.is_some());
        assert_eq!(counts.borrow().evaluated_epochs, vec![3]);
    }

    #[test]
    fn early_stopping_halts_training_and_restores_best_model() {
        let counts = Rc::new(RefCell::new(Counts::default()));
        let config = sample_config(50, 1, 5.0);
        let early_stopping = Some(EarlyStopping::new(1, true));
        let training_loop = training_loop(config, true, early_stopping, counts.clone());

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
        let training_loop = training_loop(config, false, None, counts.clone());

        let report = training_loop.run().unwrap();

        assert_eq!(
            report.outcome,
            TrainingOutcome::Diverged { recovered: false }
        );
        assert!(report.final_evaluation.is_none());
        let counts = counts.borrow();
        assert_eq!(
            counts.last_outcome,
            Some(TrainingOutcome::Diverged { recovered: false })
        );
        assert_eq!(counts.last_model_was_some, Some(false));
    }

    #[test]
    fn divergence_recovers_seeded_best_model_when_restore_enabled() {
        let counts = Rc::new(RefCell::new(Counts::default()));
        let config = sample_config(5, 1, 1e30);
        let early_stopping = Some(EarlyStopping::new(10, true));
        let training_loop = training_loop(config, true, early_stopping, counts.clone());

        let report = training_loop.run().unwrap();

        assert_eq!(
            report.outcome,
            TrainingOutcome::Diverged { recovered: true }
        );
        assert!(report.model.is_finite());
        assert!(report.final_evaluation.is_some());
    }

    #[test]
    fn callback_error_during_evaluation_is_propagated() {
        let config = sample_config(3, 1, 0.01);
        let training_loop = TrainingLoop {
            model: sample_model(),
            callbacks: Callbacks::new(vec![Box::new(FailingOnEvaluate)]),
            split: sample_split(false),
            config,
            early_stopping: None,
            epoch_start: 0,
        };

        assert!(training_loop.run().is_err());
    }

    #[test]
    fn callback_error_during_train_end_is_propagated() {
        let config = sample_config(3, 1, 0.01);
        let training_loop = TrainingLoop {
            model: sample_model(),
            callbacks: Callbacks::new(vec![Box::new(FailingOnTrainEnd)]),
            split: sample_split(false),
            config,
            early_stopping: None,
            epoch_start: 0,
        };

        assert!(training_loop.run().is_err());
    }

    #[test]
    fn final_evaluation_is_reused_when_last_epoch_is_a_checkpoint() {
        let counts = Rc::new(RefCell::new(Counts::default()));
        let config = sample_config(6, 3, 0.01);
        let report = training_loop(config, false, None, counts.clone())
            .run()
            .unwrap();

        assert!(report.final_evaluation.is_some());
        // [0, 3, 6] and NOT [0, 3, 6, 6]: proves the final checkpoint
        // evaluation is reused rather than recomputed.
        assert_eq!(counts.borrow().evaluated_epochs, vec![0, 3, 6]);
    }

    #[test]
    fn into_result_is_ok_for_non_fatal_outcomes() {
        let counts = Rc::new(RefCell::new(Counts::default()));
        let config = sample_config(7, 3, 0.01);
        let report = training_loop(config, false, None, counts).run().unwrap();

        assert_eq!(report.outcome, TrainingOutcome::Completed);
        assert!(report.into_result().is_ok());
    }

    #[test]
    fn into_result_is_err_for_unrecovered_divergence() {
        let counts = Rc::new(RefCell::new(Counts::default()));
        let config = sample_config(5, 1, 1e30);
        let report = training_loop(config, false, None, counts).run().unwrap();

        assert_eq!(
            report.outcome,
            TrainingOutcome::Diverged { recovered: false }
        );
        let final_epoch = report.final_epoch;
        let err = report
            .into_result()
            .err()
            .expect("expected a fatal divergence error");
        assert_eq!(err.final_epoch, final_epoch);
        assert_eq!(
            err.to_string(),
            format!("model diverged at epoch {final_epoch}: non-finite weights")
        );
    }

    #[test]
    fn callback_error_during_train_start_is_propagated() {
        let config = sample_config(3, 1, 0.01);
        let training_loop = TrainingLoop {
            model: sample_model(),
            callbacks: Callbacks::new(vec![Box::new(FailingOnTrainStart)]),
            split: sample_split(false),
            config,
            early_stopping: None,
            epoch_start: 0,
        };

        assert!(training_loop.run().is_err());
    }

    #[test]
    fn callback_error_during_epoch_end_is_propagated() {
        let config = sample_config(3, 1, 0.01);
        let training_loop = TrainingLoop {
            model: sample_model(),
            callbacks: Callbacks::new(vec![Box::new(FailingOnEpochEnd)]),
            split: sample_split(false),
            config,
            early_stopping: None,
            epoch_start: 0,
        };

        assert!(training_loop.run().is_err());
    }

    #[test]
    fn callback_error_during_in_loop_checkpoint_is_propagated() {
        // eval_interval == 1: epoch 0 evaluates fine, the epoch-1 checkpoint fails.
        let config = sample_config(3, 1, 0.01);
        let training_loop = TrainingLoop {
            model: sample_model(),
            callbacks: Callbacks::new(vec![Box::new(FailingOnEvaluateAfterFirst)]),
            split: sample_split(false),
            config,
            early_stopping: None,
            epoch_start: 0,
        };

        assert!(training_loop.run().is_err());
    }

    #[test]
    fn callback_error_during_final_fallback_evaluation_is_propagated() {
        // eval_interval == 0: no epoch-0 pre-eval and no in-loop checkpoints, so the
        // only evaluation is the final fallback — and that is where the failure surfaces.
        let config = sample_config(3, 0, 0.01);
        let training_loop = TrainingLoop {
            model: sample_model(),
            callbacks: Callbacks::new(vec![Box::new(FailingOnEvaluateAfterFirst)]),
            split: sample_split(false),
            config,
            early_stopping: None,
            epoch_start: 0,
        };

        assert!(training_loop.run().is_err());
    }

    #[test]
    fn early_stopping_halts_without_restoring_when_restore_disabled() {
        let counts = Rc::new(RefCell::new(Counts::default()));
        let config = sample_config(50, 1, 5.0);
        let early_stopping = Some(EarlyStopping::new(1, false));
        let report = training_loop(config, true, early_stopping, counts.clone())
            .run()
            .unwrap();

        assert_eq!(
            report.outcome,
            TrainingOutcome::EarlyStopped { restored: false }
        );
        assert!(report.final_epoch < 50);
        assert_eq!(
            counts.borrow().last_outcome,
            Some(TrainingOutcome::EarlyStopped { restored: false })
        );
    }
}
