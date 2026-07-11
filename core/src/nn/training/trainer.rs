use super::backprop::MiniBatch;
use super::callbacks::{CallbackError, CallbackResult, Callbacks, TrainerCallback};
use super::early_stopping::{EarlyStopping, EarlyStoppingConfig};
use super::evaluator::Evaluator;
use super::outcome::TrainingOutcome;
use crate::accuracies::Accuracy;
use crate::data::ModelSplit;
use crate::evaluation::EvaluationSet;
use crate::gradients::GradientClipping;
use crate::loss_functions::LossFunction;
use crate::model::NeuralNetwork;
use crate::optimizers::{Optimizer, OptimizerState};
use crate::schedulers::{Scheduler, SchedulerState};
use std::sync::Arc;

/// Panic message for the per-epoch shape invariant that
/// [`HyperParameters::build`](super::HyperParameters::build) establishes: the model and
/// dataset were checked to have compatible feature shapes.
const COMPATIBLE_SHAPES: &str = "model and dataset shapes were matched in HyperParameters::build";

/// The result of a completed [`Trainer::train`].
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
///
/// Built from a [`crate::training::HyperParameters`] specification via
/// [`HyperParameters::build`](crate::training::HyperParameters::build), which is
/// the only way to populate its instantiated optimizer/scheduler/loss.
pub struct Trainer {
    pub(super) model: NeuralNetwork,
    pub(super) callbacks: Callbacks,
    pub(super) split: ModelSplit,
    pub(super) loss: Arc<dyn LossFunction>,
    pub(super) accuracy: Arc<dyn Accuracy>,
    pub(super) optimizer: Box<dyn Optimizer>,
    pub(super) scheduler: Box<dyn Scheduler>,
    pub(super) clipping: GradientClipping,
    pub(super) batch_size: Option<usize>,
    pub(super) epochs: usize,
    pub(super) checkpoint_interval: usize,
    pub(super) early_stopping: Option<EarlyStoppingConfig>,
    pub(super) epoch_start: usize,
    pub(super) seed: u64,
}

impl Trainer {
    /// Restores the trainer to a checkpointed state before training: resumes the
    /// epoch count from `epoch_start` and reinstates any persisted optimizer and
    /// scheduler state, then notifies callbacks via
    /// [`TrainerCallback::on_restore`]. Used when resuming a run; a fresh run
    /// keeps the default `epoch_start` of 0 and needs no restore.
    /// # Errors
    /// Returns an error if the persisted optimizer state is incompatible with
    /// this run's optimizer, or if a callback fails.
    pub fn restore(
        &mut self,
        epoch_start: usize,
        optimizer_state: Option<OptimizerState>,
        scheduler_state: Option<SchedulerState>,
    ) -> CallbackResult {
        self.epoch_start = epoch_start;
        if let Some(state) = &scheduler_state {
            self.scheduler.restore(state);
        }
        if let Some(state) = &optimizer_state {
            self.optimizer.restore(state)?;
        }
        self.callbacks.on_restore(
            epoch_start,
            optimizer_state.is_some().then(|| self.optimizer.as_ref()),
            scheduler_state.is_some().then(|| self.scheduler.as_ref()),
        )
    }

    pub fn train(mut self) -> Result<TrainingReport, CallbackError> {
        self.callbacks.on_train_start(&self.split)?;

        let evaluator = Evaluator::new(self.loss.clone(), self.accuracy.clone());

        if self.epoch_start == 0 && self.checkpoint_interval != 0 {
            self.checkpoint(&evaluator, 0)?;
        }

        let mut early_stopping = self
            .split
            .validation
            .is_some()
            .then(|| {
                self.early_stopping
                    .clone()
                    .map(|config| EarlyStopping::new(config, &self.model))
            })
            .flatten();

        let mut outcome = TrainingOutcome::Completed;
        let mut final_epoch = self.epoch_start;
        let mut final_evaluation = None;

        for epoch in (self.epoch_start + 1)..=(self.epoch_start + self.epochs) {
            self.model
                .train(
                    &self.split.train,
                    &self.loss,
                    self.optimizer.as_mut(),
                    self.scheduler.as_mut(),
                    &self.clipping,
                    self.batch_size
                        .map(|size| MiniBatch::new(size, self.seed, epoch)),
                )
                .expect(COMPATIBLE_SHAPES);

            final_epoch = epoch;
            final_evaluation = None;

            if !self.model.is_finite() {
                let recovered = early_stopping.as_mut().and_then(|es| es.best_model.take());
                outcome = TrainingOutcome::Diverged {
                    recovered: recovered.is_some(),
                };
                if let Some(best) = recovered {
                    self.model = best;
                }
                break;
            }

            self.callbacks.on_epoch_end(epoch)?;

            if let Some(ref mut es) = early_stopping
                && let Some(validation) = &self.split.validation
                && es
                    .check(validation, &self.model, &evaluator)
                    .expect(COMPATIBLE_SHAPES)
            {
                outcome = TrainingOutcome::EarlyStopped {
                    restored: es.best_model.is_some(),
                };
                if let Some(best) = es.best_model.take() {
                    self.model = best;
                }
                break;
            }

            if Self::is_checkpoint(self.checkpoint_interval, epoch) {
                final_evaluation = self.checkpoint(&evaluator, epoch)?;
            }
        }

        let final_evaluation = match final_evaluation {
            Some(eval) => Some(eval),
            None => self.checkpoint(&evaluator, final_epoch)?,
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
    /// [`Callbacks::on_checkpoint`]. Returns `None` without evaluating if the model
    /// has diverged (non-finite weights), since its evaluation would be meaningless.
    fn checkpoint(
        &mut self,
        evaluator: &Evaluator,
        epoch: usize,
    ) -> Result<Option<EvaluationSet>, CallbackError> {
        if !self.model.is_finite() {
            return Ok(None);
        }
        let eval = evaluator
            .eval_set(&self.model, &self.split)
            .expect(COMPATIBLE_SHAPES);
        self.callbacks.on_checkpoint(
            &self.model,
            self.optimizer.as_ref(),
            self.scheduler.as_ref(),
            &eval,
            epoch,
        )?;
        Ok(Some(eval))
    }

    /// Whether `epoch` should trigger an evaluation + checkpoint, given
    /// `checkpoint_interval == 0` means "no checkpoints at all".
    fn is_checkpoint(checkpoint_interval: usize, epoch: usize) -> bool {
        checkpoint_interval != 0 && epoch.is_multiple_of(checkpoint_interval)
    }
}

#[cfg(test)]
mod tests {
    use super::super::early_stopping::EarlyStoppingConfig;
    use super::super::hyperparams::{
        HyperParameters, LossConfig, OptimizerConfig, SchedulerConfig,
    };
    use super::*;
    use crate::activations::SIGMOID;
    use crate::data::Dataset;
    use crate::layers::Dense;
    use crate::objectives::Objective;
    use crate::optimizers::Optimizer;
    use crate::schedulers::Scheduler;
    use crate::training::GradientClipping;
    use ndarray::array;
    use std::cell::RefCell;
    use std::rc::Rc;

    /// A 10-sample, 2-feature dataset — large enough that the ratio split in
    /// [`HyperParameters::build`] yields non-degenerate train/validation/test sets
    /// (e.g. `val_ratio = test_ratio = 0.1` gives 8/1/1).
    fn sample_dataset() -> Dataset {
        Dataset::new(
            array![
                [0.1, 0.2],
                [0.9, 0.8],
                [0.2, 0.3],
                [0.8, 0.7],
                [0.15, 0.25],
                [0.85, 0.75],
                [0.25, 0.35],
                [0.75, 0.65],
                [0.3, 0.4],
                [0.7, 0.6]
            ],
            array![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            None,
        )
        .unwrap()
    }

    /// A fixed (non-random) 2-input -> 1-output sigmoid network, so loss
    /// sequences are fully deterministic across runs.
    fn sample_model() -> NeuralNetwork {
        NeuralNetwork::single(Dense::new(
            array![[0.1, -0.2]],
            array![0.05],
            SIGMOID.clone(),
        ))
    }

    fn sample_hyperparameters(
        epochs: usize,
        checkpoint_interval: usize,
        lr: f32,
        early_stopping: Option<EarlyStoppingConfig>,
        val_ratio: f32,
    ) -> HyperParameters {
        HyperParameters::from_values(
            epochs,
            checkpoint_interval,
            None,
            lr,
            0.0,
            OptimizerConfig::Adam,
            SchedulerConfig::Constant,
            GradientClipping::None,
            LossConfig::CrossEntropy,
            early_stopping,
            val_ratio,
            0.1,
            0,
            None,
        )
        .unwrap()
    }

    #[derive(Default)]
    struct Counts {
        train_starts: usize,
        epoch_ends: usize,
        evaluated_epochs: Vec<usize>,
        train_ends: usize,
        last_outcome: Option<TrainingOutcome>,
        last_model_was_some: Option<bool>,
        /// `(epoch_start, optimizer_name, scheduler_name)` from the last `on_restore`.
        restored: Option<(usize, Option<String>, Option<String>)>,
    }

    struct CountingCallback(Rc<RefCell<Counts>>);

    impl TrainerCallback for CountingCallback {
        fn on_restore(
            &mut self,
            epoch_start: usize,
            optimizer: Option<&dyn Optimizer>,
            scheduler: Option<&dyn Scheduler>,
        ) -> CallbackResult {
            self.0.borrow_mut().restored = Some((
                epoch_start,
                optimizer.map(|o| o.name().to_string()),
                scheduler.map(|s| s.name().to_string()),
            ));
            Ok(())
        }

        fn on_train_start(&mut self, _split: &ModelSplit) -> CallbackResult {
            self.0.borrow_mut().train_starts += 1;
            Ok(())
        }

        fn on_epoch_end(&mut self, _epoch: usize) -> CallbackResult {
            self.0.borrow_mut().epoch_ends += 1;
            Ok(())
        }

        fn on_checkpoint(
            &mut self,
            _model: &NeuralNetwork,
            _optimizer: &dyn Optimizer,
            _scheduler: &dyn Scheduler,
            _eval: &EvaluationSet,
            epoch: usize,
        ) -> CallbackResult {
            self.0.borrow_mut().evaluated_epochs.push(epoch);
            Ok(())
        }

        fn on_train_end(
            &mut self,
            outcome: TrainingOutcome,
            model: Option<&NeuralNetwork>,
            _eval: Option<&EvaluationSet>,
            _epoch: usize,
        ) -> CallbackResult {
            let mut counts = self.0.borrow_mut();
            counts.train_ends += 1;
            counts.last_outcome = Some(outcome);
            counts.last_model_was_some = Some(model.is_some());
            Ok(())
        }
    }

    /// A callback whose `on_evaluate` always fails, used to verify that
    /// `Trainer::train` propagates errors from the orchestration loop.
    struct FailingOnEvaluate;

    impl TrainerCallback for FailingOnEvaluate {
        fn on_checkpoint(
            &mut self,
            _model: &NeuralNetwork,
            _optimizer: &dyn Optimizer,
            _scheduler: &dyn Scheduler,
            _eval: &EvaluationSet,
            _epoch: usize,
        ) -> CallbackResult {
            Err("boom".into())
        }
    }

    /// A callback whose `on_train_end` always fails, used to verify that
    /// `Trainer::train` propagates errors raised at the end of training.
    struct FailingOnTrainEnd;

    impl TrainerCallback for FailingOnTrainEnd {
        fn on_train_end(
            &mut self,
            _outcome: TrainingOutcome,
            _model: Option<&NeuralNetwork>,
            _eval: Option<&EvaluationSet>,
            _epoch: usize,
        ) -> CallbackResult {
            Err("boom".into())
        }
    }

    /// A callback whose `on_train_start` always fails, used to verify that
    /// `Trainer::train` propagates errors raised before the training loop.
    struct FailingOnTrainStart;

    impl TrainerCallback for FailingOnTrainStart {
        fn on_train_start(&mut self, _split: &ModelSplit) -> CallbackResult {
            Err("boom".into())
        }
    }

    /// A callback whose `on_epoch_end` always fails, used to verify that
    /// `Trainer::train` propagates errors raised inside the training loop.
    struct FailingOnEpochEnd;

    impl TrainerCallback for FailingOnEpochEnd {
        fn on_epoch_end(&mut self, _epoch: usize) -> CallbackResult {
            Err("boom".into())
        }
    }

    /// A callback whose `on_evaluate` succeeds at epoch 0 but fails afterwards,
    /// so the failure surfaces from an in-loop checkpoint or the final fallback
    /// evaluation rather than from the epoch-0 pre-evaluation.
    struct FailingOnEvaluateAfterFirst;

    impl TrainerCallback for FailingOnEvaluateAfterFirst {
        fn on_checkpoint(
            &mut self,
            _model: &NeuralNetwork,
            _optimizer: &dyn Optimizer,
            _scheduler: &dyn Scheduler,
            _eval: &EvaluationSet,
            epoch: usize,
        ) -> CallbackResult {
            if epoch == 0 {
                Ok(())
            } else {
                Err("boom".into())
            }
        }
    }

    fn trainer(hyperparameters: HyperParameters, counts: Rc<RefCell<Counts>>) -> Trainer {
        let data = hyperparameters.prepare(sample_dataset(), None).unwrap();
        hyperparameters
            .build(
                sample_model(),
                Objective::Binary,
                data,
                Callbacks::new(vec![Box::new(CountingCallback(counts))]),
            )
            .unwrap()
    }

    /// Builds a [`Trainer`] for a callback whose only registered hook is the
    /// given failing callback, to assert error propagation through `train`.
    fn failing_trainer(
        epochs: usize,
        checkpoint_interval: usize,
        callback: impl TrainerCallback + 'static,
    ) -> Trainer {
        let hyperparameters = sample_hyperparameters(epochs, checkpoint_interval, 0.01, None, 0.0);
        let data = hyperparameters.prepare(sample_dataset(), None).unwrap();
        hyperparameters
            .build(
                sample_model(),
                Objective::Binary,
                data,
                Callbacks::new(vec![Box::new(callback)]),
            )
            .unwrap()
    }

    #[test]
    fn is_checkpoint_cases() {
        assert!(Trainer::is_checkpoint(3, 0));
        assert!(Trainer::is_checkpoint(3, 3));
        assert!(!Trainer::is_checkpoint(3, 1));
        assert!(!Trainer::is_checkpoint(3, 4));
        assert!(!Trainer::is_checkpoint(0, 0));
        assert!(!Trainer::is_checkpoint(0, 3));
    }

    #[test]
    fn evaluation_schedule_includes_epoch_zero_multiples_and_final_epoch() {
        let counts = Rc::new(RefCell::new(Counts::default()));
        let hyperparameters = sample_hyperparameters(7, 3, 0.01, None, 0.0);

        let report = trainer(hyperparameters, counts.clone()).train().unwrap();

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
        let hyperparameters = sample_hyperparameters(3, 0, 0.01, None, 0.0);

        let report = trainer(hyperparameters, counts.clone()).train().unwrap();

        assert!(report.final_evaluation.is_some());
        assert_eq!(counts.borrow().evaluated_epochs, vec![3]);
    }

    #[test]
    fn early_stopping_halts_training_and_restores_best_model() {
        let counts = Rc::new(RefCell::new(Counts::default()));
        let early_stopping = Some(EarlyStoppingConfig::new(1, true).unwrap());
        let hyperparameters = sample_hyperparameters(50, 1, 5.0, early_stopping, 0.1);

        let report = trainer(hyperparameters, counts.clone()).train().unwrap();

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
        let hyperparameters = sample_hyperparameters(5, 1, f32::MAX, None, 0.0);

        let report = trainer(hyperparameters, counts.clone()).train().unwrap();

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
        let early_stopping = Some(EarlyStoppingConfig::new(10, true).unwrap());
        let hyperparameters = sample_hyperparameters(5, 1, f32::MAX, early_stopping, 0.1);

        let report = trainer(hyperparameters, counts.clone()).train().unwrap();

        assert_eq!(
            report.outcome,
            TrainingOutcome::Diverged { recovered: true }
        );
        assert!(report.model.is_finite());
        assert!(report.final_evaluation.is_some());
    }

    #[test]
    fn callback_error_during_evaluation_is_propagated() {
        assert!(failing_trainer(3, 1, FailingOnEvaluate).train().is_err());
    }

    #[test]
    fn callback_error_during_train_end_is_propagated() {
        assert!(failing_trainer(3, 1, FailingOnTrainEnd).train().is_err());
    }

    #[test]
    fn final_evaluation_is_reused_when_last_epoch_is_a_checkpoint() {
        let counts = Rc::new(RefCell::new(Counts::default()));
        let hyperparameters = sample_hyperparameters(6, 3, 0.01, None, 0.0);
        let report = trainer(hyperparameters, counts.clone()).train().unwrap();

        assert!(report.final_evaluation.is_some());
        // [0, 3, 6] and NOT [0, 3, 6, 6]: proves the final checkpoint
        // evaluation is reused rather than recomputed.
        assert_eq!(counts.borrow().evaluated_epochs, vec![0, 3, 6]);
    }

    #[test]
    fn into_result_is_ok_for_non_fatal_outcomes() {
        let counts = Rc::new(RefCell::new(Counts::default()));
        let hyperparameters = sample_hyperparameters(7, 3, 0.01, None, 0.0);
        let report = trainer(hyperparameters, counts).train().unwrap();

        assert_eq!(report.outcome, TrainingOutcome::Completed);
        assert!(report.into_result().is_ok());
    }

    #[test]
    fn into_result_is_err_for_unrecovered_divergence() {
        let counts = Rc::new(RefCell::new(Counts::default()));
        let hyperparameters = sample_hyperparameters(5, 1, f32::MAX, None, 0.0);
        let report = trainer(hyperparameters, counts).train().unwrap();

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
        assert!(failing_trainer(3, 1, FailingOnTrainStart).train().is_err());
    }

    #[test]
    fn callback_error_during_epoch_end_is_propagated() {
        assert!(failing_trainer(3, 1, FailingOnEpochEnd).train().is_err());
    }

    #[test]
    fn callback_error_during_in_loop_checkpoint_is_propagated() {
        // eval_interval == 1: epoch 0 evaluates fine, the epoch-1 checkpoint fails.
        assert!(
            failing_trainer(3, 1, FailingOnEvaluateAfterFirst)
                .train()
                .is_err()
        );
    }

    #[test]
    fn callback_error_during_final_fallback_evaluation_is_propagated() {
        // eval_interval == 0: no epoch-0 pre-eval and no in-loop checkpoints, so the
        // only evaluation is the final fallback — and that is where the failure surfaces.
        assert!(
            failing_trainer(3, 0, FailingOnEvaluateAfterFirst)
                .train()
                .is_err()
        );
    }

    #[test]
    fn early_stopping_halts_without_restoring_when_restore_disabled() {
        let counts = Rc::new(RefCell::new(Counts::default()));
        let early_stopping = Some(EarlyStoppingConfig::new(1, false).unwrap());
        let hyperparameters = sample_hyperparameters(50, 1, 5.0, early_stopping, 0.1);
        let report = trainer(hyperparameters, counts.clone()).train().unwrap();

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

    #[test]
    fn restore_reinstates_state_and_resumes_from_epoch_start() {
        use crate::gradients::LayerGradients;
        use crate::optimizers::Adam;
        use crate::schedulers::CosineAnnealing;
        use crate::weight_decay::WeightDecay;

        // Stateful optimizer/scheduler snapshots, as a checkpoint would hold them.
        let mut adam = Adam::with_defaults(0.01.try_into().unwrap(), WeightDecay::ZERO);
        let mut layer = Dense::new(array![[0.1, -0.2]], array![0.05], SIGMOID.clone());
        adam.update_layer(
            0,
            &mut layer,
            &LayerGradients(vec![array![[0.1, 0.1]].into_dyn(), array![0.1].into_dyn()]),
        );
        adam.step();
        let optimizer_state = adam.to_state();

        let mut cosine = CosineAnnealing::from_values(0.001, 0.01, 5, false, 1).unwrap();
        cosine.step();
        let scheduler_state = cosine.to_state();

        let hyperparameters = HyperParameters::from_values(
            3,
            1,
            None,
            0.01,
            0.0,
            OptimizerConfig::Adam,
            SchedulerConfig::Cosine {
                lr_min: 0.001,
                steps: 5,
                warm_restarts: false,
                cycle_multiplier: 1,
            },
            GradientClipping::None,
            LossConfig::CrossEntropy,
            None,
            0.1,
            0.1,
            0,
            None,
        )
        .unwrap();

        let counts = Rc::new(RefCell::new(Counts::default()));
        let data = hyperparameters.prepare(sample_dataset(), None).unwrap();
        let mut trainer = hyperparameters
            .build(
                sample_model(),
                Objective::Binary,
                data,
                Callbacks::new(vec![Box::new(CountingCallback(counts.clone()))]),
            )
            .unwrap();

        trainer
            .restore(10, optimizer_state, scheduler_state)
            .unwrap();
        let report = trainer.train().unwrap();

        // Resumed counting from epoch 10, then trained 3 more epochs.
        assert_eq!(report.final_epoch, 13);
        assert_eq!(
            counts.borrow().restored,
            Some((
                10,
                Some("Adam".to_string()),
                Some("Cosine Annealing".to_string())
            ))
        );
    }

    #[test]
    fn restore_accepts_states_for_stateless_optimizer_and_scheduler() {
        use crate::optimizers::OptimizerState;
        use std::collections::HashMap;

        let hyperparameters = HyperParameters::from_values(
            2,
            0,
            None,
            0.01,
            0.0,
            OptimizerConfig::Sgd,
            SchedulerConfig::Constant,
            GradientClipping::None,
            LossConfig::CrossEntropy,
            None,
            0.1,
            0.1,
            0,
            None,
        )
        .unwrap();

        let data = hyperparameters.prepare(sample_dataset(), None).unwrap();
        let mut trainer = hyperparameters
            .build(sample_model(), Objective::Binary, data, Callbacks::empty())
            .unwrap();

        // Stateless SGD / constant scheduler ignore the provided state (default no-ops).
        let optimizer_state = Some(OptimizerState {
            tensors: Vec::new(),
            metadata: HashMap::new(),
        });
        let scheduler_state = Some(SchedulerState { current_step: 7 });

        trainer
            .restore(5, optimizer_state, scheduler_state)
            .unwrap();

        assert_eq!(trainer.train().unwrap().final_epoch, 7);
    }

    #[test]
    fn restore_without_optimizer_state_only_resumes_scheduler() {
        let hyperparameters = HyperParameters::from_values(
            1,
            0,
            None,
            0.01,
            0.0,
            OptimizerConfig::Adam,
            SchedulerConfig::Cosine {
                lr_min: 0.001,
                steps: 5,
                warm_restarts: false,
                cycle_multiplier: 1,
            },
            GradientClipping::None,
            LossConfig::CrossEntropy,
            None,
            0.1,
            0.1,
            0,
            None,
        )
        .unwrap();

        let data = hyperparameters.prepare(sample_dataset(), None).unwrap();
        let mut trainer = hyperparameters
            .build(sample_model(), Objective::Binary, data, Callbacks::empty())
            .unwrap();

        trainer
            .restore(4, None, Some(SchedulerState { current_step: 2 }))
            .unwrap();

        assert_eq!(trainer.train().unwrap().final_epoch, 5);
    }

    #[test]
    fn restore_surfaces_optimizer_state_errors() {
        use crate::optimizers::OptimizerState;
        use std::collections::HashMap;

        // Adam optimizer (from `sample_hyperparameters`).
        let hyperparameters = sample_hyperparameters(1, 0, 0.01, None, 0.0);
        let data = hyperparameters.prepare(sample_dataset(), None).unwrap();
        let mut trainer = hyperparameters
            .build(sample_model(), Objective::Binary, data, Callbacks::empty())
            .unwrap();

        // Missing `time_step` metadata makes Adam's restore fail; the trainer
        // surfaces it as a callback error.
        let bad_state = Some(OptimizerState {
            tensors: Vec::new(),
            metadata: HashMap::new(),
        });

        assert!(trainer.restore(1, bad_state, None).is_err());
    }
}
