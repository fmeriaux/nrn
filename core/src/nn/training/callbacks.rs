use super::outcome::TrainingOutcome;
use crate::data::ModelSplit;
use crate::evaluation::EvaluationSet;
use crate::model::NeuralNetwork;
use crate::optimizers::Optimizer;
use crate::schedulers::Scheduler;

/// Type-erased error a [`TrainerCallback`] hook may fail with.
pub type CallbackError = Box<dyn std::error::Error>;

/// The result type of every [`TrainerCallback`] hook.
pub type CallbackResult = Result<(), CallbackError>;

/// Observes training lifecycle events.
///
/// All methods have default no-op implementations — implement only what you need.
/// The training loop owns the checkpoint scheduling: it computes an [`EvaluationSet`]
/// at epoch 0, at each multiple of `checkpoint_interval`, and at the final epoch, then
/// dispatches it via [`on_checkpoint`](TrainerCallback::on_checkpoint).
pub trait TrainerCallback {
    /// Called from [`Trainer::restore`](crate::training::Trainer::restore) when
    /// resuming a run, after the trainer's epoch counter and any persisted
    /// optimizer/scheduler state have been restored. `optimizer`/`scheduler` are
    /// `Some` only when their state was actually restored, exposing the live
    /// instance (e.g. for narration).
    fn on_restore(
        &mut self,
        _epoch_start: usize,
        _optimizer: Option<&dyn Optimizer>,
        _scheduler: Option<&dyn Scheduler>,
    ) -> CallbackResult {
        Ok(())
    }

    /// Called once before training begins, with the `split` the run will train
    /// and evaluate on. Callbacks that need the run's configuration hold their
    /// own [`crate::training::HyperParameters`].
    fn on_train_start(&mut self, _split: &ModelSplit) -> CallbackResult {
        Ok(())
    }

    /// Called after each epoch. Cheap — no evaluation has been computed.
    fn on_epoch_end(&mut self, _epoch: usize) -> CallbackResult {
        Ok(())
    }

    /// Called when the loop has computed an [`EvaluationSet`] at `epoch`
    /// (epoch 0, a multiple of `checkpoint_interval`, or the final epoch).
    /// `optimizer` and `scheduler` give access to their internal state for
    /// checkpointing (e.g. Adam moments, scheduler step count).
    fn on_checkpoint(
        &mut self,
        _model: &NeuralNetwork,
        _optimizer: &dyn Optimizer,
        _scheduler: &dyn Scheduler,
        _eval: &EvaluationSet,
        _epoch: usize,
    ) -> CallbackResult {
        Ok(())
    }

    /// Called once when training ends. `model` and `eval` are the final model and
    /// evaluation, or `None` on a fatal divergence (nothing safe to persist or
    /// evaluate).
    fn on_train_end(
        &mut self,
        _outcome: TrainingOutcome,
        _model: Option<&NeuralNetwork>,
        _eval: Option<&EvaluationSet>,
        _epoch: usize,
    ) -> CallbackResult {
        Ok(())
    }
}

/// Sequential composite of callbacks: dispatches each hook to its children in
/// registration order, short-circuiting at the first `Err`.
pub struct Callbacks(Vec<Box<dyn TrainerCallback>>);

impl Callbacks {
    pub fn new(callbacks: Vec<Box<dyn TrainerCallback>>) -> Self {
        Self(callbacks)
    }

    /// Starts an empty composite, to be built up with [`with`](Self::with) and
    /// [`with_opt`](Self::with_opt).
    pub fn empty() -> Self {
        Self(Vec::new())
    }

    /// Appends a callback.
    pub fn with(mut self, callback: impl TrainerCallback + 'static) -> Self {
        self.0.push(Box::new(callback));
        self
    }

    /// Appends a callback if present, otherwise returns `self` unchanged.
    pub fn with_opt(self, callback: Option<impl TrainerCallback + 'static>) -> Self {
        match callback {
            Some(callback) => self.with(callback),
            None => self,
        }
    }

    /// Dispatches `hook` to each child in registration order, short-circuiting at
    /// the first `Err`.
    fn propagate(
        &mut self,
        mut hook: impl FnMut(&mut dyn TrainerCallback) -> CallbackResult,
    ) -> CallbackResult {
        self.0.iter_mut().try_for_each(|cb| hook(cb.as_mut()))
    }
}

impl TrainerCallback for Callbacks {
    fn on_restore(
        &mut self,
        epoch_start: usize,
        optimizer: Option<&dyn Optimizer>,
        scheduler: Option<&dyn Scheduler>,
    ) -> CallbackResult {
        self.propagate(|cb| cb.on_restore(epoch_start, optimizer, scheduler))
    }

    fn on_train_start(&mut self, split: &ModelSplit) -> CallbackResult {
        self.propagate(|cb| cb.on_train_start(split))
    }

    fn on_epoch_end(&mut self, epoch: usize) -> CallbackResult {
        self.propagate(|cb| cb.on_epoch_end(epoch))
    }

    fn on_checkpoint(
        &mut self,
        model: &NeuralNetwork,
        optimizer: &dyn Optimizer,
        scheduler: &dyn Scheduler,
        eval: &EvaluationSet,
        epoch: usize,
    ) -> CallbackResult {
        self.propagate(|cb| cb.on_checkpoint(model, optimizer, scheduler, eval, epoch))
    }

    fn on_train_end(
        &mut self,
        outcome: TrainingOutcome,
        model: Option<&NeuralNetwork>,
        eval: Option<&EvaluationSet>,
        epoch: usize,
    ) -> CallbackResult {
        self.propagate(|cb| cb.on_train_end(outcome, model, eval, epoch))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::RELU;
    use crate::data::ModelDataset;
    use crate::evaluation::Evaluation;
    use crate::model::NeuronLayerSpec;
    use crate::optimizers::Adam;
    use crate::schedulers::ConstantScheduler;
    use crate::weight_decay::WeightDecay;
    use ndarray::array;
    use std::cell::RefCell;
    use std::rc::Rc;

    struct DefaultCallback;

    impl TrainerCallback for DefaultCallback {}

    fn sample_model() -> NeuralNetwork {
        let specs = NeuronLayerSpec::network_for(vec![3], &*RELU, 2);
        NeuralNetwork::initialization(2, &specs, 0)
    }

    fn sample_split() -> ModelSplit {
        let dataset = || ModelDataset::new(array![[0.1, 0.9], [0.2, 0.8]], array![[1.0, 0.0]]);
        ModelSplit {
            train: dataset(),
            validation: None,
            test: dataset(),
        }
    }

    fn sample_eval() -> EvaluationSet {
        EvaluationSet {
            train: Evaluation {
                loss: 0.0,
                accuracy: 0.0,
            },
            validation: None,
            test: Evaluation {
                loss: 0.0,
                accuracy: 0.0,
            },
        }
    }

    #[test]
    fn default_callback_methods_are_noop() {
        let mut callback = DefaultCallback;
        let model = sample_model();
        let optimizer = Adam::with_defaults(0.01.try_into().unwrap(), WeightDecay::ZERO);
        let scheduler = ConstantScheduler::new(0.01.try_into().unwrap());

        assert!(
            callback
                .on_restore(0, Some(&optimizer), Some(&scheduler))
                .is_ok()
        );
        assert!(callback.on_train_start(&sample_split()).is_ok());
        assert!(callback.on_epoch_end(0).is_ok());
        assert!(
            callback
                .on_checkpoint(&model, &optimizer, &scheduler, &sample_eval(), 0)
                .is_ok()
        );
        assert!(
            callback
                .on_train_end(
                    TrainingOutcome::Completed,
                    Some(&model),
                    Some(&sample_eval()),
                    0
                )
                .is_ok()
        );
    }

    #[derive(Default)]
    struct Counts {
        epoch_ends: usize,
    }

    struct CountingCallback(Rc<RefCell<Counts>>);

    impl TrainerCallback for CountingCallback {
        fn on_epoch_end(&mut self, _epoch: usize) -> CallbackResult {
            self.0.borrow_mut().epoch_ends += 1;
            Ok(())
        }
    }

    struct FailingCallback;

    impl TrainerCallback for FailingCallback {
        fn on_epoch_end(&mut self, _epoch: usize) -> CallbackResult {
            Err("boom".into())
        }
    }

    #[test]
    fn short_circuits_on_first_error() {
        let counts = Rc::new(RefCell::new(Counts::default()));
        let mut callbacks = Callbacks::new(vec![
            Box::new(FailingCallback),
            Box::new(CountingCallback(counts.clone())),
        ]);

        let result = callbacks.on_epoch_end(1);

        assert!(result.is_err());
        assert_eq!(counts.borrow().epoch_ends, 0);
    }

    #[test]
    fn builder_appends_callbacks_and_skips_none() {
        let first = Rc::new(RefCell::new(Counts::default()));
        let second = Rc::new(RefCell::new(Counts::default()));

        let mut callbacks = Callbacks::empty()
            .with(CountingCallback(first.clone()))
            .with_opt(Some(CountingCallback(second.clone())))
            .with_opt(None::<CountingCallback>);

        callbacks.on_epoch_end(1).unwrap();

        assert_eq!(first.borrow().epoch_ends, 1);
        assert_eq!(second.borrow().epoch_ends, 1);
    }

    #[test]
    fn dispatches_each_hook_to_all_children() {
        let first = Rc::new(RefCell::new(Counts::default()));
        let second = Rc::new(RefCell::new(Counts::default()));
        let mut callbacks = Callbacks::new(vec![
            Box::new(CountingCallback(first.clone())),
            Box::new(CountingCallback(second.clone())),
        ]);

        callbacks.on_epoch_end(1).unwrap();
        callbacks.on_epoch_end(2).unwrap();

        assert_eq!(first.borrow().epoch_ends, 2);
        assert_eq!(second.borrow().epoch_ends, 2);
    }
}
