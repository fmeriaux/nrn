use super::config::TrainingConfig;
use super::outcome::TrainingOutcome;
use crate::evaluation::EvaluationSet;
use crate::model::NeuralNetwork;
use std::io::Result;

/// Observes training lifecycle events.
///
/// All methods have default no-op implementations — implement only what you need.
/// The training loop owns the checkpoint scheduling: it computes an [`EvaluationSet`]
/// at epoch 0, at each multiple of `eval_interval`, and at the final epoch, then
/// dispatches it via [`on_evaluate`](TrainingCallback::on_evaluate).
pub trait TrainingCallback {
    /// Called once before training begins, with the run configuration.
    fn on_train_start(&mut self, _config: &TrainingConfig) -> Result<()> {
        Ok(())
    }

    /// Called after each epoch. Cheap — no evaluation has been computed.
    fn on_epoch_end(&mut self, _epoch: usize) -> Result<()> {
        Ok(())
    }

    /// Called when the loop has computed an [`EvaluationSet`] at `epoch`
    /// (epoch 0, a multiple of `eval_interval`, or the final epoch).
    fn on_evaluate(
        &mut self,
        _model: &NeuralNetwork,
        _eval: &EvaluationSet,
        _epoch: usize,
    ) -> Result<()> {
        Ok(())
    }

    /// Called once when training ends. `model` and `eval` are the final model and
    /// evaluation, or `None` on a fatal divergence (nothing safe to persist or evaluate).
    fn on_train_end(
        &mut self,
        _outcome: TrainingOutcome,
        _model: Option<&NeuralNetwork>,
        _eval: Option<&EvaluationSet>,
        _epoch: usize,
    ) -> Result<()> {
        Ok(())
    }
}

/// Sequential composite of callbacks: dispatches each hook to its children in
/// registration order, short-circuiting at the first `Err`.
pub struct Callbacks(Vec<Box<dyn TrainingCallback>>);

impl Callbacks {
    pub fn new(callbacks: Vec<Box<dyn TrainingCallback>>) -> Self {
        Self(callbacks)
    }

    /// Starts an empty composite, to be built up with [`with`](Self::with) and
    /// [`with_opt`](Self::with_opt).
    pub fn empty() -> Self {
        Self(Vec::new())
    }

    /// Appends a callback.
    pub fn with(mut self, callback: impl TrainingCallback + 'static) -> Self {
        self.0.push(Box::new(callback));
        self
    }

    /// Appends a callback if present, otherwise returns `self` unchanged.
    pub fn with_opt(self, callback: Option<impl TrainingCallback + 'static>) -> Self {
        match callback {
            Some(callback) => self.with(callback),
            None => self,
        }
    }
}

impl TrainingCallback for Callbacks {
    fn on_train_start(&mut self, config: &TrainingConfig) -> Result<()> {
        self.0
            .iter_mut()
            .try_for_each(|cb| cb.on_train_start(config))
    }

    fn on_epoch_end(&mut self, epoch: usize) -> Result<()> {
        self.0.iter_mut().try_for_each(|cb| cb.on_epoch_end(epoch))
    }

    fn on_evaluate(
        &mut self,
        model: &NeuralNetwork,
        eval: &EvaluationSet,
        epoch: usize,
    ) -> Result<()> {
        self.0
            .iter_mut()
            .try_for_each(|cb| cb.on_evaluate(model, eval, epoch))
    }

    fn on_train_end(
        &mut self,
        outcome: TrainingOutcome,
        model: Option<&NeuralNetwork>,
        eval: Option<&EvaluationSet>,
        epoch: usize,
    ) -> Result<()> {
        self.0
            .iter_mut()
            .try_for_each(|cb| cb.on_train_end(outcome, model, eval, epoch))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::RELU;
    use crate::evaluation::Evaluation;
    use crate::loss_functions::CROSS_ENTROPY_LOSS;
    use crate::model::NeuronLayerSpec;
    use crate::optimizers::Adam;
    use crate::schedulers::ConstantScheduler;
    use crate::training::{GradientClipping, LearningRate};
    use std::cell::RefCell;
    use std::io::Error;
    use std::rc::Rc;

    struct DefaultCallback;

    impl TrainingCallback for DefaultCallback {}

    fn sample_config() -> TrainingConfig {
        TrainingConfig {
            epochs: 1,
            eval_interval: 1,
            batch_size: None,
            loss: CROSS_ENTROPY_LOSS.clone(),
            optimizer: Box::new(Adam::with_defaults(LearningRate::new(0.01))),
            scheduler: Box::new(ConstantScheduler::new(LearningRate::new(0.01))),
            clipping: GradientClipping::None,
        }
    }

    fn sample_model() -> NeuralNetwork {
        let specs = NeuronLayerSpec::network_for(vec![3], &*RELU, 2);
        NeuralNetwork::initialization(2, &specs)
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
        let config = sample_config();
        let model = sample_model();

        assert!(callback.on_train_start(&config).is_ok());
        assert!(callback.on_epoch_end(0).is_ok());
        assert!(callback.on_evaluate(&model, &sample_eval(), 0).is_ok());
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

    impl TrainingCallback for CountingCallback {
        fn on_epoch_end(&mut self, _epoch: usize) -> Result<()> {
            self.0.borrow_mut().epoch_ends += 1;
            Ok(())
        }
    }

    struct FailingCallback;

    impl TrainingCallback for FailingCallback {
        fn on_epoch_end(&mut self, _epoch: usize) -> Result<()> {
            Err(Error::other("boom"))
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
