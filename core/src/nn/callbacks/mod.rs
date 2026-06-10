use crate::evaluation::EvaluationSet;
use crate::model::NeuralNetwork;
use crate::training::TrainingConfig;
use std::io::Result;

/// How a training run ended.
///
/// `Copy`. The bools carry the runtime facts a reporter needs to narrate the outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingOutcome {
    Completed,
    /// `restored` is `true` when the best model (by validation loss) was restored.
    EarlyStopped {
        restored: bool,
    },
    /// `recovered` is `true` when the best model was restored after a NaN/Inf divergence.
    Diverged {
        recovered: bool,
    },
}

/// Observes training lifecycle events.
///
/// All methods have default no-op implementations — implement only what you need.
/// The training loop owns the checkpoint scheduling: it computes an [`EvaluationSet`]
/// at epoch 0, at each multiple of `eval_interval`, and at the final epoch, then
/// dispatches it via [`on_evaluate`](TrainingCallback::on_evaluate).
pub trait TrainingCallback {
    /// Called once before training begins, with a borrowed view of the run configuration.
    fn on_train_start(&mut self, _config: &TrainingConfig<'_>) -> Result<()> {
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

    /// Called once when training ends. `eval` is the final evaluation, or `None`
    /// on a fatal divergence (nothing to evaluate).
    fn on_train_end(
        &mut self,
        _outcome: TrainingOutcome,
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
}

impl TrainingCallback for Callbacks {
    fn on_train_start(&mut self, config: &TrainingConfig<'_>) -> Result<()> {
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
        eval: Option<&EvaluationSet>,
        epoch: usize,
    ) -> Result<()> {
        self.0
            .iter_mut()
            .try_for_each(|cb| cb.on_train_end(outcome, eval, epoch))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::RELU;
    use crate::evaluation::Evaluation;
    use crate::model::NeuronLayerSpec;
    use std::cell::RefCell;
    use std::io::Error;
    use std::rc::Rc;

    #[derive(Default, Clone)]
    struct Counts {
        train_starts: usize,
        epoch_ends: usize,
        evaluates: usize,
        train_ends: usize,
    }

    struct CountingCallback(Rc<RefCell<Counts>>);

    impl TrainingCallback for CountingCallback {
        fn on_train_start(&mut self, _config: &TrainingConfig<'_>) -> Result<()> {
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
            _epoch: usize,
        ) -> Result<()> {
            self.0.borrow_mut().evaluates += 1;
            Ok(())
        }

        fn on_train_end(
            &mut self,
            _outcome: TrainingOutcome,
            _eval: Option<&EvaluationSet>,
            _epoch: usize,
        ) -> Result<()> {
            self.0.borrow_mut().train_ends += 1;
            Ok(())
        }
    }

    struct DefaultCallback;

    impl TrainingCallback for DefaultCallback {}

    fn sample_config<'a>(
        loss: &'a dyn crate::loss_functions::LossFunction,
        optimizer: &'a dyn crate::optimizers::Optimizer,
        scheduler: &'a dyn crate::schedulers::Scheduler,
        clipping: &'a crate::training::GradientClipping,
    ) -> TrainingConfig<'a> {
        TrainingConfig {
            epochs: 1,
            eval_interval: 1,
            batch_size: None,
            loss,
            optimizer,
            scheduler,
            clipping,
        }
    }

    #[test]
    fn default_callback_methods_are_noop() {
        use crate::loss_functions::CROSS_ENTROPY_LOSS;
        use crate::optimizers::Adam;
        use crate::schedulers::ConstantScheduler;
        use crate::training::{GradientClipping, LearningRate};

        let mut callback = DefaultCallback;
        let optimizer = Adam::with_defaults(LearningRate::new(0.01));
        let scheduler = ConstantScheduler::new(LearningRate::new(0.01));
        let clipping = GradientClipping::None;
        let config = sample_config(&**CROSS_ENTROPY_LOSS, &optimizer, &scheduler, &clipping);

        assert!(callback.on_train_start(&config).is_ok());
        assert!(callback.on_epoch_end(0).is_ok());
        assert!(
            callback
                .on_evaluate(&sample_model(), &sample_eval(), 0)
                .is_ok()
        );
        assert!(
            callback
                .on_train_end(TrainingOutcome::Completed, Some(&sample_eval()), 0)
                .is_ok()
        );
    }

    #[test]
    fn callbacks_dispatches_on_train_start() {
        use crate::loss_functions::CROSS_ENTROPY_LOSS;
        use crate::optimizers::Adam;
        use crate::schedulers::ConstantScheduler;
        use crate::training::{GradientClipping, LearningRate};

        let counts = Rc::new(RefCell::new(Counts::default()));
        let mut callbacks = Callbacks::new(vec![Box::new(CountingCallback(counts.clone()))]);
        let optimizer = Adam::with_defaults(LearningRate::new(0.01));
        let scheduler = ConstantScheduler::new(LearningRate::new(0.01));
        let clipping = GradientClipping::None;
        let config = sample_config(&**CROSS_ENTROPY_LOSS, &optimizer, &scheduler, &clipping);

        callbacks.on_train_start(&config).unwrap();

        assert_eq!(counts.borrow().train_starts, 1);
    }

    struct FailingCallback;

    impl TrainingCallback for FailingCallback {
        fn on_epoch_end(&mut self, _epoch: usize) -> Result<()> {
            Err(Error::other("boom"))
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
    fn dispatches_to_all_children_in_order() {
        let counts = Rc::new(RefCell::new(Counts::default()));
        let mut callbacks = Callbacks::new(vec![Box::new(CountingCallback(counts.clone()))]);

        callbacks.on_epoch_end(1).unwrap();
        callbacks
            .on_evaluate(&sample_model(), &sample_eval(), 1)
            .unwrap();
        callbacks
            .on_train_end(TrainingOutcome::Completed, Some(&sample_eval()), 1)
            .unwrap();

        let counts = counts.borrow();
        assert_eq!(counts.epoch_ends, 1);
        assert_eq!(counts.evaluates, 1);
        assert_eq!(counts.train_ends, 1);
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
}
