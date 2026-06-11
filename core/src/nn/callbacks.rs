pub use crate::training_outcome::TrainingOutcome;

use crate::evaluation::EvaluationSet;
use crate::model::NeuralNetwork;
use crate::training::TrainingConfig;
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

    struct CountingCallback(Rc<RefCell<usize>>);

    impl TrainingCallback for CountingCallback {
        fn on_epoch_end(&mut self, _epoch: usize) -> Result<()> {
            *self.0.borrow_mut() += 1;
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
        let epoch_ends = Rc::new(RefCell::new(0));
        let mut callbacks = Callbacks::new(vec![
            Box::new(FailingCallback),
            Box::new(CountingCallback(epoch_ends.clone())),
        ]);

        let result = callbacks.on_epoch_end(1);

        assert!(result.is_err());
        assert_eq!(*epoch_ends.borrow(), 0);
    }
}
