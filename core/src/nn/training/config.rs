use crate::gradients::GradientClipping;
use crate::loss_functions::LossFunction;
use crate::optimizers::Optimizer;
use crate::schedulers::Scheduler;
use std::sync::Arc;

/// Owned configuration of a training run, embedded in
/// [`crate::training::TrainingLoop`] and passed by reference to
/// [`crate::training::TrainingCallback::on_train_start`].
pub struct TrainingConfig {
    pub epochs: usize,
    pub eval_interval: usize,
    pub batch_size: Option<usize>,
    pub loss: Arc<dyn LossFunction>,
    pub optimizer: Box<dyn Optimizer>,
    pub scheduler: Box<dyn Scheduler>,
    pub clipping: GradientClipping,
}
