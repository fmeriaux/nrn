use crate::data::ModelSplit;
use crate::data::scalers::ScalerMethod;

/// A train/validation/test split with its input scaler applied, ready to build a
/// [`Trainer`](crate::training::Trainer). The scaler is fitted on the train split
/// alone (or supplied explicitly when resuming) and applied to every split.
/// Produced by [`HyperParameters::prepare`](crate::training::HyperParameters::prepare).
pub struct TrainingData {
    pub(super) split: ModelSplit,
    pub(super) scaler: Option<ScalerMethod>,
}

impl TrainingData {
    /// Applies `scaler` to every split, or leaves the inputs untouched when it is
    /// `None`, and bundles the result with the scaler.
    pub fn new(mut split: ModelSplit, scaler: Option<ScalerMethod>) -> Self {
        if let Some(scaler) = &scaler {
            split.scale_inplace(scaler);
        }

        Self { split, scaler }
    }

    /// The scaler applied to the splits, or `None` when no scaling was configured.
    pub fn scaler(&self) -> Option<&ScalerMethod> {
        self.scaler.as_ref()
    }
}
