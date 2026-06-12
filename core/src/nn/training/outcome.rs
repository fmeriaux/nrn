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
