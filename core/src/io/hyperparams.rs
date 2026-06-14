use crate::gradients::{GradientClipping, GradientClippingError};
use crate::learning_rate::{LearningRate, LearningRateError};
use crate::loss_functions::CROSS_ENTROPY_LOSS;
use crate::optimizers::{Adam, Optimizer, StochasticGradientDescent};
use crate::schedulers::{
    ConstantScheduler, CosineAnnealing, CosineAnnealingError, Scheduler, StepDecay, StepDecayError,
};
use crate::training::{
    EarlyStoppingConfig, EarlyStoppingConfigError, HyperParams, HyperParamsError,
};
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(tag = "type", content = "params")]
pub enum OptimizerRecord {
    Sgd,
    Adam,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(tag = "type", content = "params")]
pub enum SchedulerRecord {
    Constant,
    Cosine {
        lr_min: f32,
        steps: usize,
        warm_restarts: bool,
        cycle_multiplier: usize,
    },
    Step {
        decay_factor: f32,
        steps: usize,
    },
}

/// Mirrors [`GradientClipping`] for serialization.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(tag = "type", content = "params")]
pub enum ClippingRecord {
    None,
    Norm { max_norm: f32 },
    Value { min: f32, max: f32 },
}

impl From<&GradientClipping> for ClippingRecord {
    fn from(clipping: &GradientClipping) -> Self {
        match clipping {
            GradientClipping::None => ClippingRecord::None,
            GradientClipping::Norm { max_norm } => ClippingRecord::Norm {
                max_norm: *max_norm,
            },
            GradientClipping::Value { min, max } => ClippingRecord::Value {
                min: *min,
                max: *max,
            },
        }
    }
}

impl TryFrom<&ClippingRecord> for GradientClipping {
    type Error = GradientClippingError;

    fn try_from(record: &ClippingRecord) -> Result<Self, Self::Error> {
        match record {
            ClippingRecord::None => Ok(GradientClipping::None),
            ClippingRecord::Norm { max_norm } => GradientClipping::norm(*max_norm),
            ClippingRecord::Value { min, max } => GradientClipping::value(*min, *max),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum LossRecord {
    CrossEntropy,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct EarlyStoppingRecord {
    pub patience: usize,
    pub restore_best_model: bool,
}

/// Serializable mirror of [`HyperParams`], persisted in [`crate::io::run::TrainingMeta`].
///
/// `layers` is intentionally omitted: the model architecture is reconstructed from
/// `model.safetensors` by [`crate::io::run::CheckpointArchive::model_at`].
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct HyperParamsRecord {
    pub epochs: usize,
    pub checkpoint_interval: usize,
    pub batch_size: Option<usize>,
    pub lr: f32,
    pub optimizer: OptimizerRecord,
    pub scheduler: SchedulerRecord,
    pub clipping: ClippingRecord,
    pub early_stopping: Option<EarlyStoppingRecord>,
    pub val_ratio: f32,
    pub test_ratio: f32,
    pub loss: LossRecord,
}

/// Returned by [`HyperParamsRecord::into_hyperparams`] when the record describes an
/// invalid hyperparameter spec.
#[derive(Debug)]
pub enum HyperParamsRecordError {
    LearningRate(LearningRateError),
    StepDecay(StepDecayError),
    CosineAnnealing(CosineAnnealingError),
    Clipping(GradientClippingError),
    EarlyStopping(EarlyStoppingConfigError),
    Validation(HyperParamsError),
}

impl fmt::Display for HyperParamsRecordError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HyperParamsRecordError::LearningRate(e) => write!(f, "{e}"),
            HyperParamsRecordError::StepDecay(e) => write!(f, "{e}"),
            HyperParamsRecordError::CosineAnnealing(e) => write!(f, "{e}"),
            HyperParamsRecordError::Clipping(e) => write!(f, "{e}"),
            HyperParamsRecordError::EarlyStopping(e) => write!(f, "{e}"),
            HyperParamsRecordError::Validation(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for HyperParamsRecordError {}

impl From<LearningRateError> for HyperParamsRecordError {
    fn from(e: LearningRateError) -> Self {
        HyperParamsRecordError::LearningRate(e)
    }
}

impl From<StepDecayError> for HyperParamsRecordError {
    fn from(e: StepDecayError) -> Self {
        HyperParamsRecordError::StepDecay(e)
    }
}

impl From<CosineAnnealingError> for HyperParamsRecordError {
    fn from(e: CosineAnnealingError) -> Self {
        HyperParamsRecordError::CosineAnnealing(e)
    }
}

impl From<GradientClippingError> for HyperParamsRecordError {
    fn from(e: GradientClippingError) -> Self {
        HyperParamsRecordError::Clipping(e)
    }
}

impl From<EarlyStoppingConfigError> for HyperParamsRecordError {
    fn from(e: EarlyStoppingConfigError) -> Self {
        HyperParamsRecordError::EarlyStopping(e)
    }
}

impl From<HyperParamsError> for HyperParamsRecordError {
    fn from(e: HyperParamsError) -> Self {
        HyperParamsRecordError::Validation(e)
    }
}

impl HyperParamsRecord {
    /// Reconstructs the domain [`HyperParams`] spec from this record.
    ///
    /// Every component is rebuilt through its fallible constructor, so an invalid
    /// record (e.g. hand-edited `meta.json`) yields an [`HyperParamsRecordError`]
    /// instead of a panic.
    /// # Errors
    /// Returns [`HyperParamsRecordError`] if any component or the resulting spec is invalid.
    pub fn into_hyperparams(&self) -> Result<HyperParams, HyperParamsRecordError> {
        let lr = LearningRate::new(self.lr)?;

        let optimizer: Box<dyn Optimizer> = match self.optimizer {
            OptimizerRecord::Sgd => Box::new(StochasticGradientDescent::new(lr)),
            OptimizerRecord::Adam => Box::new(Adam::with_defaults(lr)),
        };

        let scheduler: Box<dyn Scheduler> = match &self.scheduler {
            SchedulerRecord::Constant => Box::new(ConstantScheduler::from_value(self.lr)?),
            SchedulerRecord::Cosine {
                lr_min,
                steps,
                warm_restarts,
                cycle_multiplier,
            } => {
                let cosine = CosineAnnealing::from_values(*lr_min, self.lr, *steps)?;
                if *warm_restarts {
                    Box::new(cosine.with_restarts(true, *cycle_multiplier)?)
                } else {
                    Box::new(cosine)
                }
            }
            SchedulerRecord::Step {
                decay_factor,
                steps,
            } => Box::new(StepDecay::from_values(self.lr, *steps, *decay_factor)?),
        };

        let clipping = GradientClipping::try_from(&self.clipping)?;

        let early_stopping = self
            .early_stopping
            .as_ref()
            .map(|record| EarlyStoppingConfig::new(record.patience, record.restore_best_model))
            .transpose()?;

        let loss = match self.loss {
            LossRecord::CrossEntropy => CROSS_ENTROPY_LOSS.clone(),
        };

        let hyperparams = HyperParams::new(
            self.epochs,
            self.checkpoint_interval,
            self.batch_size,
            loss,
            optimizer,
            scheduler,
            clipping,
            early_stopping,
            self.val_ratio,
            self.test_ratio,
        )?;

        Ok(hyperparams)
    }
}

#[cfg(test)]
impl HyperParamsRecord {
    /// A representative, valid record used as a fixture in tests.
    pub fn sample() -> Self {
        HyperParamsRecord {
            epochs: 10,
            checkpoint_interval: 5,
            batch_size: Some(32),
            lr: 0.001,
            optimizer: OptimizerRecord::Adam,
            scheduler: SchedulerRecord::Constant,
            clipping: ClippingRecord::Norm { max_norm: 1.0 },
            early_stopping: Some(EarlyStoppingRecord {
                patience: 3,
                restore_best_model: true,
            }),
            val_ratio: 0.1,
            test_ratio: 0.1,
            loss: LossRecord::CrossEntropy,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::SIGMOID;
    use crate::model::{NeuralNetwork, NeuronLayerSpec};

    #[test]
    fn optimizer_record_roundtrips_through_json() {
        for record in [OptimizerRecord::Sgd, OptimizerRecord::Adam] {
            let json = serde_json::to_string(&record).unwrap();
            assert_eq!(
                serde_json::from_str::<OptimizerRecord>(&json).unwrap(),
                record
            );
        }
    }

    #[test]
    fn scheduler_record_roundtrips_through_json() {
        for record in [
            SchedulerRecord::Constant,
            SchedulerRecord::Cosine {
                lr_min: 0.0001,
                steps: 100,
                warm_restarts: true,
                cycle_multiplier: 2,
            },
            SchedulerRecord::Step {
                decay_factor: 0.5,
                steps: 10,
            },
        ] {
            let json = serde_json::to_string(&record).unwrap();
            assert_eq!(
                serde_json::from_str::<SchedulerRecord>(&json).unwrap(),
                record
            );
        }
    }

    #[test]
    fn clipping_record_roundtrips_through_json() {
        for record in [
            ClippingRecord::None,
            ClippingRecord::Norm { max_norm: 1.0 },
            ClippingRecord::Value {
                min: -1.0,
                max: 1.0,
            },
        ] {
            let json = serde_json::to_string(&record).unwrap();
            assert_eq!(
                serde_json::from_str::<ClippingRecord>(&json).unwrap(),
                record
            );
        }
    }

    #[test]
    fn clipping_record_converts_both_ways() {
        for clipping in [
            GradientClipping::None,
            GradientClipping::Norm { max_norm: 2.0 },
            GradientClipping::Value {
                min: -1.0,
                max: 1.0,
            },
        ] {
            let record = ClippingRecord::from(&clipping);
            let back = GradientClipping::try_from(&record).unwrap();
            assert_eq!(ClippingRecord::from(&back), record);
        }
    }

    #[test]
    fn hyperparams_record_roundtrips_through_json() {
        let record = HyperParamsRecord::sample();
        let json = serde_json::to_string(&record).unwrap();
        assert_eq!(
            serde_json::from_str::<HyperParamsRecord>(&json).unwrap(),
            record
        );
    }

    #[test]
    fn into_hyperparams_reconstructs_the_spec() {
        let record = HyperParamsRecord::sample();
        let hyperparams = record.into_hyperparams().unwrap();

        assert_eq!(hyperparams.epochs(), 10);
        assert_eq!(hyperparams.checkpoint_interval(), 5);
        assert_eq!(hyperparams.batch_size(), Some(32));
        assert_eq!(hyperparams.optimizer().name(), "Adam");
        assert_eq!(hyperparams.scheduler().name(), "Constant");
        assert!(matches!(
            hyperparams.clipping(),
            GradientClipping::Norm { max_norm } if *max_norm == 1.0
        ));
        let specs = NeuronLayerSpec::network_for(vec![2], &*SIGMOID, 2);
        let model = NeuralNetwork::initialization(2, &specs);
        assert!(hyperparams.build_early_stopping(&model).is_some());
        assert_eq!(hyperparams.val_ratio(), 0.1);
        assert_eq!(hyperparams.test_ratio(), 0.1);
    }

    #[test]
    fn into_hyperparams_builds_cosine_with_warm_restarts() {
        let mut record = HyperParamsRecord::sample();
        record.scheduler = SchedulerRecord::Cosine {
            lr_min: 0.0001,
            steps: 100,
            warm_restarts: true,
            cycle_multiplier: 2,
        };

        let hyperparams = record.into_hyperparams().unwrap();
        assert_eq!(hyperparams.scheduler().name(), "Cosine Annealing");
    }

    #[test]
    fn into_hyperparams_builds_step_decay() {
        let mut record = HyperParamsRecord::sample();
        record.scheduler = SchedulerRecord::Step {
            decay_factor: 0.5,
            steps: 10,
        };

        let hyperparams = record.into_hyperparams().unwrap();
        assert_eq!(hyperparams.scheduler().name(), "Step Decay");
    }

    #[test]
    fn into_hyperparams_rejects_invalid_learning_rate() {
        let mut record = HyperParamsRecord::sample();
        record.lr = -1.0;

        assert!(matches!(
            record.into_hyperparams(),
            Err(HyperParamsRecordError::LearningRate(_))
        ));
    }

    #[test]
    fn into_hyperparams_rejects_invalid_clipping() {
        let mut record = HyperParamsRecord::sample();
        record.clipping = ClippingRecord::Norm { max_norm: -1.0 };

        assert!(matches!(
            record.into_hyperparams(),
            Err(HyperParamsRecordError::Clipping(_))
        ));
    }

    #[test]
    fn into_hyperparams_rejects_zero_early_stopping_patience() {
        let mut record = HyperParamsRecord::sample();
        record.early_stopping = Some(EarlyStoppingRecord {
            patience: 0,
            restore_best_model: true,
        });

        assert!(matches!(
            record.into_hyperparams(),
            Err(HyperParamsRecordError::EarlyStopping(_))
        ));
    }

    #[test]
    fn into_hyperparams_rejects_invalid_split_ratios() {
        let mut record = HyperParamsRecord::sample();
        record.val_ratio = 0.6;
        record.test_ratio = 0.6;

        assert!(matches!(
            record.into_hyperparams(),
            Err(HyperParamsRecordError::Validation(_))
        ));
    }
}
