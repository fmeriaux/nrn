use crate::data::scalers::ScalerKind;
use crate::gradients::{GradientClipping, GradientClippingError};
use crate::loss_functions::Reduction;
use crate::training::{
    EarlyStoppingConfig, EarlyStoppingConfigError, HyperParameters, HyperParametersError,
    LossConfig, LossKind, OptimizerConfig, SchedulerConfig,
};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(tag = "type", content = "params")]
pub enum OptimizerRecord {
    Sgd,
    Adam,
}

impl From<&OptimizerConfig> for OptimizerRecord {
    fn from(config: &OptimizerConfig) -> Self {
        match config {
            OptimizerConfig::Sgd => OptimizerRecord::Sgd,
            OptimizerConfig::Adam => OptimizerRecord::Adam,
        }
    }
}

impl From<&OptimizerRecord> for OptimizerConfig {
    fn from(record: &OptimizerRecord) -> Self {
        match record {
            OptimizerRecord::Sgd => OptimizerConfig::Sgd,
            OptimizerRecord::Adam => OptimizerConfig::Adam,
        }
    }
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

impl From<&SchedulerConfig> for SchedulerRecord {
    fn from(config: &SchedulerConfig) -> Self {
        match config {
            SchedulerConfig::Constant => SchedulerRecord::Constant,
            SchedulerConfig::Cosine {
                lr_min,
                steps,
                warm_restarts,
                cycle_multiplier,
            } => SchedulerRecord::Cosine {
                lr_min: *lr_min,
                steps: *steps,
                warm_restarts: *warm_restarts,
                cycle_multiplier: *cycle_multiplier,
            },
            SchedulerConfig::Step {
                decay_factor,
                steps,
            } => SchedulerRecord::Step {
                decay_factor: *decay_factor,
                steps: *steps,
            },
        }
    }
}

impl From<&SchedulerRecord> for SchedulerConfig {
    fn from(record: &SchedulerRecord) -> Self {
        match record {
            SchedulerRecord::Constant => SchedulerConfig::Constant,
            SchedulerRecord::Cosine {
                lr_min,
                steps,
                warm_restarts,
                cycle_multiplier,
            } => SchedulerConfig::Cosine {
                lr_min: *lr_min,
                steps: *steps,
                warm_restarts: *warm_restarts,
                cycle_multiplier: *cycle_multiplier,
            },
            SchedulerRecord::Step {
                decay_factor,
                steps,
            } => SchedulerConfig::Step {
                decay_factor: *decay_factor,
                steps: *steps,
            },
        }
    }
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
pub struct LossRecord {
    pub kind: LossKindRecord,
    pub reduction: ReductionRecord,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum LossKindRecord {
    BinaryCrossEntropy,
    CategoricalCrossEntropy,
    MeanSquaredError,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ReductionRecord {
    Sum,
    Mean,
}

impl From<&LossConfig> for LossRecord {
    fn from(config: &LossConfig) -> Self {
        LossRecord {
            kind: (&config.kind).into(),
            reduction: (&config.reduction).into(),
        }
    }
}

impl From<&LossRecord> for LossConfig {
    fn from(record: &LossRecord) -> Self {
        LossConfig {
            kind: (&record.kind).into(),
            reduction: (&record.reduction).into(),
        }
    }
}

impl From<&LossKind> for LossKindRecord {
    fn from(kind: &LossKind) -> Self {
        match kind {
            LossKind::BinaryCrossEntropy => LossKindRecord::BinaryCrossEntropy,
            LossKind::CategoricalCrossEntropy => LossKindRecord::CategoricalCrossEntropy,
            LossKind::MeanSquaredError => LossKindRecord::MeanSquaredError,
        }
    }
}

impl From<&LossKindRecord> for LossKind {
    fn from(record: &LossKindRecord) -> Self {
        match record {
            LossKindRecord::BinaryCrossEntropy => LossKind::BinaryCrossEntropy,
            LossKindRecord::CategoricalCrossEntropy => LossKind::CategoricalCrossEntropy,
            LossKindRecord::MeanSquaredError => LossKind::MeanSquaredError,
        }
    }
}

impl From<&Reduction> for ReductionRecord {
    fn from(reduction: &Reduction) -> Self {
        match reduction {
            Reduction::Sum => ReductionRecord::Sum,
            Reduction::Mean => ReductionRecord::Mean,
        }
    }
}

impl From<&ReductionRecord> for Reduction {
    fn from(record: &ReductionRecord) -> Self {
        match record {
            ReductionRecord::Sum => Reduction::Sum,
            ReductionRecord::Mean => Reduction::Mean,
        }
    }
}

/// Mirrors [`ScalerKind`] for serialization.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(tag = "type", content = "params")]
pub enum ScalerKindRecord {
    MinMax,
    ZScore,
}

impl From<ScalerKind> for ScalerKindRecord {
    fn from(kind: ScalerKind) -> Self {
        match kind {
            ScalerKind::MinMax => ScalerKindRecord::MinMax,
            ScalerKind::ZScore => ScalerKindRecord::ZScore,
        }
    }
}

impl From<&ScalerKindRecord> for ScalerKind {
    fn from(record: &ScalerKindRecord) -> Self {
        match record {
            ScalerKindRecord::MinMax => ScalerKind::MinMax,
            ScalerKindRecord::ZScore => ScalerKind::ZScore,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct EarlyStoppingRecord {
    pub patience: usize,
    pub restore_best_model: bool,
}

impl From<&EarlyStoppingConfig> for EarlyStoppingRecord {
    fn from(config: &EarlyStoppingConfig) -> Self {
        EarlyStoppingRecord {
            patience: config.patience(),
            restore_best_model: config.restore_best_model(),
        }
    }
}

impl TryFrom<&EarlyStoppingRecord> for EarlyStoppingConfig {
    type Error = EarlyStoppingConfigError;

    fn try_from(record: &EarlyStoppingRecord) -> Result<Self, Self::Error> {
        EarlyStoppingConfig::new(record.patience, record.restore_best_model)
    }
}

/// Serializable mirror of [`HyperParameters`], persisted in
/// [`crate::io::model::run::TrainingMeta`].
///
/// `layers` is intentionally omitted: the model architecture is reconstructed from
/// `model.safetensors` by [`crate::io::model::run::CheckpointArchive::model_at`].
///
/// Conversions are lossless both ways: [`From<&HyperParameters>`](HyperParametersRecord)
/// projects the domain spec onto this record, and [`TryFrom<HyperParametersRecord>`]
/// validates a record back into [`HyperParameters`].
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct HyperParametersRecord {
    pub epochs: usize,
    #[serde(default)]
    pub checkpoint_interval: usize,
    #[serde(default)]
    pub batch_size: Option<usize>,
    pub lr: f32,
    #[serde(default)]
    pub weight_decay: f32,
    pub optimizer: OptimizerRecord,
    pub scheduler: SchedulerRecord,
    pub clipping: ClippingRecord,
    #[serde(default)]
    pub early_stopping: Option<EarlyStoppingRecord>,
    pub val_ratio: f32,
    pub test_ratio: f32,
    pub loss: LossRecord,
    pub seed: u64,
    #[serde(default)]
    pub scaler: Option<ScalerKindRecord>,
}

impl From<&HyperParameters> for HyperParametersRecord {
    fn from(hyperparameters: &HyperParameters) -> Self {
        HyperParametersRecord {
            epochs: hyperparameters.epochs(),
            checkpoint_interval: hyperparameters.checkpoint_interval(),
            batch_size: hyperparameters.batch_size(),
            lr: hyperparameters.lr().value(),
            weight_decay: hyperparameters.weight_decay().value(),
            optimizer: hyperparameters.optimizer().into(),
            scheduler: hyperparameters.scheduler().into(),
            clipping: hyperparameters.clipping().into(),
            early_stopping: hyperparameters.early_stopping().map(Into::into),
            val_ratio: hyperparameters.val_ratio(),
            test_ratio: hyperparameters.test_ratio(),
            loss: hyperparameters.loss().into(),
            seed: hyperparameters.seed(),
            scaler: hyperparameters.scaler().map(Into::into),
        }
    }
}

impl TryFrom<HyperParametersRecord> for HyperParameters {
    type Error = HyperParametersError;

    /// Validates a record back into the domain spec. The caller-specific
    /// components (clipping, early stopping) are rebuilt through their own
    /// fallible conversions, whose errors fold into [`HyperParametersError`] via
    /// `?`; the rest is delegated to [`HyperParameters::from_values`]. So an
    /// invalid record (e.g. a hand-edited `meta.json`) yields a single typed
    /// error instead of a panic.
    fn try_from(record: HyperParametersRecord) -> Result<Self, Self::Error> {
        let clipping = GradientClipping::try_from(&record.clipping)?;
        let early_stopping = record
            .early_stopping
            .as_ref()
            .map(EarlyStoppingConfig::try_from)
            .transpose()?;

        HyperParameters::from_values(
            record.epochs,
            record.checkpoint_interval,
            record.batch_size,
            record.lr,
            record.weight_decay,
            (&record.optimizer).into(),
            (&record.scheduler).into(),
            clipping,
            (&record.loss).into(),
            early_stopping,
            record.val_ratio,
            record.test_ratio,
            record.seed,
            record.scaler.as_ref().map(Into::into),
        )
    }
}

#[cfg(test)]
impl HyperParametersRecord {
    /// A representative, valid record used as a fixture in tests.
    pub fn sample() -> Self {
        HyperParametersRecord {
            epochs: 10,
            checkpoint_interval: 5,
            batch_size: Some(32),
            lr: 0.001,
            weight_decay: 0.0,
            optimizer: OptimizerRecord::Adam,
            scheduler: SchedulerRecord::Constant,
            clipping: ClippingRecord::Norm { max_norm: 1.0 },
            early_stopping: Some(EarlyStoppingRecord {
                patience: 3,
                restore_best_model: true,
            }),
            val_ratio: 0.1,
            test_ratio: 0.1,
            loss: LossRecord {
                kind: LossKindRecord::BinaryCrossEntropy,
                reduction: ReductionRecord::Mean,
            },
            seed: 42,
            scaler: Some(ScalerKindRecord::MinMax),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn scaler_kind_record_converts_both_ways() {
        for kind in [ScalerKind::MinMax, ScalerKind::ZScore] {
            let record = ScalerKindRecord::from(kind);
            assert_eq!(ScalerKind::from(&record), kind);
        }
    }

    #[test]
    fn missing_optional_keys_deserialize_to_disabled_defaults() {
        // A meta.json from an older schema may omit the optional, off-by-default
        // knobs (e.g. weight decay, which postdates earlier runs); each must still
        // deserialize, falling back to its disabled sentinel.
        let mut value = serde_json::to_value(HyperParametersRecord::sample()).unwrap();
        let object = value.as_object_mut().unwrap();
        for key in [
            "checkpoint_interval",
            "batch_size",
            "weight_decay",
            "early_stopping",
            "scaler",
        ] {
            object.remove(key);
        }

        let record: HyperParametersRecord = serde_json::from_value(value).unwrap();
        assert_eq!(record.checkpoint_interval, 0);
        assert_eq!(record.batch_size, None);
        assert_eq!(record.weight_decay, 0.0);
        assert_eq!(record.early_stopping, None);
        assert_eq!(record.scaler, None);
    }

    #[test]
    fn hyperparameters_record_roundtrips_through_json() {
        let record = HyperParametersRecord::sample();
        let json = serde_json::to_string(&record).unwrap();
        assert_eq!(
            serde_json::from_str::<HyperParametersRecord>(&json).unwrap(),
            record
        );
    }

    #[test]
    fn record_roundtrips_through_the_domain_spec() {
        // The conversion is lossless both ways for *every* optimizer and scheduler
        // variant, not just the sample's — exercising both projection directions
        // (`From<&Config>` and the `TryFrom` back) for each.
        let with_sgd = HyperParametersRecord {
            optimizer: OptimizerRecord::Sgd,
            ..HyperParametersRecord::sample()
        };
        let with_cosine = HyperParametersRecord {
            scheduler: SchedulerRecord::Cosine {
                lr_min: 0.0001,
                steps: 100,
                warm_restarts: true,
                cycle_multiplier: 2,
            },
            ..HyperParametersRecord::sample()
        };
        let with_step = HyperParametersRecord {
            scheduler: SchedulerRecord::Step {
                decay_factor: 0.5,
                steps: 10,
            },
            ..HyperParametersRecord::sample()
        };
        let with_weight_decay = HyperParametersRecord {
            weight_decay: 0.0001,
            ..HyperParametersRecord::sample()
        };

        for record in [
            HyperParametersRecord::sample(),
            with_sgd,
            with_cosine,
            with_step,
            with_weight_decay,
        ] {
            let hyperparameters = HyperParameters::try_from(record.clone()).unwrap();
            assert_eq!(HyperParametersRecord::from(&hyperparameters), record);
        }
    }

    #[test]
    fn try_from_reconstructs_the_spec() {
        let record = HyperParametersRecord::sample();
        let hyperparameters = HyperParameters::try_from(record).unwrap();

        assert_eq!(hyperparameters.epochs(), 10);
        assert_eq!(hyperparameters.checkpoint_interval(), 5);
        assert_eq!(hyperparameters.batch_size(), Some(32));
        assert_eq!(*hyperparameters.optimizer(), OptimizerConfig::Adam);
        assert_eq!(*hyperparameters.scheduler(), SchedulerConfig::Constant);
        assert!(matches!(
            hyperparameters.clipping(),
            GradientClipping::Norm { max_norm } if *max_norm == 1.0
        ));
        assert!(hyperparameters.early_stopping().is_some());
        assert_eq!(hyperparameters.val_ratio(), 0.1);
        assert_eq!(hyperparameters.test_ratio(), 0.1);
    }

    #[test]
    fn try_from_builds_cosine_with_warm_restarts() {
        let mut record = HyperParametersRecord::sample();
        record.scheduler = SchedulerRecord::Cosine {
            lr_min: 0.0001,
            steps: 100,
            warm_restarts: true,
            cycle_multiplier: 2,
        };

        let hyperparameters = HyperParameters::try_from(record).unwrap();
        assert!(matches!(
            hyperparameters.scheduler(),
            SchedulerConfig::Cosine {
                warm_restarts: true,
                cycle_multiplier: 2,
                ..
            }
        ));
    }

    #[test]
    fn try_from_builds_step_decay() {
        let mut record = HyperParametersRecord::sample();
        record.scheduler = SchedulerRecord::Step {
            decay_factor: 0.5,
            steps: 10,
        };

        let hyperparameters = HyperParameters::try_from(record).unwrap();
        assert!(matches!(
            hyperparameters.scheduler(),
            SchedulerConfig::Step { steps: 10, .. }
        ));
    }

    #[test]
    fn try_from_rejects_invalid_learning_rate() {
        let mut record = HyperParametersRecord::sample();
        record.lr = -1.0;

        assert!(matches!(
            HyperParameters::try_from(record),
            Err(HyperParametersError::LearningRate(_))
        ));
    }

    #[test]
    fn try_from_rejects_invalid_clipping() {
        let mut record = HyperParametersRecord::sample();
        record.clipping = ClippingRecord::Norm { max_norm: -1.0 };

        assert!(matches!(
            HyperParameters::try_from(record),
            Err(HyperParametersError::Clipping(_))
        ));
    }

    #[test]
    fn try_from_rejects_zero_early_stopping_patience() {
        let mut record = HyperParametersRecord::sample();
        record.early_stopping = Some(EarlyStoppingRecord {
            patience: 0,
            restore_best_model: true,
        });

        assert!(matches!(
            HyperParameters::try_from(record),
            Err(HyperParametersError::EarlyStopping(_))
        ));
    }

    #[test]
    fn try_from_rejects_invalid_split_ratios() {
        let mut record = HyperParametersRecord::sample();
        record.val_ratio = 0.6;
        record.test_ratio = 0.6;

        assert!(matches!(
            HyperParameters::try_from(record),
            Err(HyperParametersError::SplitRatiosTooLarge { .. })
        ));
    }

    #[test]
    fn try_from_rejects_invalid_cosine_bounds() {
        let mut record = HyperParametersRecord::sample();
        // lr_min >= lr (max) is rejected by the cosine schedule.
        record.lr = 0.001;
        record.scheduler = SchedulerRecord::Cosine {
            lr_min: 0.01,
            steps: 100,
            warm_restarts: false,
            cycle_multiplier: 1,
        };

        assert!(matches!(
            HyperParameters::try_from(record),
            Err(HyperParametersError::Cosine(_))
        ));
    }
}
