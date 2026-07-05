//! End-to-end IO round-trip over the full safetensors pipeline:
//! dataset → scaler → model → training run → inputs, all saved and reloaded.
#![cfg(feature = "io")]

use ndarray::array;
use nrn::activations::RELU;
use nrn::data::scalers::{MinMaxScaler, Scaler, ScalerMethod};
use nrn::data::{Dataset, Instance};
use nrn::evaluation::{Evaluation, EvaluationSet};
use nrn::io::hyperparams::{
    ClippingRecord, HyperParametersRecord, LossRecord, OptimizerRecord, SchedulerRecord,
};
use nrn::io::run::{TrainingMeta, TrainingRun};
use nrn::io::scalers::ScalerRecord;
use nrn::loss_functions::{CROSS_ENTROPY_LOSS, LossFunction};
use nrn::model::{LayerPlan, NeuralNetwork, NeuronLayerSpec, Predictor};
use nrn::optimizers::Adam;
use nrn::schedulers::ConstantScheduler;
use nrn::training::{GradientClipping, TrainerCallback};
use nrn::weight_decay::WeightDecay;
use std::path::PathBuf;
use std::sync::Arc;

fn temp_dir() -> PathBuf {
    let dir = PathBuf::from(format!("target/nrn_it_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn sample_hyperparams() -> HyperParametersRecord {
    HyperParametersRecord {
        epochs: 10,
        checkpoint_interval: 5,
        batch_size: Some(32),
        lr: 0.05,
        weight_decay: 0.0,
        optimizer: OptimizerRecord::Adam,
        scheduler: SchedulerRecord::Constant,
        clipping: ClippingRecord::None,
        early_stopping: None,
        val_ratio: 0.1,
        test_ratio: 0.1,
        loss: LossRecord::CrossEntropy,
        seed: 0,
        scaler: None,
    }
}

#[test]
fn full_pipeline_roundtrips_through_safetensors() {
    let dir = temp_dir();

    // --- Dataset --------------------------------------------------------
    let dataset = Dataset::new(
        array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        array![0.0, 1.0, 1.0, 0.0],
        None,
    )
    .unwrap();

    let dataset_path = dir.join("dataset");
    dataset.save(&dataset_path).unwrap();
    let loaded = Dataset::load(&dataset_path).unwrap();
    assert_eq!(dataset.features(), loaded.features());
    assert_eq!(dataset.labels(), loaded.labels());

    // --- Scaler (serialized as JSON, not safetensors) -------------------
    // Scalers work on the model dataset's samples-last inputs (features leading).
    let model_dataset = dataset.to_model_dataset();
    let scaler = ScalerMethod::MinMax(MinMaxScaler::default().fit(model_dataset.inputs().view()));
    let mut expected = model_dataset.inputs().clone();
    scaler.apply_inplace(expected.view_mut()).unwrap();

    let scaler_path = dir.join("scaler");
    let record: ScalerRecord = scaler.into();
    record.save(&scaler_path).unwrap();
    let reloaded_scaler: ScalerMethod = ScalerRecord::load(&scaler_path).unwrap().into();

    let mut actual = model_dataset.inputs().clone();
    reloaded_scaler.apply_inplace(actual.view_mut()).unwrap();
    assert_eq!(expected, actual);

    // --- Model + training run (incremental writer → directory load) --
    let specs = NeuronLayerSpec::plan(LayerPlan::Explicit(vec![4]), 2, &*RELU).unwrap();
    let mut model = NeuralNetwork::initialization(2, &specs, 0);

    let loss_fn: Arc<dyn LossFunction> = CROSS_ENTROPY_LOSS.clone();
    let mut optimizer = Adam::with_defaults(0.05.try_into().unwrap(), WeightDecay::ZERO);
    let mut scheduler = ConstantScheduler::new(0.05.try_into().unwrap());
    let clipping = GradientClipping::None;

    let run_dir = dir.join("training");
    let run = TrainingRun::create(
        &run_dir,
        &TrainingMeta {
            dataset: "test_dataset".to_string(),
            model: "model-test_dataset".to_string(),
            hyperparams: sample_hyperparams(),
            scaler: None,
        },
        false,
    )
    .unwrap();
    assert_eq!(run.meta().dataset, "test_dataset");
    assert_eq!(run.meta().model, "model-test_dataset");
    let mut recorder = run.recorder();
    let mut last_recorded_predictions = None;

    for epoch in 0..10 {
        model
            .train(
                &model_dataset,
                &loss_fn,
                &mut optimizer,
                &mut scheduler,
                &clipping,
                None,
            )
            .unwrap();
        if epoch % 5 == 0 {
            let predictions = model.predict(model_dataset.inputs().view()).unwrap();
            let loss = loss_fn.compute(predictions.view(), model_dataset.targets().view());
            let evaluation = EvaluationSet {
                train: Evaluation {
                    loss,
                    accuracy: 0.5,
                },
                validation: Some(Evaluation {
                    loss,
                    accuracy: 0.5,
                }),
                test: Evaluation {
                    loss,
                    accuracy: 0.5,
                },
            };
            recorder
                .on_checkpoint(&model, &optimizer, &scheduler, &evaluation, epoch * 5)
                .unwrap();
            last_recorded_predictions = Some(predictions);
        }
    }

    let model_path = dir.join("model");
    model.save(&model_path).unwrap();
    let reloaded_model = NeuralNetwork::load(&model_path).unwrap();
    assert_eq!(
        model.predict(model_dataset.inputs().view()),
        reloaded_model.predict(model_dataset.inputs().view())
    );

    let archive = run.archive().unwrap();
    assert_eq!(archive.len(), 2);
    // Last checkpoint was written at epoch 5, not at the final epoch.
    // Load the model lazily — only one in memory at a time.
    let last_model = archive.model_at(archive.len() - 1).unwrap();
    assert_eq!(
        last_recorded_predictions.unwrap(),
        last_model.predict(model_dataset.inputs().view()).unwrap()
    );

    // Adam has internal state, so each checkpoint also has an optimizer.safetensors.
    let last_optimizer_state = archive.optimizer_at(archive.len() - 1).unwrap().unwrap();
    assert!(!last_optimizer_state.tensors.is_empty());
    assert!(last_optimizer_state.metadata.contains_key("time_step"));

    // --- Instance (single-vector prediction file) -----------------------
    let instance = Instance::new(array![0.0, 1.0]);
    let instance_path = dir.join("instance");
    instance.save(&instance_path).unwrap();
    assert_eq!(instance, Instance::load(&instance_path).unwrap());

    let _ = std::fs::remove_dir_all(&dir);
}

fn sample_network() -> NeuralNetwork {
    let specs = NeuronLayerSpec::plan(LayerPlan::Explicit(vec![4]), 2, &*RELU).unwrap();
    NeuralNetwork::initialization(2, &specs, 0)
}

#[test]
fn predictor_with_scaler_roundtrips_through_directory() {
    let dir = temp_dir().join("predictor_scaled");
    let input = array![0.3, 0.7];

    let features = array![[0.0, 0.0], [1.0, 1.0]];
    let scaler = ScalerMethod::MinMax(MinMaxScaler::default().fit(features.view()));
    let predictor = Predictor::new(sample_network(), Some(scaler));
    let expected = predictor.predict_single(input.view());

    predictor.save(&dir).unwrap();
    let reloaded = Predictor::load(&dir).unwrap();

    assert!(reloaded.scaler.is_some());
    assert_eq!(expected, reloaded.predict_single(input.view()));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn predictor_without_scaler_roundtrips_with_none() {
    let dir = temp_dir().join("predictor_unscaled");
    let input = array![0.3, 0.7];

    let predictor = Predictor::new(sample_network(), None);
    let expected = predictor.predict_single(input.view());

    predictor.save(&dir).unwrap();
    let reloaded = Predictor::load(&dir).unwrap();

    assert!(reloaded.scaler.is_none());
    assert_eq!(expected, reloaded.predict_single(input.view()));

    let _ = std::fs::remove_dir_all(&dir);
}
