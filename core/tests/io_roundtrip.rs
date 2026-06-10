//! End-to-end IO round-trip over the full safetensors pipeline:
//! dataset → scaler → model → training history → inputs, all saved and reloaded.
#![cfg(feature = "io")]

use ndarray::array;
use nrn::activations::RELU;
use nrn::callbacks::TrainingCallback;
use nrn::data::Dataset;
use nrn::data::scalers::{MinMaxScaler, Scaler, ScalerMethod};
use nrn::evaluation::{Evaluation, EvaluationSet};
use nrn::io::data::{load_inputs, save_inputs};
use nrn::io::scalers::ScalerRecord;
use nrn::io::snapshot::{SnapshotArchive, SnapshotRecorder};
use nrn::loss_functions::{CROSS_ENTROPY_LOSS, LossFunction};
use nrn::model::{NeuralNetwork, NeuronLayerSpec};
use nrn::optimizers::Adam;
use nrn::schedulers::ConstantScheduler;
use nrn::training::{GradientClipping, LearningRate};
use std::path::PathBuf;
use std::sync::Arc;

fn temp_dir() -> PathBuf {
    let dir = PathBuf::from(format!("target/nrn_it_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn full_pipeline_roundtrips_through_safetensors() {
    let dir = temp_dir();

    // --- Dataset --------------------------------------------------------
    let dataset = Dataset {
        features: array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        labels: array![0.0, 1.0, 1.0, 0.0],
    };

    let dataset_path = dir.join("dataset");
    dataset.save(&dataset_path).unwrap();
    let loaded_dataset = Dataset::load(&dataset_path).unwrap();
    assert_eq!(dataset.features, loaded_dataset.features);
    assert_eq!(dataset.labels, loaded_dataset.labels);

    // --- Scaler (serialized as JSON, not safetensors) -------------------
    let scaler = ScalerMethod::MinMax(MinMaxScaler::default().fit(dataset.features.view()));
    let mut expected = dataset.features.clone();
    scaler.apply_inplace(expected.view_mut());

    let scaler_path = dir.join("scaler");
    let record: ScalerRecord = scaler.into();
    record.save(&scaler_path).unwrap();
    let reloaded_scaler: ScalerMethod = ScalerRecord::load(&scaler_path).unwrap().into();

    let mut actual = dataset.features.clone();
    reloaded_scaler.apply_inplace(actual.view_mut());
    assert_eq!(expected, actual);

    // --- Model + training history (incremental writer → directory load) --
    let model_dataset = dataset.to_model_dataset();
    let specs = NeuronLayerSpec::network_for(vec![4], &*RELU, 2);
    let mut model = NeuralNetwork::initialization(2, &specs);

    let loss_fn: Arc<dyn LossFunction> = CROSS_ENTROPY_LOSS.clone();
    let mut optimizer = Adam::with_defaults(LearningRate::new(0.05));
    let mut scheduler = ConstantScheduler::new(LearningRate::new(0.05));
    let clipping = GradientClipping::None;

    let history_dir = dir.join("training");
    let mut recorder = SnapshotRecorder::create(&history_dir, "test_dataset", false).unwrap();
    let mut last_recorded_predictions = None;

    for epoch in 0..10 {
        model.train(
            &model_dataset,
            &loss_fn,
            &mut optimizer,
            &mut scheduler,
            &clipping,
            None,
        );
        if epoch % 5 == 0 {
            let predictions = model.predict(model_dataset.inputs.view());
            let loss = loss_fn.compute(predictions.view(), model_dataset.targets.view());
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
                .on_evaluate(&model, &evaluation, epoch * 5)
                .unwrap();
            last_recorded_predictions = Some(predictions);
        }
    }

    let model_path = dir.join("model");
    model.save(&model_path).unwrap();
    let reloaded_model = NeuralNetwork::load(&model_path).unwrap();
    assert_eq!(
        model.predict(model_dataset.inputs.view()),
        reloaded_model.predict(model_dataset.inputs.view())
    );

    let archive = SnapshotArchive::load(&history_dir).unwrap();
    assert_eq!(archive.len(), 2);
    // Last snapshot was written at epoch 5, not at the final epoch.
    // Load the model lazily — only one in memory at a time.
    let last_model = archive.model_at(archive.len() - 1).unwrap();
    assert_eq!(
        last_recorded_predictions.unwrap(),
        last_model.predict(model_dataset.inputs.view())
    );

    // --- Inputs (single-vector prediction file) -------------------------
    let inputs = array![0.0, 1.0];
    let inputs_path = dir.join("inputs");
    save_inputs(&inputs_path, &inputs).unwrap();
    assert_eq!(inputs, load_inputs(&inputs_path).unwrap());

    let _ = std::fs::remove_dir_all(&dir);
}
