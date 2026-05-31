//! End-to-end IO round-trip over the full safetensors pipeline:
//! dataset → scaler → model → checkpoints → inputs, all saved and reloaded.
#![cfg(feature = "io")]

use ndarray::array;
use nrn::activations::RELU;
use nrn::checkpoints::Checkpoints;
use nrn::data::Dataset;
use nrn::data::scalers::{MinMaxScaler, Scaler, ScalerMethod};
use nrn::evaluation::{Evaluation, EvaluationSet};
use nrn::io::data::{load_inputs, save_inputs};
use nrn::io::scalers::ScalerRecord;
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
    // Dataset is row-major: (samples, features). XOR = 4 samples, 2 features.
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

    // --- Model + checkpoints (trained briefly for non-trivial weights) --
    let model_dataset = dataset.to_model_dataset();
    let specs = NeuronLayerSpec::network_for(vec![4], &*RELU, 2);
    let mut model = NeuralNetwork::initialization(2, &specs);

    let loss_fn: Arc<dyn LossFunction> = CROSS_ENTROPY_LOSS.clone();
    let mut optimizer = Adam::with_defaults(LearningRate::new(0.05));
    let mut scheduler = ConstantScheduler::new(LearningRate::new(0.05));
    let clipping = GradientClipping::None;

    let mut checkpoints = Checkpoints::by_interval(5, 10).unwrap();
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
            checkpoints.record(&model, &evaluation);
        }
    }

    let model_path = dir.join("model");
    model.save(&model_path).unwrap();
    let reloaded_model = NeuralNetwork::load(&model_path).unwrap();
    assert_eq!(
        model.predict(model_dataset.inputs.view()),
        reloaded_model.predict(model_dataset.inputs.view())
    );

    let checkpoints_path = dir.join("training");
    checkpoints.save(&checkpoints_path).unwrap();
    let reloaded_checkpoints = Checkpoints::load(&checkpoints_path).unwrap();
    assert_eq!(reloaded_checkpoints.interval, checkpoints.interval);
    assert_eq!(reloaded_checkpoints.len(), checkpoints.len());
    assert_eq!(
        checkpoints
            .snapshots
            .last()
            .unwrap()
            .predict(model_dataset.inputs.view()),
        reloaded_checkpoints
            .snapshots
            .last()
            .unwrap()
            .predict(model_dataset.inputs.view())
    );

    // --- Inputs (single-vector prediction file) -------------------------
    let inputs = array![0.0, 1.0];
    let inputs_path = dir.join("inputs");
    save_inputs(&inputs_path, &inputs).unwrap();
    assert_eq!(inputs, load_inputs(&inputs_path).unwrap());

    let _ = std::fs::remove_dir_all(&dir);
}
