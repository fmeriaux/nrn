use ndarray::array;
use nrn::activations::{IDENTITY, SIGMOID};
use nrn::data::{Dataset, Targets};
use nrn::loss_functions::Reduction;
use nrn::model::{NetworkConfig, NeuralNetwork};
use nrn::task::Task;
use nrn::training::{
    Callbacks, GradientClipping, HyperParameters, LossConfig, LossKind, OptimizerConfig,
    SchedulerConfig,
};

#[test]
fn xor_converges_to_low_loss() {
    // XOR: non-linearly separable, requires at least one hidden layer.
    // The dataset holds two identical copies of the 4 XOR points (8 samples); with
    // `test_ratio = 0.5` the shuffled split trains on half of them, so every XOR
    // point is still represented in training.
    let xor_dataset = || {
        Dataset::tabular(
            array![
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0]
            ], // (8 samples = 4 XOR points ×2, 2 features)
            Targets::class_ids(array![0u32, 1, 1, 0, 0, 1, 1, 0]).unwrap(),
            None,
        )
        .unwrap()
    };

    let config = NetworkConfig::builder(vec![2])
        .dense(8, &SIGMOID)
        .dense(1, &IDENTITY)
        .build();

    let hyperparameters = HyperParameters::from_values(
        10_000,
        0,
        None,
        0.1,
        0.0,
        OptimizerConfig::Adam,
        SchedulerConfig::Constant,
        GradientClipping::None,
        LossConfig {
            kind: LossKind::BinaryCrossEntropy,
            reduction: Reduction::Mean,
        },
        None,
        0.0,
        0.5,
        42,
        None,
    )
    .unwrap();
    let data = hyperparameters.prepare(xor_dataset(), None).unwrap();
    let report = hyperparameters
        .build(
            NeuralNetwork::from_config(config, 42).unwrap(),
            Task::Binary,
            data,
            Callbacks::new(vec![]),
        )
        .unwrap()
        .train()
        .unwrap();

    let loss = report.final_evaluation.unwrap().train.loss;
    assert!(loss < 0.05, "XOR did not converge: final loss = {loss:.4}");
}

#[test]
fn xor_converges_with_mini_batch() {
    // Mini-batch of 2 on the 4-sample XOR train split (2 batches per epoch, shuffled).
    // Adam's v → 0 after convergence can cause NaN if training continues too long without
    // lr decay, so we stop at 8 000 epochs where loss is reliably < 0.01.
    // See `xor_converges_to_low_loss` for the doubled-dataset / split rationale.
    let xor_dataset = || {
        Dataset::tabular(
            array![
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0]
            ],
            Targets::class_ids(array![0u32, 1, 1, 0, 0, 1, 1, 0]).unwrap(),
            None,
        )
        .unwrap()
    };

    let config = NetworkConfig::builder(vec![2])
        .dense(8, &SIGMOID)
        .dense(1, &IDENTITY)
        .build();

    let hyperparameters = HyperParameters::from_values(
        8_000,
        0,
        Some(2),
        0.01,
        0.0,
        OptimizerConfig::Adam,
        SchedulerConfig::Constant,
        GradientClipping::None,
        LossConfig {
            kind: LossKind::BinaryCrossEntropy,
            reduction: Reduction::Mean,
        },
        None,
        0.0,
        0.5,
        42,
        None,
    )
    .unwrap();
    let data = hyperparameters.prepare(xor_dataset(), None).unwrap();
    let report = hyperparameters
        .build(
            NeuralNetwork::from_config(config, 42).unwrap(),
            Task::Binary,
            data,
            Callbacks::new(vec![]),
        )
        .unwrap()
        .train()
        .unwrap();

    let loss = report.final_evaluation.unwrap().train.loss;
    assert!(
        loss < 0.05,
        "XOR (mini-batch) did not converge: final loss = {loss:.4}"
    );
}

#[test]
fn three_class_converges_to_low_loss() {
    // 3 linearly separable points, one per class. Exercises the full softmax
    // output path end-to-end. The dataset holds two copies of the 3 points (6
    // samples); with `test_ratio = 0.5` the shuffled split trains on half of them.
    let three_class_dataset = || {
        Dataset::tabular(
            array![
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0]
            ], // (6 samples, 2 features)
            Targets::class_ids(array![0u32, 1, 2, 0, 1, 2]).unwrap(),
            None,
        )
        .unwrap()
    };

    // Sigmoid avoids the dead-neuron risk of ReLU(0)=0 for the [0.0, 0.0] sample
    // (He init sets biases to zero, so relu([0,0]) = 0 and its gradient is dead at epoch 0).
    let config = NetworkConfig::builder(vec![2])
        .dense(8, &SIGMOID)
        .dense(3, &IDENTITY)
        .build();

    let hyperparameters = HyperParameters::from_values(
        5_000,
        0,
        None,
        0.05,
        0.0,
        OptimizerConfig::Adam,
        SchedulerConfig::Constant,
        GradientClipping::None,
        LossConfig {
            kind: LossKind::SparseCategoricalCrossEntropy,
            reduction: Reduction::Mean,
        },
        None,
        0.0,
        0.5,
        42,
        None,
    )
    .unwrap();
    let data = hyperparameters
        .prepare(three_class_dataset(), None)
        .unwrap();
    let report = hyperparameters
        .build(
            NeuralNetwork::from_config(config, 42).unwrap(),
            Task::MultiClass { class_count: 3 },
            data,
            Callbacks::new(vec![]),
        )
        .unwrap()
        .train()
        .unwrap();

    let loss = report.final_evaluation.unwrap().train.loss;
    assert!(
        loss < 0.05,
        "3-class classification did not converge: final loss = {loss:.4}"
    );
}
