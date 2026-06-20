use ndarray::array;
use nrn::activations::SIGMOID;
use nrn::data::ModelDataset;
use nrn::model::{LayerPlan, NeuralNetwork, NeuronLayerSpec};
use nrn::training::{
    Callbacks, GradientClipping, HyperParameters, LossConfig, OptimizerConfig, SchedulerConfig,
};

#[test]
fn xor_converges_to_low_loss() {
    // XOR: non-linearly separable, requires at least one hidden layer.
    // The dataset holds two identical copies of the 4 XOR points; with
    // `val_ratio = 0.0` and `test_ratio = 0.5` the split's `train` is exactly the
    // first copy (the full XOR set) and `test` the second — so the model still
    // trains on all four points.
    let xor_dataset = || ModelDataset {
        inputs: array![
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        ], // (2 features, 8 samples = 4 XOR points ×2)
        targets: array![[0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]], // (1 output, 8 samples)
    };

    let specs = NeuronLayerSpec::plan(LayerPlan::Explicit(vec![8]), 2, &*SIGMOID).unwrap();

    let report = HyperParameters::from_values(
        10_000,
        0,
        None,
        0.1,
        OptimizerConfig::Adam,
        SchedulerConfig::Constant,
        GradientClipping::None,
        LossConfig::CrossEntropy,
        None,
        0.0,
        0.5,
        42,
    )
    .unwrap()
    .build(
        NeuralNetwork::initialization(2, &specs, 42),
        xor_dataset(),
        Callbacks::new(vec![]),
    )
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
    let xor_dataset = || ModelDataset {
        inputs: array![
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        ],
        targets: array![[0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]],
    };

    let specs = NeuronLayerSpec::plan(LayerPlan::Explicit(vec![8]), 2, &*SIGMOID).unwrap();

    let report = HyperParameters::from_values(
        8_000,
        0,
        Some(2),
        0.01,
        OptimizerConfig::Adam,
        SchedulerConfig::Constant,
        GradientClipping::None,
        LossConfig::CrossEntropy,
        None,
        0.0,
        0.5,
        42,
    )
    .unwrap()
    .build(
        NeuralNetwork::initialization(2, &specs, 42),
        xor_dataset(),
        Callbacks::new(vec![]),
    )
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
    // output path end-to-end. The dataset holds two copies of the 3 points; with
    // `val_ratio = 0.0` and `test_ratio = 0.5` the `train` split is exactly the
    // first copy (one sample per class).
    let three_class_dataset = || ModelDataset {
        inputs: array![
            [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
        ], // (2 features, 6 samples)
        targets: array![
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
        ], // (3 classes, 6 samples)
    };

    // Sigmoid avoids the dead-neuron risk of ReLU(0)=0 for the [0.0, 0.0] sample
    // (He init sets biases to zero, so relu([0,0]) = 0 and its gradient is dead at epoch 0).
    let specs = NeuronLayerSpec::plan(LayerPlan::Explicit(vec![8]), 3, &*SIGMOID).unwrap();

    let report = HyperParameters::from_values(
        5_000,
        0,
        None,
        0.05,
        OptimizerConfig::Adam,
        SchedulerConfig::Constant,
        GradientClipping::None,
        LossConfig::CrossEntropy,
        None,
        0.0,
        0.5,
        42,
    )
    .unwrap()
    .build(
        NeuralNetwork::initialization(2, &specs, 42),
        three_class_dataset(),
        Callbacks::new(vec![]),
    )
    .train()
    .unwrap();

    let loss = report.final_evaluation.unwrap().train.loss;
    assert!(
        loss < 0.05,
        "3-class classification did not converge: final loss = {loss:.4}"
    );
}
