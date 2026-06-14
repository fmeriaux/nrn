use ndarray::array;
use nrn::activations::SIGMOID;
use nrn::data::{ModelDataset, ModelSplit};
use nrn::model::{NeuralNetwork, NeuronLayerSpec};
use nrn::training::{
    Callbacks, GradientClipping, HyperParameters, LearningRate, LossConfig, OptimizerConfig,
    SchedulerConfig,
};

#[test]
fn xor_converges_to_low_loss() {
    // XOR: non-linearly separable, requires at least one hidden layer
    let xor_dataset = || ModelDataset {
        inputs: array![[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0]], // (2 features, 4 samples)
        targets: array![[0.0, 1.0, 1.0, 0.0]],                      // (1 output, 4 samples)
    };

    let specs = NeuronLayerSpec::network_for(vec![8], &*SIGMOID, 2);

    let report = HyperParameters::new(
        10_000,
        0,
        None,
        LearningRate::new(0.1).unwrap(),
        OptimizerConfig::Adam,
        SchedulerConfig::Constant,
        GradientClipping::None,
        LossConfig::CrossEntropy,
        None,
        0.1,
        0.1,
    )
    .unwrap()
    .build(
        NeuralNetwork::initialization(2, &specs),
        ModelSplit {
            train: xor_dataset(),
            validation: None,
            test: xor_dataset(),
        },
        Callbacks::new(vec![]),
        0,
    )
    .train()
    .unwrap();

    let loss = report.final_evaluation.unwrap().train.loss;
    assert!(loss < 0.05, "XOR did not converge: final loss = {loss:.4}");
}

#[test]
fn xor_converges_with_mini_batch() {
    // Mini-batch of 2 on a 4-sample XOR dataset (2 batches per epoch, shuffled).
    // Adam's v → 0 after convergence can cause NaN if training continues too long without
    // lr decay, so we stop at 8 000 epochs where loss is reliably < 0.01.
    let xor_dataset = || ModelDataset {
        inputs: array![[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0]],
        targets: array![[0.0, 1.0, 1.0, 0.0]],
    };

    let specs = NeuronLayerSpec::network_for(vec![8], &*SIGMOID, 2);

    let report = HyperParameters::new(
        8_000,
        0,
        Some(2),
        LearningRate::new(0.01).unwrap(),
        OptimizerConfig::Adam,
        SchedulerConfig::Constant,
        GradientClipping::None,
        LossConfig::CrossEntropy,
        None,
        0.1,
        0.1,
    )
    .unwrap()
    .build(
        NeuralNetwork::initialization(2, &specs),
        ModelSplit {
            train: xor_dataset(),
            validation: None,
            test: xor_dataset(),
        },
        Callbacks::new(vec![]),
        0,
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
    // 3 linearly separable points, one per class.
    // Exercises the full softmax output path end-to-end.
    let three_class_dataset = || ModelDataset {
        inputs: array![[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], // (2 features, 3 samples)
        targets: array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], // (3 classes, 3 samples)
    };

    // Sigmoid avoids the dead-neuron risk of ReLU(0)=0 for the [0.0, 0.0] sample
    // (He init sets biases to zero, so relu([0,0]) = 0 and its gradient is dead at epoch 0).
    let specs = NeuronLayerSpec::network_for(vec![8], &*SIGMOID, 3);

    let report = HyperParameters::new(
        5_000,
        0,
        None,
        LearningRate::new(0.05).unwrap(),
        OptimizerConfig::Adam,
        SchedulerConfig::Constant,
        GradientClipping::None,
        LossConfig::CrossEntropy,
        None,
        0.1,
        0.1,
    )
    .unwrap()
    .build(
        NeuralNetwork::initialization(2, &specs),
        ModelSplit {
            train: three_class_dataset(),
            validation: None,
            test: three_class_dataset(),
        },
        Callbacks::new(vec![]),
        0,
    )
    .train()
    .unwrap();

    let loss = report.final_evaluation.unwrap().train.loss;
    assert!(
        loss < 0.05,
        "3-class classification did not converge: final loss = {loss:.4}"
    );
}
