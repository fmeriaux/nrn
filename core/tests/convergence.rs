use ndarray::array;
use nrn::activations::SIGMOID;
use nrn::data::{ModelDataset, ModelSplit};
use nrn::loss_functions::CROSS_ENTROPY_LOSS;
use nrn::model::{NeuralNetwork, NeuronLayerSpec};
use nrn::optimizers::Adam;
use nrn::schedulers::ConstantScheduler;
use nrn::training::{Callbacks, GradientClipping, HyperParams, LearningRate, TrainingLoop};

#[test]
fn xor_converges_to_low_loss() {
    // XOR: non-linearly separable, requires at least one hidden layer
    let xor_dataset = || ModelDataset {
        inputs: array![[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0]], // (2 features, 4 samples)
        targets: array![[0.0, 1.0, 1.0, 0.0]],                      // (1 output, 4 samples)
    };

    let specs = NeuronLayerSpec::network_for(vec![8], &*SIGMOID, 2);

    let report = TrainingLoop {
        model: NeuralNetwork::initialization(2, &specs),
        callbacks: Callbacks::new(vec![]),
        split: ModelSplit {
            train: xor_dataset(),
            validation: None,
            test: xor_dataset(),
        },
        hyperparams: HyperParams {
            epochs: 10_000,
            checkpoint_interval: 0,
            batch_size: None,
            loss: CROSS_ENTROPY_LOSS.clone(),
            optimizer: Box::new(Adam::with_defaults(LearningRate::new(0.1).unwrap())),
            scheduler: Box::new(ConstantScheduler::new(LearningRate::new(0.1).unwrap())),
            clipping: GradientClipping::None,
            early_stopping: None,
            val_ratio: 0.1,
            test_ratio: 0.1,
        },
        epoch_start: 0,
    }
    .run()
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

    let report = TrainingLoop {
        model: NeuralNetwork::initialization(2, &specs),
        callbacks: Callbacks::new(vec![]),
        split: ModelSplit {
            train: xor_dataset(),
            validation: None,
            test: xor_dataset(),
        },
        hyperparams: HyperParams {
            epochs: 8_000,
            checkpoint_interval: 0,
            batch_size: Some(2),
            loss: CROSS_ENTROPY_LOSS.clone(),
            optimizer: Box::new(Adam::with_defaults(LearningRate::new(0.01).unwrap())),
            scheduler: Box::new(ConstantScheduler::new(LearningRate::new(0.01).unwrap())),
            clipping: GradientClipping::None,
            early_stopping: None,
            val_ratio: 0.1,
            test_ratio: 0.1,
        },
        epoch_start: 0,
    }
    .run()
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

    let report = TrainingLoop {
        model: NeuralNetwork::initialization(2, &specs),
        callbacks: Callbacks::new(vec![]),
        split: ModelSplit {
            train: three_class_dataset(),
            validation: None,
            test: three_class_dataset(),
        },
        hyperparams: HyperParams {
            epochs: 5_000,
            checkpoint_interval: 0,
            batch_size: None,
            loss: CROSS_ENTROPY_LOSS.clone(),
            optimizer: Box::new(Adam::with_defaults(LearningRate::new(0.05).unwrap())),
            scheduler: Box::new(ConstantScheduler::new(LearningRate::new(0.05).unwrap())),
            clipping: GradientClipping::None,
            early_stopping: None,
            val_ratio: 0.1,
            test_ratio: 0.1,
        },
        epoch_start: 0,
    }
    .run()
    .unwrap();

    let loss = report.final_evaluation.unwrap().train.loss;
    assert!(
        loss < 0.05,
        "3-class classification did not converge: final loss = {loss:.4}"
    );
}
