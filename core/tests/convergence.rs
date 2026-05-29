use ndarray::array;
use nrn::activations::{RELU, SIGMOID};
use nrn::data::ModelDataset;
use nrn::loss_functions::{CROSS_ENTROPY_LOSS, LossFunction};
use nrn::model::{NeuralNetwork, NeuronLayerSpec};
use nrn::optimizers::{Adam, Optimizer};
use nrn::schedulers::{ConstantScheduler, Scheduler};
use nrn::training::{GradientClipping, LearningRate};
use std::sync::{Arc, Mutex};

#[test]
fn xor_converges_to_low_loss() {
    // XOR: non-linearly separable, requires at least one hidden layer
    let dataset = ModelDataset {
        inputs: array![[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0]], // (2 features, 4 samples)
        targets: array![[0.0, 1.0, 1.0, 0.0]],                      // (1 output, 4 samples)
    };

    let specs = NeuronLayerSpec::network_for(vec![8], &*SIGMOID, 2);
    let mut model = NeuralNetwork::initialization(2, &specs);

    let loss_fn: Arc<dyn LossFunction> = CROSS_ENTROPY_LOSS.clone();
    let optimizer: Arc<Mutex<dyn Optimizer>> =
        Arc::new(Mutex::new(Adam::with_defaults(LearningRate::new(0.1))));
    let scheduler: Arc<Mutex<dyn Scheduler>> =
        Arc::new(Mutex::new(ConstantScheduler::new(LearningRate::new(0.1))));
    let clipping = GradientClipping::None;

    for _ in 0..10_000 {
        model.train(&dataset, &loss_fn, &optimizer, &scheduler, &clipping, None);
    }

    let predictions = model.predict(dataset.inputs.view());
    let loss = loss_fn.compute(predictions.view(), dataset.targets.view());

    assert!(loss < 0.05, "XOR did not converge: final loss = {loss:.4}");
}

#[test]
fn xor_converges_with_mini_batch() {
    // Mini-batch of 2 on a 4-sample XOR dataset (2 batches per epoch, shuffled).
    // Adam's v → 0 after convergence can cause NaN if training continues too long without
    // lr decay, so we stop at 8 000 epochs where loss is reliably < 0.01.
    let dataset = ModelDataset {
        inputs: array![[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0]],
        targets: array![[0.0, 1.0, 1.0, 0.0]],
    };

    let specs = NeuronLayerSpec::network_for(vec![8], &*SIGMOID, 2);
    let mut model = NeuralNetwork::initialization(2, &specs);

    let loss_fn: Arc<dyn LossFunction> = CROSS_ENTROPY_LOSS.clone();
    let optimizer: Arc<Mutex<dyn Optimizer>> =
        Arc::new(Mutex::new(Adam::with_defaults(LearningRate::new(0.01))));
    let scheduler: Arc<Mutex<dyn Scheduler>> =
        Arc::new(Mutex::new(ConstantScheduler::new(LearningRate::new(0.01))));
    let clipping = GradientClipping::None;

    for _ in 0..8_000 {
        model.train(
            &dataset,
            &loss_fn,
            &optimizer,
            &scheduler,
            &clipping,
            Some(2),
        );
    }

    let predictions = model.predict(dataset.inputs.view());
    let loss = loss_fn.compute(predictions.view(), dataset.targets.view());

    assert!(
        loss < 0.05,
        "XOR (mini-batch) did not converge: final loss = {loss:.4}"
    );
}

#[test]
fn three_class_converges_to_low_loss() {
    // 3 linearly separable points, one per class.
    // Exercises the full softmax output path end-to-end.
    let dataset = ModelDataset {
        inputs: array![[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], // (2 features, 3 samples)
        targets: array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], // (3 classes, 3 samples)
    };

    let specs = NeuronLayerSpec::network_for(vec![8], &*RELU, 3);
    let mut model = NeuralNetwork::initialization(2, &specs);

    let loss_fn: Arc<dyn LossFunction> = CROSS_ENTROPY_LOSS.clone();
    let optimizer: Arc<Mutex<dyn Optimizer>> =
        Arc::new(Mutex::new(Adam::with_defaults(LearningRate::new(0.05))));
    let scheduler: Arc<Mutex<dyn Scheduler>> =
        Arc::new(Mutex::new(ConstantScheduler::new(LearningRate::new(0.05))));
    let clipping = GradientClipping::None;

    for _ in 0..5_000 {
        model.train(&dataset, &loss_fn, &optimizer, &scheduler, &clipping, None);
    }

    let predictions = model.predict(dataset.inputs.view());
    let loss = loss_fn.compute(predictions.view(), dataset.targets.view());

    assert!(
        loss < 0.05,
        "3-class classification did not converge: final loss = {loss:.4}"
    );
}
