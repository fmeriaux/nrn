use ndarray::array;
use nrn::activations::SIGMOID;
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
        targets: array![[0.0, 1.0, 1.0, 0.0]],                        // (1 output, 4 samples)
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
        model.train(&dataset, &loss_fn, &optimizer, &scheduler, &clipping);
    }

    let predictions = model.predict(dataset.inputs.view());
    let loss = loss_fn.compute(predictions.view(), dataset.targets.view());

    assert!(loss < 0.05, "XOR did not converge: final loss = {loss:.4}");
}
