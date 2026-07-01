//! Performance benchmarks for the MLP hot paths: inference (`forward`/`predict`)
//! and a single training epoch (full-batch and mini-batch SGD).
//!
//! Two workloads bracket the realistic range:
//! - `small`: a 2-feature binary problem (the synthetic-dataset CLI demos).
//! - `mnist`: a 784-feature, 10-class problem with two hidden layers.
//!
//! Each epoch benchmark clones the model in `iter_batched`'s setup so the clone
//! is excluded from the measured region and every iteration starts from the same
//! weights (training mutates them in place).

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use ndarray::Array2;
use nrn::activations::{RELU, SIGMOID, SOFTMAX};
use nrn::data::ModelDataset;
use nrn::gradients::GradientClipping;
use nrn::learning_rate::LearningRate;
use nrn::loss_functions::{CROSS_ENTROPY_LOSS, LossFunction};
use nrn::model::{NeuralNetwork, NeuronLayerSpec};
use nrn::optimizers::{Adam, Optimizer};
use nrn::schedulers::{ConstantScheduler, Scheduler};
use nrn::training::MiniBatch;
use nrn::weight_decay::WeightDecay;
use std::hint::black_box;
use std::sync::Arc;

/// A reproducible workload: a network paired with a column-major dataset.
struct Workload {
    model: NeuralNetwork,
    dataset: ModelDataset,
    batch_size: usize,
}

/// Deterministic pseudo-random fill in `[-1, 1)` — avoids pulling an RNG into the
/// hot-path setup while still giving the matmuls non-degenerate inputs.
fn pseudo_random(shape: (usize, usize), mix: u64) -> Array2<f32> {
    Array2::from_shape_fn(shape, |(r, c)| {
        let n = (r as u64).wrapping_mul(73_856_093) ^ (c as u64).wrapping_mul(19_349_663) ^ mix;
        ((n % 2000) as f32) / 1000.0 - 1.0
    })
}

/// Builds a `(features, samples)` input matrix and a one-hot / binary target matrix.
fn make_dataset(n_features: usize, n_classes: usize, n_samples: usize) -> ModelDataset {
    let inputs = pseudo_random((n_features, n_samples), 0);
    let target_rows = if n_classes == 2 { 1 } else { n_classes };
    let targets = Array2::from_shape_fn((target_rows, n_samples), |(r, c)| {
        let label = c % n_classes;
        if n_classes == 2 {
            label as f32
        } else if r == label {
            1.0
        } else {
            0.0
        }
    });
    ModelDataset { inputs, targets }
}

fn small_workload() -> Workload {
    let specs = vec![
        NeuronLayerSpec {
            neurons: 16,
            activation: RELU.clone(),
        },
        NeuronLayerSpec {
            neurons: 1,
            activation: SIGMOID.clone(),
        },
    ];
    Workload {
        model: NeuralNetwork::initialization(2, &specs, 42),
        dataset: make_dataset(2, 2, 1_000),
        batch_size: 64,
    }
}

fn mnist_workload() -> Workload {
    let specs = vec![
        NeuronLayerSpec {
            neurons: 128,
            activation: RELU.clone(),
        },
        NeuronLayerSpec {
            neurons: 64,
            activation: RELU.clone(),
        },
        NeuronLayerSpec {
            neurons: 10,
            activation: SOFTMAX.clone(),
        },
    ];
    Workload {
        model: NeuralNetwork::initialization(784, &specs, 42),
        dataset: make_dataset(784, 10, 2_000),
        batch_size: 64,
    }
}

fn loss() -> Arc<dyn LossFunction> {
    CROSS_ENTROPY_LOSS.clone()
}

/// Runs one training epoch on `model` — the unit the epoch benchmarks measure.
fn run_epoch(model: &mut NeuralNetwork, w: &Workload, mini_batch: Option<MiniBatch>) {
    let mut optimizer = Adam::with_defaults(LearningRate::new(0.01).unwrap(), WeightDecay::ZERO);
    let mut scheduler = ConstantScheduler::new(LearningRate::new(0.01).unwrap());
    model
        .train(
            &w.dataset,
            &loss(),
            &mut optimizer as &mut dyn Optimizer,
            &mut scheduler as &mut dyn Scheduler,
            &GradientClipping::None,
            mini_batch,
        )
        .unwrap();
}

fn bench_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("inference");
    for (name, w) in [("small", small_workload()), ("mnist", mnist_workload())] {
        group.bench_function(name, |b| {
            b.iter(|| black_box(w.model.predict(w.dataset.inputs.view())));
        });
    }
    group.finish();
}

fn bench_epoch_full_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("epoch_full_batch");
    for (name, w) in [("small", small_workload()), ("mnist", mnist_workload())] {
        group.bench_function(name, |b| {
            b.iter_batched(
                || w.model.clone(),
                |mut model| run_epoch(&mut model, &w, None),
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn bench_epoch_mini_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("epoch_mini_batch");
    for (name, w) in [("small", small_workload()), ("mnist", mnist_workload())] {
        group.bench_function(name, |b| {
            b.iter_batched(
                || w.model.clone(),
                |mut model| run_epoch(&mut model, &w, Some(MiniBatch::new(w.batch_size, 42, 0))),
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_inference,
    bench_epoch_full_batch,
    bench_epoch_mini_batch
);
criterion_main!(benches);
