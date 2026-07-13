//! Performance benchmarks for the network hot paths: inference (`forward`/`predict`)
//! and a single training epoch (full-batch and mini-batch SGD).
//!
//! Three workloads bracket the realistic range:
//! - `small`: a 2-feature binary MLP (the synthetic-dataset CLI demos).
//! - `mnist`: a 784-feature, 10-class MLP with two hidden layers.
//! - `cnn`: a 1×28×28, 10-class convolutional net (Conv2d → Flatten → Dense),
//!   exercising the spatial `im2col`/`col2im` paths on a rank-4 batch.
//!
//! Each epoch benchmark clones the model in `iter_batched`'s setup so the clone
//! is excluded from the measured region and every iteration starts from the same
//! weights (training mutates them in place).

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use ndarray::{Array2, Array4};
use nrn::activations::{IDENTITY, RELU};
use nrn::data::ModelDataset;
use nrn::gradients::GradientClipping;
use nrn::learning_rate::LearningRate;
use nrn::loss_functions::{BinaryCrossEntropy, CategoricalCrossEntropy, LossFunction, Reduction};
use nrn::model::{NetworkConfig, NeuralNetwork};
use nrn::optimizers::{Adam, Optimizer};
use nrn::schedulers::{ConstantScheduler, Scheduler};
use nrn::training::MiniBatch;
use nrn::weight_decay::WeightDecay;
use std::hint::black_box;
use std::sync::Arc;

/// A reproducible workload: a network paired with a column-major dataset and the
/// loss its task calls for (binary vs categorical cross-entropy).
struct Workload {
    model: NeuralNetwork,
    dataset: ModelDataset,
    loss: Arc<dyn LossFunction>,
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

/// A one-hot (multi-class) or binary `(target_rows, samples)` target matrix whose labels
/// cycle through the classes, so every class is represented.
fn make_targets(n_classes: usize, n_samples: usize) -> Array2<f32> {
    let target_rows = if n_classes == 2 { 1 } else { n_classes };
    Array2::from_shape_fn((target_rows, n_samples), |(r, c)| {
        let label = c % n_classes;
        if n_classes == 2 {
            label as f32
        } else if r == label {
            1.0
        } else {
            0.0
        }
    })
}

/// Builds a `(features, samples)` input matrix paired with matching targets.
fn make_dataset(n_features: usize, n_classes: usize, n_samples: usize) -> ModelDataset {
    let inputs = pseudo_random((n_features, n_samples), 0);
    ModelDataset::new(inputs, make_targets(n_classes, n_samples))
}

/// Builds a `(channels, height, width, samples)` spatial batch (samples-last, the
/// convention a leading `Conv2d` consumes) paired with matching targets.
fn make_spatial_dataset(
    channels: usize,
    height: usize,
    width: usize,
    n_classes: usize,
    n_samples: usize,
) -> ModelDataset {
    let inputs = Array4::from_shape_fn((channels, height, width, n_samples), |(ch, h, w, s)| {
        let n = (ch as u64).wrapping_mul(73_856_093)
            ^ (h as u64).wrapping_mul(19_349_663)
            ^ (w as u64).wrapping_mul(83_492_791)
            ^ (s as u64);
        ((n % 2000) as f32) / 1000.0 - 1.0
    });
    ModelDataset::new(inputs, make_targets(n_classes, n_samples))
}

fn small_workload() -> Workload {
    let config = NetworkConfig::builder(vec![2])
        .dense(16, &RELU)
        .dense(1, &IDENTITY)
        .build();
    Workload {
        model: NeuralNetwork::from_config(config, 42).unwrap(),
        dataset: make_dataset(2, 2, 1_000),
        loss: Arc::new(BinaryCrossEntropy::new(Reduction::Mean)),
        batch_size: 64,
    }
}

fn mnist_workload() -> Workload {
    let config = NetworkConfig::builder(vec![784])
        .dense(128, &RELU)
        .dense(64, &RELU)
        .dense(10, &IDENTITY)
        .build();
    Workload {
        model: NeuralNetwork::from_config(config, 42).unwrap(),
        dataset: make_dataset(784, 10, 2_000),
        loss: Arc::new(CategoricalCrossEntropy::new(Reduction::Mean)),
        batch_size: 64,
    }
}

fn cnn_workload() -> Workload {
    // 1×28×28 → Conv2d(16 filters, 3×3, stride 1) → 16×26×26 → Flatten → Dense(10, logits).
    let config = NetworkConfig::builder(vec![1, 28, 28])
        .conv2d(16, (3, 3), 1, 0, &RELU)
        .flatten()
        .dense(10, &IDENTITY)
        .build();
    Workload {
        model: NeuralNetwork::from_config(config, 42).unwrap(),
        dataset: make_spatial_dataset(1, 28, 28, 10, 1_000),
        loss: Arc::new(CategoricalCrossEntropy::new(Reduction::Mean)),
        batch_size: 64,
    }
}

/// Runs one training epoch on `model` — the unit the epoch benchmarks measure.
fn run_epoch(model: &mut NeuralNetwork, w: &Workload, mini_batch: Option<MiniBatch>) {
    let mut optimizer = Adam::with_defaults(LearningRate::new(0.01).unwrap(), WeightDecay::ZERO);
    let mut scheduler = ConstantScheduler::new(LearningRate::new(0.01).unwrap());
    model
        .train(
            &w.dataset,
            &w.loss,
            &mut optimizer as &mut dyn Optimizer,
            &mut scheduler as &mut dyn Scheduler,
            &GradientClipping::None,
            mini_batch,
        )
        .unwrap();
}

fn bench_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("inference");
    for (name, w) in [
        ("small", small_workload()),
        ("mnist", mnist_workload()),
        ("cnn", cnn_workload()),
    ] {
        group.bench_function(name, |b| {
            b.iter(|| black_box(w.model.output(w.dataset.inputs().view())));
        });
    }
    group.finish();
}

fn bench_epoch_full_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("epoch_full_batch");
    for (name, w) in [
        ("small", small_workload()),
        ("mnist", mnist_workload()),
        ("cnn", cnn_workload()),
    ] {
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
    for (name, w) in [
        ("small", small_workload()),
        ("mnist", mnist_workload()),
        ("cnn", cnn_workload()),
    ] {
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
