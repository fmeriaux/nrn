# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Educational from-scratch neural network implementation in Rust (edition 2024). Implements feedforward networks (SLP and MLP) with a CLI for the full ML workflow: data generation → scaling → training → visualization → prediction.

## Commands

The project uses [Task](https://taskfile.dev) (`brew install go-task/tap/go-task`). Run `task` (no arguments) to list all available tasks.

Key extras not covered by the Taskfile:

```sh
cargo test <test_name>          # single test by name, across workspace
cargo test -p nrn <test_name>   # core crate only
task coverage                   # core coverage summary (needs cargo-llvm-cov)
task coverage-html              # HTML coverage report, opened in a browser
```

All serialization formats are pure Rust — no system C library is required to build or run.

## Commit conventions

This project follows [Conventional Commits](https://conventionalcommits.org). Every commit message must be prefixed with a type:

| Type | When to use |
|------|-------------|
| `feat:` | new feature or capability |
| `fix:` | bug fix |
| `refactor:` | code change that is neither a feature nor a fix |
| `test:` | adding or updating tests |
| `docs:` | documentation only |
| `ci:` | CI/CD configuration |
| `build:` | build system or dependencies (e.g. Cargo.toml) |
| `chore:` | everything else (tooling, config files, housekeeping) |

A scope can be added in parentheses: `feat(training): add cosine scheduler`.

## Workspace structure

Two crates:

- **`core/`** (crate `nrn`) — neural network library, no binary. Two optional feature flags:
  - `io`: enables safetensors/JSON/image I/O (`safetensors`, `serde`, `serde_json`, `image`, `png`, `gif`)
  - `charts`: enables plotting (`plotters`)
- **`cli/`** (crate `nrn-cli`) — `nrn` binary built on top of `core` with `features = ["io", "charts"]`, using `clap` for argument parsing

## Core architecture

### Data layout convention
Arrays use `(features, samples)` shape throughout — columns are samples, rows are features. This is the transpose of the typical scikit-learn convention. `Dataset` (row-major, `(samples, features)`) is converted to `ModelDataset` (column-major) via `to_model_dataset()`.

### Neural network (`core/src/nn/`)

- **`model.rs`**: `NeuralNetwork` (Vec of `NeuronLayer`) and `NeuronLayerSpec`. Key methods: `forward()` returns all intermediate activations as `Vec<Array2<f32>>`; `predict()` returns only the last activation. `NeuronLayerSpec::output_for(n_classes)` auto-selects sigmoid (binary, 2 classes → 1 neuron) or softmax (multi-class).
- **`training.rs`**: `train()` is implemented on `NeuralNetwork` — runs forward + backward + parameter update per epoch (or per mini-batch). `GradientClipping` (None / L2 Norm / Value). `EarlyStopping` with optional best-model restore.
- **`activations/`**: `Activation` trait registered via the `inventory` crate for dynamic lookup by name (`ActivationProvider::get_by_name`). Built-ins: ReLU (He init), Sigmoid (Xavier init), Softmax (Xavier init).
- **`optimizers/`**: `Optimizer` trait with SGD and Adam implementations. Passed to `train()` as `&mut dyn Optimizer`.
- **`schedulers/`**: `Scheduler` trait stepping once per epoch, passed as `&mut dyn Scheduler`. Implementations: `ConstantScheduler`, `StepDecay`, `CosineAnnealing` (with optional warm restarts).
- **`loss_functions/`**: `LossFunction` trait; cross-entropy is the only implementation (used for both binary and multi-class).
- **`accuracies/`**: `Accuracy` trait. `accuracy_for(n_classes)` picks `BINARY_ACCURACY` (2 classes) or `MULTI_CLASS_ACCURACY` (argmax match).
- **`evaluation.rs`**: `Evaluation` (loss + accuracy for one dataset) and `EvaluationSet` (train / optional validation / test), computed from a model or from precomputed predictions.
- **`checkpoints.rs`**: `Checkpoints` records model snapshots and `EvaluationSet`s at a fixed epoch interval; exposes per-split loss/accuracy series and their ranges for plotting.
- **`initializations/`**: weight initializers (`he`, `xavier`) selected per activation.

### Data (`core/src/data/`)

- **`dataset.rs`**: `Dataset` (raw, row-major) → `ModelDataset` (training-ready, column-major). `ModelDataset::batches()` shuffles and chunks for mini-batch SGD. `ModelDataset::split()` produces `ModelSplit` (train/val/test) — expects pre-shuffled data.
- **`preprocessors/scalers/`**: `Scaler` trait with `MinMax` and `ZScore` implementations; `ScalerMethod` enum dispatches to them for CLI/serialization. Scaler params are serialized to JSON for reuse at prediction time.
- **`preprocessors/vectorizers/`**: flattens images into feature vectors (behind the `io` feature).
- **`data/synth/`**: Synthetic dataset generators (`uniform`, `ring` distributions).

### Analysis (`core/src/analysis/`)

- **`boundary.rs`**: `decision_boundary()` samples a grid over the input bounds and returns the points near the decision threshold (binary: prediction ≈ 0.5; multi-class: top two class probabilities nearly equal). Pure computation — no plotting.

### I/O (`core/src/io/`, behind `io` feature)

[safetensors](https://github.com/huggingface/safetensors) is the primary format for datasets, models, and training checkpoints: `f32` tensors plus a `__metadata__` string map (activation names, intervals, snapshot counts). Checkpoint evaluation series are stored as `f32` tensors too. Scalers are stored as JSON. `io/tensors.rs` holds the shared `View` adapter and (de)serialization helpers; the `io` module handles the activation name → `Arc<dyn Activation>` round-trip via `ActivationProvider::get_by_name`.

### CLI (`cli/src/`)

- **`cli.rs`**: Top-level `clap` command enum dispatching to subcommands.
- **`commands/`**: One file per subcommand (`train`, `predict`, `scale`, `synth`, `encode`, `plot`). `train.rs` contains the full training loop with optimizer/scheduler/clipping wiring.
- **`actions.rs`**: Shared helpers for loading/saving models, datasets, and scalers.
