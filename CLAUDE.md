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
```

**System dependency**: HDF5 C library is required (`brew install hdf5` on macOS, `sudo apt-get install libhdf5-dev` on Ubuntu).

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
  - `io`: enables HDF5/JSON/image I/O (`hdf5-metno`, `serde`, `image`, `png`, `gif`)
  - `charts`: enables plotting (`plotters`)
- **`cli/`** (crate `nrn-cli`) — `nrn` binary built on top of `core` with `features = ["io", "charts"]`, using `clap` for argument parsing

## Core architecture

### Data layout convention
Arrays use `(features, samples)` shape throughout — columns are samples, rows are features. This is the transpose of the typical scikit-learn convention. `Dataset` (row-major, `(samples, features)`) is converted to `ModelDataset` (column-major) via `to_model_dataset()`.

### Neural network (`core/src/nn/`)

- **`model.rs`**: `NeuralNetwork` (Vec of `NeuronLayer`) and `NeuronLayerSpec`. Key methods: `forward()` returns all intermediate activations as `Vec<Array2<f32>>`; `predict()` returns only the last activation. `NeuronLayerSpec::output_for(n_classes)` auto-selects sigmoid (binary, 2 classes → 1 neuron) or softmax (multi-class).
- **`training.rs`**: `train()` is implemented on `NeuralNetwork` — runs forward + backward + parameter update per epoch (or per mini-batch). `GradientClipping` (None / L2 Norm / Value). `EarlyStopping` with optional best-model restore.
- **`activations/`**: `Activation` trait registered via the `inventory` crate for dynamic lookup by name (`ActivationProvider::get_by_name`). Built-ins: ReLU (He init), Sigmoid (Xavier init), Softmax (Xavier init).
- **`optimizers/`**: `Optimizer` trait with SGD and Adam implementations. Wrapped in `Arc<Mutex<dyn Optimizer>>` to allow shared mutable access during training.
- **`schedulers/`**: `Scheduler` trait stepping once per epoch. Implementations: `ConstantScheduler`, `StepDecay`, `CosineAnnealing` (with optional warm restarts).
- **`loss_functions/`**: Cross-entropy loss (the only loss function; used for both binary and multi-class).

### Data (`core/src/data/`)

- **`dataset.rs`**: `Dataset` (raw, row-major) → `ModelDataset` (training-ready, column-major). `ModelDataset::batches()` shuffles and chunks for mini-batch SGD. `ModelDataset::split()` produces `ModelSplit` (train/val/test) — expects pre-shuffled data.
- **`preprocessors/scalers/`**: `Scaler` trait with `MinMax` and `ZScore` implementations. Scaler params are serialized to JSON for reuse at prediction time.
- **`data/synth/`**: Synthetic dataset generators (`uniform`, `ring` distributions).

### I/O (`core/src/io/`, behind `io` feature)

HDF5 is the primary format for datasets, models, and training checkpoints. Scalers are stored as JSON. The `io` module handles all serialization/deserialization including the activation name → `Arc<dyn Activation>` round-trip via `ActivationProvider::get_by_name`.

### CLI (`cli/src/`)

- **`cli.rs`**: Top-level `clap` command enum dispatching to subcommands.
- **`commands/`**: One file per subcommand (`train`, `predict`, `scale`, `synth`, `encode`, `plot`). `train.rs` contains the full training loop with optimizer/scheduler/clipping wiring.
- **`actions.rs`**: Shared helpers for loading/saving models, datasets, and scalers.
