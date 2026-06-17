# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Educational from-scratch neural network implementation in Rust (edition 2024, MSRV 1.88 — the
code relies on let-chains). Implements feedforward networks (SLP and MLP) with a CLI for the full
ML workflow: data generation → scaling → training → visualization → prediction.

## Commands

The project uses [Task](https://taskfile.dev) (`brew install go-task/tap/go-task`). Run `task`
(no arguments) to list all tasks. `task checks` (lint + build + test) is the pre-commit gate.

Useful extras:

```sh
cargo test <name>               # single test by name, across workspace
cargo test -p nrn <name>        # core crate only
task lint                       # rustfmt --check + clippy (-D warnings), with & without features
task audit                      # cargo-audit advisory scan
task coverage / coverage-html   # cargo-llvm-cov summary / HTML report
```

All serialization is pure Rust — no system C library is needed to build or run.

## Commit conventions

[Conventional Commits](https://conventionalcommits.org). Every message is prefixed with a type,
optionally scoped (`feat(training): …`):

`feat:` feature · `fix:` bug fix · `refactor:` neither · `test:` tests · `docs:` docs only ·
`ci:` CI/CD · `build:` build system / deps · `chore:` everything else.

## Workspace structure

Two crates:

- **`core/`** (crate `nrn`) — neural-network library, no binary. Optional features:
  - `io`: safetensors/JSON/image I/O (`safetensors`, `serde`, `serde_json`, `image`, `png`, `gif`)
  - `charts`: plotting (`plotters`)
  The `nn` module is re-exported flat at the crate root (`pub use nn::*`), so library types live at
  e.g. `nrn::model`, `nrn::training`, `nrn::evaluation_history` — not under `nrn::nn::`.
- **`cli/`** (crate `nrn-cli`) — `nrn` binary over `core` with `features = ["io", "charts"]`, `clap` for args.

## Core architecture

### Data layout convention
Arrays are `(features, samples)` throughout — columns are samples, rows are features (the transpose
of the scikit-learn convention). `Dataset` (row-major, `(samples, features)`) converts to
`ModelDataset` (column-major) via `to_model_dataset()`.

### Neural network (`core/src/nn/`)

- **`model.rs`** — `NeuralNetwork` (Vec of `NeuronLayer`) and `NeuronLayerSpec`. `forward()` returns
  all intermediate activations; `predict()` returns only the last. `NeuronLayerSpec::output_for(n_classes)`
  auto-selects sigmoid (2 classes → 1 neuron) or softmax (multi-class).
- **`training/`** — the training stack, built around a **declarative spec → runtime trainer** split.
  `HyperParameters` is the single source of truth for a run: plain config, no trait objects, cross-field
  invariants validated on construction. Its `build(model, dataset, callbacks)` is the one place
  declarative config becomes concrete — it instantiates the optimizer/scheduler/loss, splits the
  dataset, and returns a `Trainer`. The `Trainer` runs the loop and is **pure**: no I/O, every side
  effect (persistence, display) goes through the `TrainerCallback` trait, composed via
  `Callbacks::empty().with(..).with_opt(..)`. `train()` returns a `TrainingReport` whose
  `into_result()` turns an unrecovered divergence into a `FatalDivergence`. Resuming a run = calling
  `restore(..)` on the built trainer before `train()` (this is what reinstates optimizer/scheduler
  state and the epoch counter).
- **`gradients.rs`** — `Gradients` and `GradientClipping` (None / L2 Norm / Value).
- **`learning_rate.rs`** — `LearningRate` newtype.
- **`activations/`** — `Activation` trait registered via the `inventory` crate for lookup by name
  (`ActivationProvider::get_by_name`). Built-ins: ReLU (He init), Sigmoid / Softmax (Xavier init).
- **`optimizers/`** — `Optimizer` trait: SGD and Adam. Passed to `train()` as `&mut dyn Optimizer`.
- **`schedulers/`** — `Scheduler` trait (steps once per epoch): `ConstantScheduler`, `StepDecay`,
  `CosineAnnealing` (optional warm restarts).
- **`loss_functions/`** — `LossFunction` trait; cross-entropy (used for binary and multi-class).
- **`accuracies/`** — `Accuracy` trait; `accuracy_for(n_classes)` picks binary vs argmax-match.
- **`evaluation.rs`** — `Evaluation` (loss + accuracy for one split) and `EvaluationSet`
  (train / optional validation / test).
- **`evaluation_history.rs`** — `EvaluationHistory(Vec<EpochEvaluation { epoch, evaluation }>)`: pure
  value object, ordered by epoch; exposes per-split loss/accuracy series, ranges, and `epochs()`
  (chart X-axis).
- **`initializations/`** — `he` / `xavier` weight initializers, selected per activation.

### Data (`core/src/data/`)

- **`dataset.rs`** — `Dataset` (raw) → `ModelDataset` (column-major). `batches()` shuffles and chunks
  for mini-batch SGD; `split()` produces `ModelSplit` (train/val/test) from pre-shuffled data.
- **`preprocessors/scalers/`** — `Scaler` trait (`MinMax`, `ZScore`); `ScalerMethod` enum dispatches
  for CLI/serialization. Params serialized to JSON for reuse at prediction time.
- **`preprocessors/vectorizers/`** — image → feature-vector flattening (behind `io`).
- **`synth/`** — synthetic generators (`uniform`, `ring`).

### Analysis (`core/src/analysis/`)

- **`boundary.rs`** — `decision_boundary()` samples a grid over the input bounds and returns points
  near the threshold (binary ≈ 0.5; multi-class: top two class probabilities near-equal). Pure.

### Charts (`core/src/charts/`, behind `charts` feature)

`plotters`-based rendering to in-memory RGB buffers. `RenderConfig` holds dimensions/fonts/padding;
`evaluation_history.rs` draws training curves, `dataset.rs` draws scatter + decision boundary. Pure rendering —
the caller persists the bytes.

### I/O (`core/src/io/`, behind `io` feature)

[safetensors](https://github.com/huggingface/safetensors) is the primary format for datasets, models,
and checkpoints (`f32` tensors plus a `__metadata__` string map); scalers and run hyperparameters are
JSON. The guiding rule: **core runtime types stay serde-free**, so `io` owns their serializable mirrors
— `HyperParametersRecord` for `HyperParameters`, plus the optimizer/scheduler `*State` round-trips.
`io/tensors.rs` holds the shared `View` adapter; the activation-name → `Arc<dyn Activation>` round-trip
goes through `ActivationProvider::get_by_name`.

A run lives in a directory managed by `TrainingRun`: `create`/`open` persist run-level `TrainingMeta`
(`meta.json`), `trim_after` rewinds the trajectory (removing later checkpoints), `archive()` reads it
back. Each `checkpoint-{epoch:06}/` is written by a `CheckpointRecorder` (the `TrainerCallback` from
`TrainingRun::recorder`) and read by a `CheckpointArchive` (`model_at` / `optimizer_at` / `epoch_at` /
`evaluation_history`).

### CLI (`cli/src/`)

- **`cli.rs`** — top-level `clap` command enum dispatching to subcommands.
- **`commands/`** — one module per subcommand (`synth`, `encode`, `scale`, `predict`, `plot`), plus the
  `train/` group (`train start` / `train resume`). The CLI's job is to parse args into a core
  `HyperParameters` spec (via `TryFrom`, in `train/args.rs`) and run it: `execute_training()` composes
  the callbacks, calls `build(..)`, optionally `restore(..)`s for resume, then `train()`. The
  console-facing callbacks (`ModelSaver`, `ConsoleMonitor` with its progress bar) are `TrainerCallback`
  impls under `train/`.
- **`actions.rs`** — shared load/save helpers for models, datasets, and scalers.
- **`console.rs`** — display helpers: status icons, `Summary`, formatted output used across commands.
