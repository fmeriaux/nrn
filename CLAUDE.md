# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Educational from-scratch neural network implementation in Rust (edition 2024, MSRV 1.88 — the
code relies on let-chains). Implements feedforward networks (SLP and MLP) with a CLI for the full
ML workflow: data generation/encoding → training (with optional scaling) → visualization → prediction.

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
task coverage-check             # fail under the line-coverage threshold (CI gate)
```

All serialization is pure Rust — no system C library is needed to build or run.

## Testing

New public behavior ships with a test; every bug fix ships with a regression test that fails
before the fix. Unit tests live in-module under `#[cfg(test)]`; CLI behavior is covered
end-to-end through `assert_cmd` in `cli/tests/`. CI gates merges on a line-coverage threshold
(`task coverage-check`) — keep coverage from regressing.

## Commit conventions

[Conventional Commits](https://conventionalcommits.org). Every message is prefixed with a type,
optionally scoped (`feat(training): …`):

`feat:` feature · `fix:` bug fix · `refactor:` neither · `test:` tests · `docs:` docs only ·
`ci:` CI/CD · `build:` build system / deps · `chore:` everything else.

## Workspace structure

Two crates:

- **`core/`** (crate `nrn`) — neural-network library, no binary. Optional features:
  - `io`: safetensors/JSON/image I/O (`safetensors`, `serde`, `serde_json`, `image`, `png`, `gif`)
  - `raster`: image rendering (`plotters`)
  - `console`: terminal rendering (`textplots`, `rgb`)
  - `blas`: route ndarray's matmul through a system BLAS (`sgemm`, backend picked per OS)
    instead of the pure-Rust default; ~8× on matmul-heavy paths. See `docs/benchmarks.md`.
  The `nn` module is re-exported flat at the crate root (`pub use nn::*`), so library types live at
  e.g. `nrn::model`, `nrn::training`, `nrn::evaluation_history` — not under `nrn::nn::`.
- **`cli/`** (crate `nrn-cli`) — `nrn` binary over `core` with `features = ["io", "raster", "console"]`, `clap` for args.

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
  for mini-batch SGD; `split()` shuffles (seeded) then partitions into `ModelSplit` (train/val/test),
  so producers store data in natural order and the run seed governs the partition.
- **`preprocessors/scalers/`** — `Scaler` trait (`MinMax`, `ZScore`); `ScalerKind` names the method,
  `ScalerKind::fit` turns it into a fitted `ScalerMethod` (the dispatch enum carrying parameters).
  Params serialized to JSON for reuse at prediction time.
- **`preprocessors/vectorizers/`** — image → feature-vector flattening (behind `io`).
- **`synth/`** — synthetic generators (`uniform`, `ring`).

### Analysis (`core/src/analysis/`)

- **`boundary.rs`** — `Predictor::decision_boundary(mins, maxs, resolution)` marches a grid over the
  input bounds and emits interpolated points where the predicted class flips (binary crossing of 0.5;
  multi-class argmax change). Pure; `resolution` controls smoothness/cost, not correctness (no tolerance).

### Plot (`core/src/plot/`)

A backend-neutral figure IR with feature-gated renderers, in three stages:

- **`scene.rs`** (always compiled) — the pure IR: `Figure` (vertically stacked `Panel`s), `Series`
  (`Line` / `Points`), `Color` (tab10 palette + role constants, including `POSITIVE`/`NEGATIVE` and a
  `scaled` dimmer). No rendering commitment.
- **`build.rs`** (always compiled) — derives figures from domain objects: `Dataset::figure`,
  `Predictor::boundary_figure`, `EvaluationHistory::figure` (each with a `*_with_padding` override;
  the default padding lives here). `n_features != 2` becomes an `Err`.
- **`activations.rs`** (always compiled) — a *separate* IR from `Figure`, for a single instance's
  forward pass rather than a chart: `ActivationDiagram` (per-layer `DiagramLayer`s of colored `Unit`s
  and weighted `Edge`s, plus the `Classification`). `NeuralNetwork::activation_diagram` /
  `Predictor::activation_diagram` (the latter scales the input first) build it; large layers are
  sampled evenly to `DiagramOptions::max_units` and weak edges pruned below `min_edge_magnitude`.
  `Unit::marker_color` (sign → blue/orange, dimmed by intensity) and `DiagramLayer::heading` are shared
  by both renderers.
- **`image/`** (`raster`, `plotters`) and **`console/`** (`console`, `textplots`) — the two renderers,
  each a folder split by IR: `mod.rs` owns the shared config (`ImageConfig` / `ConsoleConfig`) and
  color helpers, `figure.rs` renders `Figure`, `diagram.rs` renders `ActivationDiagram`. `Figure::to_image`
  produces a `RasterImage`, and `ActivationDiagram::to_image` draws a horizontal node-link graph (nodes
  tinted by activation with input/output values annotated, edges colored by weight sign with width/opacity
  by magnitude, plus a legend). `Figure::to_console` produces a `String`, and `ActivationDiagram::to_console`
  lists the layers vertically (nodes only, no edges). Pure rendering — the caller persists the bytes (`io`)
  or prints the text. Animations are streamed frame-by-frame to a GIF by `io`'s `GifWriter` (no in-memory
  frame buffer).

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
`evaluation_history`, plus `sample` for evenly-spaced animation frames).

### CLI (`cli/src/`)

- **`cli.rs`** — top-level `clap` command enum dispatching to subcommands.
- **`commands/`** — one module per subcommand (`synth`, `encode`, `predict`), plus the `train/`
  group (`train start` / `train resume`) and the `plot/` group (`plot dataset` / `plot run` /
  `plot activations`). Each group is a module directory: `mod.rs` holds the subcommand enum, dispatch
  and small shared bits (e.g. plot's `Format` / `render`), one file per leaf subcommand (`start.rs` /
  `resume.rs`, `dataset.rs` / `run.rs` / `activations.rs`), and larger shared concerns get their own
  file (`train/args.rs`, `train/callbacks.rs`). Scaling is a `train --scale` option (no separate
  command): the scaler is fitted during training and bundled with the model, so `predict` loads a
  composite `Predictor` (network + optional scaler) from a model directory. The CLI parses args into a
  core `HyperParameters` spec (via `TryFrom`, in `train/args.rs`) and runs it: it composes the
  callbacks, calls `build(..)`, optionally `restore(..)`s for resume, then `train()`. The
  console-facing callbacks (`ModelSaver`, `ConsoleMonitor` with its progress bar) are `TrainerCallback`
  impls under `train/`. `plot` renders a figure inline (`--format console`) or to a file
  (`--format image`: a PNG, or a streamed GIF when `plot run --animate`); `plot activations <model>
  --instance <file>` builds an `ActivationDiagram` for one instance (console nodes-only diagram, or
  `--format image` node-link PNG, with `--max-units` / `--min-edge`); `predict --activations` prints
  the same console diagram before the classification; `synth` previews a 2-feature dataset inline.
- **`display/`** — console rendering: the `Describe`/`Named` entity traits, status icons/verbs,
  `Artifacts`, and `terminal.rs` (figure `preview` and `play_frames` console animation, sized to the
  terminal). Loading/saving goes through core `.load()` / `.save()` methods on the types
  (`Dataset::load`, `Predictor::load`, `dataset.save`, …).
