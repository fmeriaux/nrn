# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Educational from-scratch neural network implementation in Rust (edition 2024, MSRV 1.88 — the
code relies on let-chains). Implements feedforward networks (SLP and MLP) with a CLI for the full
ML workflow: data generation/encoding → training (with optional scaling) → visualization → prediction.

## Commands

Build/test/lint commands, testing conventions, and commit conventions are documented in
[CONTRIBUTING.md](CONTRIBUTING.md) — that's the source of truth, followed here too (`task checks`
is the pre-commit gate; single test: `cargo nextest run <name>`).

## Code style

- **Doc comments (`///`)** — descriptive only, never justify a design choice ("so that…",
  "because…") or enumerate a field's consumers. Document all fields of a struct/variant, or none.
  Applies beyond `///` too: `Cargo.toml`/config comments, CI/Taskfile comments, Markdown docs.
- **Panic messages** — `expect()` (internal invariant, addressed to a maintainer) has no trailing
  period; `assert!`/`panic!` (caller precondition, documented under `# Panics`) ends with one.
  Casing follows natural language; code identifiers and acronyms stay verbatim.
- **API shape** — prefer an inherent method on the domain type (`predictor.decision_boundary(..)`)
  over a free function taking the type as its first argument.
- **Docs stay in sync** — when a CLI flag, command, or workflow step changes, update
  README.md/CONTRIBUTING.md in the same PR; verify examples by actually running the binary.

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

- **`model/`** — `NetworkConfig` (weight-free architecture, assembled via `NetworkConfigBuilder`)
  and the `NeuralNetwork` it builds. `forward()` returns all intermediate activations; `predict()`
  returns only the last. Task-folded output width (binary → 1 neuron, multi-class → `n_classes`)
  is sourced by the CLI from `Task::output_size()`, not baked into the builder. `ModelConfig` pairs
  a `Task` with the `Labels` naming its classes, when known; `Predictor` pairs a `NeuralNetwork`
  with its `ModelConfig` and an optional fitted scaler. `Predictor::infer`/`infer_instance` return
  an `Inference`, split by task into a ranked `Classification` (`Binary`/`MultiClass`) or bare
  `Values` (`MultiLabel`/`Regression`) — resolving class ids to names stays with the CLI.
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
- **`loss_functions/`** — `LossFunction` trait; cross-entropy (binary and categorical) and
  mean squared error (regression).
- **`accuracies/`** — `Accuracy` trait; `accuracy_for(n_classes)` picks binary vs argmax-match.
- **`evaluation.rs`** — `Evaluation` (loss + accuracy for one split) and `EvaluationSet`
  (train / optional validation / test).
- **`evaluation_history.rs`** — `EvaluationHistory(Vec<EpochEvaluation { epoch, evaluation }>)`: pure
  value object, ordered by epoch; exposes per-split loss/accuracy series, ranges, and `epochs()`
  (chart X-axis).
- **`initializations/`** — `he` / `xavier` weight initializers, selected per activation.

### Data (`core/src/data/`)

- **`dataset.rs`** — `Dataset` pairs samples-major `inputs` with typed `targets` (`targets.rs`) and
  an optional `DatasetInfo`; converts to `ModelDataset` (column-major) via `to_model_dataset()`.
  `batches()` shuffles and chunks for mini-batch SGD; `split()` shuffles (seeded) then partitions
  into `ModelSplit` (train/val/test), so producers store data in natural order and the run seed
  governs the partition.
- **`targets.rs`** — `Targets::ClassLabel(ClassLabel)` / `Targets::Value(Values)` give a dataset's
  targets a dtype instead of a bare array; `ClassLabel` pairs ids with optional `Classes` names.
- **`classes.rs`** — `Classes`, a name → 0-indexed label mapping; `Classes::scan` (behind `io`)
  builds one from a directory's subfolders.
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
  and weighted `Edge`s). `NeuralNetwork::activation_diagram` /
  `Predictor::activation_diagram` (the latter scales the input first) build it, applying
  `DiagramOptions` to cap the units shown per layer (`max_units`) and prune weak edges by contribution
  (`min_edge_magnitude`). Output `Unit`s carry the class they represent, so both renderers read them as
  class probabilities. The diagram deliberately does *not* render the ranked decision — that stays with
  the CLI's `evaluated` presenter (via `Describe for Inference`), so the ranking has one home.
- **`image/`** (`raster`, `plotters`) and **`console/`** (`console`, `textplots`) — the two renderers,
  each a folder split by IR: `mod.rs` owns the shared config (`ImageConfig` / `ConsoleConfig`) and
  color helpers, `figure.rs` renders `Figure`, `diagram.rs` renders `ActivationDiagram`. `Figure` becomes
  a `RasterImage` or a `String`; `ActivationDiagram::to_image` draws a horizontal node-link graph and
  `ActivationDiagram::to_console` lists the layers vertically (nodes only, no edges). Pure rendering — the
  caller persists the bytes (`io`) or prints the text. Animations are streamed frame-by-frame to a GIF by
  `io`'s `GifWriter` (no in-memory frame buffer).

### I/O (`core/src/io/`, behind `io` feature)

Persistence is split by artifact: models, checkpoints, and optimizer/scheduler state use
[safetensors](https://github.com/huggingface/safetensors) (`f32` tensors plus a `__metadata__`
string map); datasets use Parquet, storing samples-major arrays as the canonical
`arrow.fixed_shape_tensor` extension and rank-1 targets as a scalar `label` column (`Int64` for
`ClassLabel`, `Float32` for `Value`); instances, scalers, and run hyperparameters are JSON. The
guiding rule: **core runtime types stay serde-free**, so `io` owns their serializable mirrors —
`HyperParametersRecord` for `HyperParameters`, plus the optimizer/scheduler `*State` round-trips.
`io/model/tensors.rs` holds the shared `View` adapter; the activation-name → `Arc<dyn Activation>`
round-trip goes through `ActivationProvider::get_by_name`.

A run lives in a directory managed by `TrainingRun`: `create`/`open` persist run-level `TrainingMeta`
(`meta.json` — dataset, final-model name, hyperparameters) alongside the model blueprint
(`config.json`, a `ModelConfigRecord` of architecture + task) and, when the run scales its inputs,
a `preprocessor.json` sidecar — the same `config.json`/`preprocessor.json` pair a `Predictor`
directory carries. `trim_after` rewinds the trajectory (removing later checkpoints), `archive()`
reads it back. Each `checkpoint-{epoch:06}/` is written by a `CheckpointRecorder` (the
`TrainerCallback` from `TrainingRun::recorder`) and read by a `CheckpointArchive` (`model_at` /
`optimizer_at` / `epoch_at` / `evaluation_history`, plus `sample` for evenly-spaced animation frames).

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
  `--format image` node-link PNG, with `--max-units` / `--min-edge`) — pure visualization, it does not
  print the ranked decision; `predict --activations` prints that same console diagram above the
  classification, which `evaluated` always renders (with the winning class arrow-marked); `synth`
  previews a 2-feature dataset inline.
- **`display/`** — console rendering: the `Describe`/`Named` entity traits, status icons/verbs,
  `Artifacts`, and `terminal.rs` (figure `preview` and `play_frames` console animation, sized to the
  terminal). Loading/saving goes through core `.load()` / `.save()` methods on the types
  (`Dataset::load`, `Predictor::load`, `dataset.save`, …).

<!-- code-review-graph MCP tools -->
## MCP Tools: code-review-graph

**IMPORTANT: This project has a knowledge graph. ALWAYS use the
code-review-graph MCP tools BEFORE using Grep/Glob/Read to explore
the codebase.** The graph is faster, cheaper (fewer tokens), and gives
you structural context (callers, dependents, test coverage) that file
scanning cannot.

### When to use graph tools FIRST

- **Exploring code**: `semantic_search_nodes_tool` or `query_graph_tool` instead of Grep
- **Understanding impact**: `get_impact_radius_tool` instead of manually tracing imports
- **Code review**: `detect_changes_tool` + `get_review_context_tool` instead of reading entire files
- **Finding relationships**: `query_graph_tool` with callers_of/callees_of/imports_of/tests_for
- **Architecture questions**: `get_architecture_overview_tool` + `list_communities_tool`

Fall back to Grep/Glob/Read **only** when the graph doesn't cover what you need.

### Key Tools

| Tool | Use when |
| ------ | ---------- |
| `detect_changes_tool` | Reviewing code changes — gives risk-scored analysis |
| `get_review_context_tool` | Need source snippets for review — token-efficient |
| `get_impact_radius_tool` | Understanding blast radius of a change |
| `get_affected_flows_tool` | Finding which execution paths are impacted |
| `query_graph_tool` | Tracing callers, callees, imports, tests, dependencies |
| `semantic_search_nodes_tool` | Finding functions/classes by name or keyword |
| `get_architecture_overview_tool` | Understanding high-level codebase structure |
| `refactor_tool` | Planning renames, finding dead code |

### Workflow

1. The graph auto-updates on file changes (via hooks).
2. Use `detect_changes_tool` for code review.
3. Use `get_affected_flows_tool` to understand impact.
4. Use `query_graph_tool` pattern="tests_for" to check coverage.
