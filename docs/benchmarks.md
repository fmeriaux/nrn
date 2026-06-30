# Benchmarks

Performance benchmarks for the MLP hot paths, measured with
[criterion](https://github.com/bheisler/criterion.rs). Run them with:

```sh
task bench                                        # pure-Rust default backend
cargo bench -p nrn --bench training --features blas   # system BLAS (per-OS backend)
```

The harness lives in [`core/benches/training.rs`](../core/benches/training.rs) and
covers three operations on two workloads:

| Operation          | What it measures                                        |
| ------------------ | ------------------------------------------------------- |
| `inference`        | `NeuralNetwork::predict` on the whole dataset           |
| `epoch_full_batch` | one training epoch, full-batch gradient descent         |
| `epoch_mini_batch` | one training epoch, mini-batch SGD (batch size 64)      |

| Workload | Architecture                         | Samples |
| -------- | ------------------------------------ | ------- |
| `small`  | 2 → 16 (relu) → 1 (sigmoid)          | 1 000   |
| `mnist`  | 784 → 128 → 64 (relu) → 10 (softmax) | 2 000   |

Each epoch benchmark clones the model in criterion's `iter_batched` setup, so the
clone is excluded from the measured region and every iteration starts from the
same weights (training mutates them in place). The optimizer is Adam with default
moments; gradient clipping is off.

## Results

Times are the criterion median (point estimate). Lower is better. Re-measure on
your own machine before drawing conclusions — these are a relative baseline, not
an absolute spec.

### Baseline — 2026-06-30

- **Machine:** Apple M4 (10 cores), `rustc 1.95.0`, `--release`
- **Commit:** `perf/training-inference-buffers` @ baseline (criterion harness added,
  no optimization applied yet)

| Operation          | small (pure) | small (BLAS) | mnist (pure) | mnist (BLAS) | mnist speedup |
| ------------------ | ------------ | ------------ | ------------ | ------------ | ------------- |
| `inference`        | 11.6 µs      | 11.2 µs      | 4.54 ms      | 0.52 ms      | **8.8×**      |
| `epoch_full_batch` | 37.4 µs      | 26.1 µs      | 9.35 ms      | 1.17 ms      | **8.0×**      |
| `epoch_mini_batch` | 78.7 µs      | 76.9 µs      | 14.1 ms      | 6.03 ms      | **2.3×**      |

> Measured with a short run (`--warm-up-time 1 --measurement-time 3`) for a quick
> baseline. The default (longer) settings give tighter confidence intervals.

## The `blas` feature

By default `dot` runs on ndarray's pure-Rust
[`matrixmultiply`](https://crates.io/crates/matrixmultiply) backend. A `blas`
backend routes matmul through a system BLAS (`sgemm`):

- **~8×** on `inference` and `epoch_full_batch` for the MNIST-sized workload.
- **2.3×** on `epoch_mini_batch`: that path is bound by the per-batch index gather
  (`select`), not the matmul.
- **No change** on `small`: 2-feature matmuls are dominated by per-call overhead.

The backend is picked per OS: Accelerate on macOS (no extra install), OpenBLAS
elsewhere.

```sh
cargo build -p nrn-cli --release --features blas   # or: task install (BLAS on macOS)
```

BLAS reorders f32 summation, so results differ slightly from the pure-Rust
backend. Each backend is deterministic for a fixed seed (locked by
`training_is_deterministic_for_a_fixed_seed`).

The dataset stays C-major `(features, samples)`: a column-major (F-order) input
measured a wash on the pure-Rust backend and ~10% slower under BLAS (mnist
inference 524 µs vs 577 µs).
