# Benchmarks

Performance benchmarks for the MLP hot paths, measured with
[criterion](https://github.com/bheisler/criterion.rs). Run them with:

```sh
task bench                                        # pure-Rust default backend
cargo bench -p nrn --bench training --features blas   # system BLAS (per-OS backend)
```

The harness lives in [`core/benches/training.rs`](../core/benches/training.rs) and
covers three operations on three workloads:

| Operation          | What it measures                                        |
| ------------------ | ------------------------------------------------------- |
| `inference`        | `NeuralNetwork::predict` on the whole dataset           |
| `epoch_full_batch` | one training epoch, full-batch gradient descent         |
| `epoch_mini_batch` | one training epoch, mini-batch SGD (batch size 64)      |

| Workload | Architecture                                         | Samples |
| -------- | ---------------------------------------------------- | ------- |
| `small`  | 2 → 16 (relu) → 1 (sigmoid)                          | 1 000   |
| `mnist`  | 784 → 128 → 64 (relu) → 10 (softmax)                 | 2 000   |
| `cnn`    | 1×28×28 → conv2d(16, 3×3, relu) → flatten → 10 (softmax) | 1 000   |

The `cnn` workload takes a rank-4 `(channels, height, width, samples)` batch, so it
exercises the spatial `im2col`/`col2im` paths the MLP workloads never touch.

Each epoch benchmark clones the model in criterion's `iter_batched` setup, so the
clone is excluded from the measured region and every iteration starts from the
same weights (training mutates them in place). The optimizer is Adam with default
moments; gradient clipping is off.

## Results

Times are the criterion median (point estimate). Lower is better. Re-measure on
your own machine before drawing conclusions — these are a relative baseline, not
an absolute spec.

### Baseline — 2026-07-05

- **Machine:** Apple M4 (10 cores), `rustc 1.95.0`, `--release`
- **Commit:** `test(bench): add a CNN workload to the training benches` (`0371846`)

| Operation          | small (pure) | small (BLAS) | mnist (pure) | mnist (BLAS) | mnist × | cnn (pure) | cnn (BLAS) | cnn × |
| ------------------ | ------------ | ------------ | ------------ | ------------ | ------- | ---------- | ---------- | ----- |
| `inference`        | 12.5 µs      | 11.1 µs      | 4.22 ms      | 0.53 ms      | **8.0×** | 13.5 ms    | 10.5 ms    | **1.3×** |
| `epoch_full_batch` | 29.8 µs      | 26.0 µs      | 8.85 ms      | 1.09 ms      | **8.1×** | 33.4 ms    | 21.4 ms    | **1.6×** |
| `epoch_mini_batch` | 79.9 µs      | 80.6 µs      | 11.9 ms      | 4.15 ms      | **2.9×** | 35.0 ms    | 20.8 ms    | **1.7×** |

## The `blas` feature

By default `dot` runs on ndarray's pure-Rust
[`matrixmultiply`](https://crates.io/crates/matrixmultiply) backend. A `blas`
backend routes matmul through a system BLAS (`sgemm`):

- **~8×** on `inference` and `epoch_full_batch` for the MNIST-sized workload.
- **~3×** on `epoch_mini_batch`: that path is bound by the per-batch index gather
  (`select`), not the matmul.
- **No change** on `small`: 2-feature matmuls are dominated by per-call overhead.
- **~1.3–1.7×** on `cnn`: each sample's `im2col` matmul is small `(16, 9)·(9, 676)`,
  and the `im2col`/`col2im` reshaping around it is pure-Rust either way.

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
