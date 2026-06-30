# Benchmarks

Performance benchmarks for the MLP hot paths, measured with
[criterion](https://github.com/bheisler/criterion.rs). Run them with:

```sh
task bench           # or: cargo bench -p nrn --bench training
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

| Operation          | small     | mnist     |
| ------------------ | --------- | --------- |
| `inference`        | 11.6 µs   | 4.54 ms   |
| `epoch_full_batch` | 37.4 µs   | 9.35 ms   |
| `epoch_mini_batch` | 78.7 µs   | 14.1 ms   |

> Measured with a short run (`--warm-up-time 1 --measurement-time 3`) for a quick
> baseline. The default (longer) settings give tighter confidence intervals.
