//! The `train`/`plot` test family's dataset generation and checkpoint helpers.

use super::cli::run;
use std::fs;
use std::path::Path;

/// The number of `checkpoint-*` directories under `run_dir`.
pub fn checkpoint_count(run_dir: &Path) -> usize {
    fs::read_dir(run_dir)
        .map(|rd| {
            rd.filter_map(Result::ok)
                .filter(|e| {
                    e.file_name()
                        .to_str()
                        .map(|n| n.starts_with("checkpoint-"))
                        .unwrap_or(false)
                        && e.path().is_dir()
                })
                .count()
        })
        .unwrap_or(0)
}

/// Generates a small 2-class ring dataset in `dir` over the `[0, 10]` feature range, returning
/// its filename.
pub fn synth_ring(dir: &Path, seed: &str, samples: &str) -> String {
    run(
        dir,
        &[
            "synth",
            "--min",
            "0",
            "--max",
            "10",
            "--seed",
            seed,
            "--distribution",
            "ring",
            "--clusters",
            "2",
            "--samples",
            samples,
        ],
    )
    .success();
    format!("ring-seed{seed}-c2-f2-n{samples}")
}
