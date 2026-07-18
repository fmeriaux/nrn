//! A sandboxed workspace and running the `nrn` binary inside it.

use assert_cmd::Command;
use assert_cmd::assert::Assert;
use std::path::Path;
use tempfile::TempDir;

/// A temp dir under the crate directory (within the path-safety boundary).
pub fn workspace() -> TempDir {
    TempDir::new_in(".").unwrap()
}

/// A fresh `nrn` invocation rooted at `dir`.
pub fn nrn(dir: &Path) -> Command {
    let mut cmd = Command::cargo_bin("nrn").unwrap();
    cmd.current_dir(dir);
    cmd
}

/// Runs `nrn` with `args` from `dir`.
pub fn run(dir: &Path, args: &[&str]) -> Assert {
    nrn(dir).args(args).assert()
}
