# Contributing

Thanks for your interest! This is an educational project, so corrections, clearer explanations and
questions are as welcome as code. Open an issue to discuss anything non-trivial before sending a PR.

## Prerequisites

- Rust **edition 2024** (MSRV **1.88** — the code relies on let-chains). Install/update via
  [rustup.rs](https://rustup.rs/).
- [Task](https://taskfile.dev) for the project's command runner (`brew install go-task/tap/go-task`, or see
  [taskfile.dev/installation](https://taskfile.dev/installation/)). Optional but recommended.

All serialization is pure Rust — no system C library is needed to build or run.

## Building from source

```sh
git clone https://github.com/fmeriaux/nrn.git
cd nrn

task build        # release binary at target/release/nrn
task dev          # debug build
# or, without Task:
cargo build --release
```

## Project layout

A two-crate Cargo workspace:

- **`core/`** (crate `nrn`) — the neural-network library: model, training stack (optimizers, schedulers,
  losses), data handling, a backend-neutral plotting IR, and safetensors/JSON I/O. No binary. The library
  types are re-exported flat at the crate root (e.g. `nrn::model`, `nrn::training`).
- **`cli/`** (crate `nrn-cli`) — the `nrn` binary: `clap` argument parsing and one module per subcommand
  (`synth`, `encode`, `train`, `predict`, `plot`).

Arrays follow a `(features, samples)` layout throughout — columns are samples, rows are features.

## Common commands

Run `task` with no arguments to list everything. The most useful:

```sh
task checks            # lint + build + test — the pre-commit gate
task lint              # rustfmt --check + clippy (-D warnings), with & without features
task test              # workspace tests
task coverage          # coverage summary
task coverage-html     # HTML coverage report
task coverage-check    # fail under the line-coverage threshold (CI gate)
task audit             # cargo-audit advisory scan
```

`task test` runs the suite through [cargo-nextest](https://nexte.st) (a process per test, parallelized)
plus the doctests, which nextest doesn't cover. Install it with `cargo install cargo-nextest --locked`
or `brew install cargo-nextest`.

Single tests, as usual:

```sh
cargo nextest run <name>        # by name, across the workspace
cargo nextest run -p nrn <name> # core crate only
```

## Testing

New public behavior ships with a test; every bug fix ships with a regression test that fails before the fix.
Unit tests live in-module under `#[cfg(test)]`; CLI behavior is covered end-to-end through `assert_cmd` in
`cli/tests/`. CI gates merges on a line-coverage threshold (`task coverage-check`) — please keep coverage
from regressing.

## Commit conventions

This repository follows [Conventional Commits](https://conventionalcommits.org). Every message is prefixed
with a type, optionally scoped (`feat(training): …`):

| Type | Meaning |
| --- | --- |
| `feat:` | a new feature |
| `fix:` | a bug fix |
| `refactor:` | a change that is neither a feature nor a fix |
| `test:` | tests only |
| `docs:` | documentation only |
| `ci:` | CI/CD |
| `build:` | build system or dependencies |
| `chore:` | everything else |

Run `task checks` before pushing — it's the same gate CI enforces.
