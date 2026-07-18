//! Shared fixtures for the CLI's integration tests, split by concern: running the `nrn` binary
//! ([`cli`]), a plain binary predictor ([`predictor`]), and the `train`/`plot` test family's
//! dataset and checkpoint helpers ([`training`]).
//!
//! Each `tests/*.rs` file compiles this module as its own copy, so a fixture (or a whole
//! submodule's glob re-export) unused by one of them still triggers `dead_code` /
//! `unused_imports` there.
#![allow(dead_code, unused_imports)]

mod cli;
mod predictor;
mod training;

pub use cli::*;
pub use predictor::*;
pub use training::*;
