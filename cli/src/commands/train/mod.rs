mod args;
mod callbacks;
mod resume;
mod start;

use clap::Subcommand;
use nrn::training::FatalDivergence;
use resume::ResumeArgs;
use start::StartArgs;
use std::error::Error;
use std::fmt;

// ─── TrainCommand ─────────────────────────────────────────────────────────────

#[derive(Subcommand, Debug)]
pub enum TrainCommand {
    /// Start a new training run from scratch
    Start(StartArgs),
    /// Resume training from an existing run directory
    Resume(ResumeArgs),
}

impl TrainCommand {
    pub fn run(&self) -> Result<(), Box<dyn Error>> {
        match self {
            TrainCommand::Start(args) => args.run(),
            TrainCommand::Resume(args) => args.run(),
        }
    }
}

// ─── DivergedRun ────────────────────────────────────────────────────────────────

/// Reported when training ends in an unrecovered divergence, with actionable hints.
#[derive(Debug)]
struct DivergedRun {
    final_epoch: usize,
}

impl From<FatalDivergence> for DivergedRun {
    fn from(divergence: FatalDivergence) -> Self {
        Self {
            final_epoch: divergence.final_epoch,
        }
    }
}

impl fmt::Display for DivergedRun {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "model diverged at epoch {} (NaN/Inf in weights). \
             Try: --early-stopping with --restore-best-model, --scheduler cosine, \
             a lower --lr, or stronger gradient clipping.",
            self.final_epoch
        )
    }
}

impl Error for DivergedRun {}
