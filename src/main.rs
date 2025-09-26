use clap::Parser;

mod cli;
mod commands;
mod progression;
mod plot;
mod log;
mod encoder;

use crate::commands::Command;

/// Command line interface for a neural network training and prediction tool.
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    cli::handle(args.command)?;

    Ok(())
}
