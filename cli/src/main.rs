use crate::cli::CliArgs;
use clap::Parser;
use std::error::Error;

mod actions;
mod cli;
mod commands;
mod console;
mod progression;

fn main() -> Result<(), Box<dyn Error>> {
    let args = CliArgs::parse();
    args.command.run()
}
