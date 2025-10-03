use crate::cli::CliArgs;
use crate::display::error;
use clap::Parser;
use std::error::Error;

mod actions;
mod cli;
mod commands;
pub mod display;
mod progression;

fn main() -> Result<(), Box<dyn Error>> {
    let args = CliArgs::parse();
    args.command.run().or_else(|e| {
        error(&e.to_string());
        Ok(())
    })
}
