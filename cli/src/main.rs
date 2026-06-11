use crate::cli::CliArgs;
use crate::console::error;
use clap::Parser;

mod actions;
mod cli;
mod commands;
pub mod console;

fn main() {
    CliArgs::parse().command.run().unwrap_or_else(|e| {
        error(&e.to_string());
        std::process::exit(1);
    });
}
