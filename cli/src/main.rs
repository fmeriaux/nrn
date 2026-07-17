use crate::cli::CliArgs;
use crate::display::error;
use clap::Parser;

mod actions;
mod cli;
mod commands;
mod display;

fn main() {
    CliArgs::parse().command.run().unwrap_or_else(|e| {
        error!("{e}");
        std::process::exit(1);
    });
}
