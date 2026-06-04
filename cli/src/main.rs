use crate::cli::CliArgs;
use crate::display::error;
use clap::Parser;

mod actions;
mod cli;
mod commands;
pub mod display;
mod progression;

fn main() {
    if let Err(e) = CliArgs::parse().command.run() {
        error(&e.to_string());
        std::process::exit(1);
    }
}
