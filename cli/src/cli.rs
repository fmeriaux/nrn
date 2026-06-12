use crate::cli::Command::*;
use crate::commands::*;
use clap::{Parser, Subcommand};
use std::error::Error;

/// Command line interface for a neural network training and prediction tool.
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct CliArgs {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Generate synthetic data
    Synth(SynthArgs),
    /// Encode data to a representative format
    Encode(EncodeArgs),
    /// Scale the dataset features
    Scale(ScaleArgs),
    /// Train a model on the dataset
    #[command(subcommand)]
    Train(TrainCommand),
    /// Predict using a trained model
    Predict(PredictArgs),
    /// Plot training curves and decision boundaries
    Plot(PlotArgs),
}

impl Command {
    pub(crate) fn run(self) -> Result<(), Box<dyn Error>> {
        match self {
            // 🧪 SYNTHETIC DATA GENERATION
            Synth(args) => args.run(),
            // 🖼️ / 📄 DATASET ENCODING
            Encode(args) => args.run(),
            // 📊 DATASET SCALING
            Scale(args) => args.run(),
            // 🧠 NEURAL NETWORK TRAINING
            Train(args) => args.run(),
            // 🔮 PREDICTION
            Predict(args) => args.run(),
            // 📊 PLOT TRAINING CURVES AND DECISION BOUNDARIES
            Plot(args) => args.run(),
        }
    }
}
