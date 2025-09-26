use clap::{Subcommand, ValueEnum};
use ndarray::ArrayView2;
use nrn::scalers::{MinMaxScaler, ScalerMethod, ZScoreScaler};
use std::fmt;

#[derive(ValueEnum, Clone, Debug)]
pub enum ScalingOption {
    MinMax,
    ZScore,
}

impl ScalingOption {
    pub fn fit(&self, data: ArrayView2<f32>) -> ScalerMethod {
        match self {
            ScalingOption::MinMax => ScalerMethod::MinMax(MinMaxScaler::default().fit(data)),
            ScalingOption::ZScore => ScalerMethod::ZScore(ZScoreScaler::default().fit(data)),
        }
    }
}

#[derive(ValueEnum, Clone, Debug)]
pub enum DistributionOption {
    Uniform,
    Ring,
}

impl fmt::Display for DistributionOption {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DistributionOption::Uniform => write!(f, "uniform"),
            DistributionOption::Ring => write!(f, "ring"),
        }
    }
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Generate synthetic data
    Synth {
        /// Seed for random number generation reproducibility
        #[arg(short, long)]
        seed: u64,

        /// Type of distribution to use for generating data
        #[arg(short, long)]
        distribution: DistributionOption,

        /// Number of samples to generate
        #[arg(short = 'n', long, default_value_t = 100)]
        samples: usize,

        /// Number of features in the generated data
        #[arg(short, long, default_value_t = 2)]
        features: usize,

        /// Number of clusters to generate in the dataset
        #[arg(short, long, default_value_t = 2)]
        clusters: usize,

        /// Specify the training ratio for the dataset split
        #[arg(long, default_value_t = 0.8)]
        train_ratio: f32,

        /// Minimum value for each feature in the dataset
        #[arg(long, default_value_t = 0.0)]
        min: f32,

        /// Maximum value for each feature in the dataset
        #[arg(long, default_value_t = 10.0)]
        max: f32,

        /// Indicates whether to visualize the generated dataset (requires exactly two features)
        #[arg(long, default_value_t = false)]
        plot: bool,
    },
    /// Encode data to a representative format
    Encode {
        #[command(subcommand)]
        subcommand: EncodeCommand,
    },
    /// Scale the dataset features
    Scale {
        /// Name of the dataset to scale
        dataset: String,

        /// Specify the scaling method to apply to the dataset
        scaling: ScalingOption,

        /// Indicates whether to visualize the scaled dataset
        #[arg(long, default_value_t = false)]
        plot: bool,
    },
    /// Train a model on the dataset
    Train {
        /// Name of the dataset to train on
        dataset: String,

        /// Provide a pre-trained model to continue training, if not provided, a new model will be initialized
        #[arg(short, long)]
        model: Option<String>,

        /// The number of epochs to train the model
        #[arg(short, long)]
        epochs: usize,

        #[arg(short = 'k', long, default_value_t = 10)]
        /// Specify the checkpoint interval for saving the model state,
        /// if set to 0, no checkpoints will be saved
        checkpoint_interval: usize,

        /// Specify the hidden layers of the model when a new model is initialized
        #[arg(long, value_delimiter = ',', conflicts_with = "model")]
        layers: Option<Vec<usize>>,

        /// Specify the learning rate for the training process
        #[arg(long, default_value_t = 0.001)]
        learning_rate: f32,

        /// Specify the maximum norm for gradient clipping
        #[arg(long, default_value_t = 1.0)]
        max_norm: f32,
    },
    /// Predict using a trained model
    Predict {
        // Name of the dataset to predict on
        model: String,

        /// Specify the input data for prediction, if not provided, it will read from stdin
        #[arg(short, long)]
        input: Option<String>,

        /// Specify the scaler used for the dataset features
        #[arg(short, long, value_enum)]
        scaler: Option<String>,
    },
    /// Plot the training history
    Plot {
        /// Name of the training history file to plot
        history: String,

        /// Name of the dataset used for training for decision boundary visualization (only for 2D datasets)
        #[arg(short, long)]
        dataset: Option<String>,

        /// Specify the number of frames for the decision boundary animation
        #[arg(short, long, default_value_t = 20, requires = "dataset", value_parser = clap::value_parser!(u8).range(2..201))]
        frames: u8,

        /// Specify the width of the plot in pixels
        #[arg(long, default_value_t = 800, value_parser = clap::value_parser!(u32).range(100..=4096))]
        width: u32,

        /// Specify the height of the plot in pixels
        #[arg(long, default_value_t = 600, value_parser = clap::value_parser!(u32).range(100..=4096))]
        height: u32,
    },
}

#[derive(Subcommand, Debug)]
pub enum EncodeCommand {
    /// Encode images from a directory into a dataset
    ImgDir {
        /// Seed for shuffling the dataset
        #[arg(long)]
        seed: u64,

        /// Path to the directory containing images
        #[arg(short, long)]
        input: String,

        /// Path to save the encoded dataset
        #[arg(short, long)]
        output: String,

        /// Indicates whether to convert images to grayscale
        #[arg(long, default_value_t = false)]
        grayscale: bool,

        /// Specify the image shape for encoding
        #[arg(short, long, default_value_t = 64, value_parser = clap::value_parser!(u32).range(1..=128))]
        shape: u32,

        /// Specify the training ratio for the dataset split
        #[arg(long, default_value_t = 0.8)]
        train_ratio: f32,
    },
    /// Encode a single image
    Img {
        /// Path to the image file to encode
        #[arg(short, long)]
        input: String,

        /// Path to save the encoded image
        #[arg(short, long)]
        output: String,

        /// Indicates whether to convert the image to grayscale
        #[arg(long, default_value_t = false)]
        grayscale: bool,

        /// Specify the image shape for encoding
        #[arg(short, long, default_value_t = 64, value_parser = clap::value_parser!(u32).range(1..=128))]
        shape: u32,
    },
}
