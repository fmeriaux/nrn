use crate::actions;
use crate::progression::Progression;
use clap::{Args, Subcommand};
use console::style;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand::prelude::StdRng;
use nrn::data::Dataset;
use nrn::data::vectorizers::{ImageEncoder, VectorEncoder};
use nrn::io::bytes::secure_read;
use nrn::io::classes::extract_classes;
use nrn::io::data::save_inputs;
use std::error::Error;
use std::fs::read_dir;
use std::path::Path;

#[derive(Args, Debug)]
pub struct EncodeArgs {
    #[command(subcommand)]
    subcommand: EncodeCommand,
}

#[derive(Subcommand, Debug)]
pub enum EncodeCommand {
    /// Encode images from a directory into a dataset
    ImgDir(ImgDirArgs),
    /// Encode a single image
    Img(ImgArgs),
}

impl EncodeArgs {
    pub fn run(self) -> Result<(), Box<dyn Error>> {
        match self.subcommand {
            EncodeCommand::ImgDir(args) => args.run(),
            EncodeCommand::Img(args) => args.run(),
        }
    }
}

#[derive(Args, Debug)]
pub struct ImgDirArgs {
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
}

impl ImgDirArgs {
    pub fn run(&self) -> Result<(), Box<dyn Error>> {
        // Define categories by iterating over directories
        let root = Path::new(&self.input);

        let classes = extract_classes(&root)?;

        println!(
            "{}: {}",
            style("Classes found").bright().cyan(),
            classes
                .iter()
                .map(|(name, label)| format!(
                    "{} (as {})",
                    style(name).bright().blue(),
                    style(label).yellow()
                ))
                .collect::<Vec<_>>()
                .join(", ")
        );

        let mut data = Vec::new();
        let mut labels = Vec::new();

        for (i, (category, label)) in classes.iter().enumerate() {
            let total_img = read_dir(&root.join(category))?
                .filter_map(Result::ok)
                .count();

            let progression = Progression::new(
                total_img,
                format!(
                    "Encoding category [{}/{}]: {}",
                    i + 1,
                    classes.len(),
                    style(category).bright().blue()
                ),
            );

            let encoder = ImageEncoder {
                img_shape: (self.shape, self.shape),
                grayscale: self.grayscale,
            };

            for entry in read_dir(&root.join(category))?.filter_map(Result::ok) {
                progression.inc();

                let img = secure_read(entry.path())?;

                if let Ok(img) = encoder.encode(&img) {
                    data.push(img);
                    labels.push(label.to_owned());
                }
            }

            progression.done();
        }

        if data.is_empty() {
            return Err("No images found in the specified directory".into());
        }

        let mut rng = StdRng::seed_from_u64(self.seed);
        let dataset = Dataset::from_vec(&mut rng, data, labels)?;

        let split_dataset = dataset.split(self.train_ratio);

        println!("{}", style("Image dataset encoded").bright().green());

        actions::save_dataset(split_dataset, "IMAGE DATASET", false, &self.output)?;

        Ok(())
    }
}

#[derive(Args, Debug)]
pub struct ImgArgs {
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
}

impl ImgArgs {
    pub fn run(&self) -> Result<(), Box<dyn Error>> {
        let encoder = ImageEncoder {
            img_shape: (self.shape, self.shape),
            grayscale: self.grayscale,
        };

        let image = encoder.encode(&secure_read(Path::new(&self.input))?)?;

        save_inputs(&self.output, &image)?;

        println!("{}", style("Image encoded").bright().green());

        Ok(())
    }
}
