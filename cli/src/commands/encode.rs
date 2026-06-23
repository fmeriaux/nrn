use crate::display::{Artifacts, bar, completed, saved, show};
use crate::path::PathExt;
use clap::{Args, Subcommand};
use console::style;
use nrn::data::vectorizers::{ImageEncoder, VectorEncoder};
use nrn::data::{Classes, Dataset, Instance};
use nrn::io::bytes::secure_read;
use std::error::Error;
use std::fs::read_dir;
use std::path::PathBuf;

#[derive(Args, Debug)]
pub struct EncodeArgs {
    #[command(subcommand)]
    subcommand: EncodeCommand,
}

#[derive(Subcommand, Debug)]
pub enum EncodeCommand {
    /// Encode a directory of per-class image folders into a dataset
    ImgDir(ImgDirArgs),
    /// Encode a single image into an instance
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

/// Image-to-vector encoding knobs shared by both subcommands.
#[derive(Args, Debug)]
pub struct EncoderArgs {
    /// Convert images to grayscale instead of keeping RGB
    #[arg(long, default_value_t = false)]
    grayscale: bool,

    /// Side length, in pixels, to resize each image to before flattening
    #[arg(short, long, default_value_t = 64, value_parser = clap::value_parser!(u32).range(1..=128))]
    shape: u32,
}

impl From<&EncoderArgs> for ImageEncoder {
    fn from(args: &EncoderArgs) -> Self {
        ImageEncoder {
            img_shape: (args.shape, args.shape),
            grayscale: args.grayscale,
        }
    }
}

#[derive(Args, Debug)]
pub struct ImgDirArgs {
    /// Directory of per-class subfolders of images to encode
    input: PathBuf,

    /// Name to save the dataset under (defaults to the dataset's identifier)
    #[arg(short, long)]
    output: Option<PathBuf>,

    #[command(flatten)]
    encoder: EncoderArgs,
}

impl ImgDirArgs {
    pub fn run(&self) -> Result<(), Box<dyn Error>> {
        let classes = Classes::scan(&self.input)?;
        show(&classes);

        let encoder = ImageEncoder::from(&self.encoder);

        let mut data = Vec::new();
        let mut labels = Vec::new();

        for (i, (category, label)) in classes.iter().enumerate() {
            let entries = read_dir(self.input.join(category))?
                .filter_map(Result::ok)
                .collect::<Vec<_>>();

            let progression = bar(
                entries.len(),
                format!(
                    "Encoding category [{}/{}]: {}",
                    i + 1,
                    classes.len(),
                    style(category).bright().blue()
                ),
            );

            for entry in entries {
                progression.inc(1);

                let img = secure_read(entry.path())?;

                if let Ok(img) = encoder.encode(&img) {
                    data.push(img);
                    labels.push(label.to_owned());
                }
            }

            progression.finish_and_clear();
        }

        let dataset = Dataset::from_encoded(self.input.file_stem_string(), data, labels)?;

        completed!("Encoding completed");

        let filename = self.output.clone().unwrap_or_else(|| dataset.id().into());
        saved(&Artifacts::single(
            "Image Dataset",
            dataset.save(&filename)?,
        ));

        Ok(())
    }
}

#[derive(Args, Debug)]
pub struct ImgArgs {
    /// Image file to encode
    input: PathBuf,

    /// Name to save the instance under (defaults to the image's file stem)
    #[arg(short, long)]
    output: Option<PathBuf>,

    #[command(flatten)]
    encoder: EncoderArgs,
}

impl ImgArgs {
    /// The path to save the instance under: the `--output` override, or the input
    /// image's file stem by default.
    fn output_path(&self) -> PathBuf {
        self.output
            .clone()
            .unwrap_or_else(|| self.input.file_stem_string().into())
    }

    pub fn run(&self) -> Result<(), Box<dyn Error>> {
        let encoder = ImageEncoder::from(&self.encoder);

        let image = encoder.encode(&secure_read(&self.input)?)?;
        let instance = Instance::from(image);

        completed!("Encoding completed");

        saved(&Artifacts::single(
            "Instance",
            instance.save(self.output_path())?,
        ));

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        encoder: EncoderArgs,
    }

    fn encoder(extra: &[&str]) -> ImageEncoder {
        let mut argv = vec!["encode"];
        argv.extend_from_slice(extra);
        ImageEncoder::from(&Cli::parse_from(argv).encoder)
    }

    #[test]
    fn encoder_defaults_to_rgb_64px() {
        let encoder = encoder(&[]);
        assert_eq!(encoder.img_shape, (64, 64));
        assert!(!encoder.grayscale);
    }

    #[test]
    fn shape_sets_both_dimensions() {
        assert_eq!(encoder(&["--shape", "32"]).img_shape, (32, 32));
    }

    #[test]
    fn grayscale_flag_selects_single_channel() {
        assert!(encoder(&["--grayscale"]).grayscale);
    }

    #[derive(Parser)]
    struct ImgCli {
        #[command(flatten)]
        args: ImgArgs,
    }

    fn img_args(extra: &[&str]) -> ImgArgs {
        let mut argv = vec!["img"];
        argv.extend_from_slice(extra);
        ImgCli::parse_from(argv).args
    }

    #[test]
    fn output_path_defaults_to_the_input_file_stem() {
        assert_eq!(
            img_args(&["pics/digit.png"]).output_path(),
            PathBuf::from("digit")
        );
    }

    #[test]
    fn output_path_honours_the_override() {
        assert_eq!(
            img_args(&["pics/digit.png", "--output", "out/encoded"]).output_path(),
            PathBuf::from("out/encoded")
        );
    }
}
