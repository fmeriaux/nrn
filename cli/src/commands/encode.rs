use crate::display::{Artifacts, Encoding, completed, saved, show};
use crate::path::PathExt;
use clap::{Args, Subcommand};
use nrn::data::vectorizers::{ImageEncoder, VectorEncoder};
use nrn::data::{Classes, Dataset, Instance};
use nrn::io::bytes::secure_read;
use std::error::Error;
use std::fs::read_dir;
use std::io;
use std::path::PathBuf;

#[derive(Args, Debug)]
pub struct EncodeArgs {
    #[command(subcommand)]
    subcommand: EncodeCommand,
}

#[derive(Subcommand, Debug)]
pub enum EncodeCommand {
    /// Encode a directory of per-class image folders into a dataset
    #[command(visible_alias = "ds")]
    Dataset(DatasetArgs),
    /// Encode a single image into an instance
    #[command(visible_alias = "inst")]
    Instance(InstanceArgs),
}

impl EncodeArgs {
    pub fn run(self) -> Result<(), Box<dyn Error>> {
        match self.subcommand {
            EncodeCommand::Dataset(args) => args.run(),
            EncodeCommand::Instance(args) => args.run(),
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
pub struct DatasetArgs {
    /// Directory of per-class subfolders of images to encode
    input: PathBuf,

    /// Name to save the dataset under (defaults to the dataset's identifier)
    #[arg(short, long)]
    output: Option<PathBuf>,

    #[command(flatten)]
    encoder: EncoderArgs,
}

impl DatasetArgs {
    pub fn run(&self) -> Result<(), Box<dyn Error>> {
        let classes = Classes::scan(&self.input)?;
        show(&classes);

        let encoder = ImageEncoder::from(&self.encoder);

        // Read every category's directory up front so a single bar can span the
        // whole run's images with an honest end-to-end ETA.
        let categories = classes
            .iter()
            .map(|(category, label)| {
                let entries = read_dir(self.input.join(category))?
                    .filter_map(Result::ok)
                    .collect::<Vec<_>>();
                Ok((category, *label, entries))
            })
            .collect::<Result<Vec<_>, io::Error>>()?;

        let total = categories.iter().map(|(_, _, entries)| entries.len()).sum();

        let mut data = Vec::new();
        let mut labels = Vec::new();

        let progress = Encoding::new(total);
        for (index, (category, label, entries)) in categories.iter().enumerate() {
            progress.category(index, classes.len(), category);

            for entry in entries {
                progress.advance();

                let img = secure_read(entry.path())?;

                if let Ok(img) = encoder.encode(&img) {
                    data.push(img);
                    labels.push(*label as u32);
                }
            }
        }
        progress.finish();

        let source = self.input.file_stem_string();
        let n_classes = classes.len();
        let dataset = Dataset::from_encoded(&source, data, labels, Some(classes))?;

        completed!("Encoding completed");

        let filename = self.output.clone().unwrap_or_else(|| {
            format!(
                "{source}-c{n_classes}-f{}-n{}",
                dataset.n_features(),
                dataset.n_samples()
            )
            .into()
        });
        saved(&Artifacts::single(
            "Image Dataset",
            dataset.save(&filename)?,
        ));

        Ok(())
    }
}

#[derive(Args, Debug)]
pub struct InstanceArgs {
    /// Image file to encode
    input: PathBuf,

    /// Name to save the instance under (defaults to the image's file stem)
    #[arg(short, long)]
    output: Option<PathBuf>,

    #[command(flatten)]
    encoder: EncoderArgs,
}

impl InstanceArgs {
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
        args: InstanceArgs,
    }

    fn img_args(extra: &[&str]) -> InstanceArgs {
        let mut argv = vec!["instance"];
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

    #[derive(Parser)]
    struct EncodeCli {
        #[command(subcommand)]
        command: EncodeCommand,
    }

    #[test]
    fn ds_is_an_alias_for_dataset() {
        let cli = EncodeCli::try_parse_from(["encode", "ds", "images/"]).unwrap();
        assert!(matches!(cli.command, EncodeCommand::Dataset(_)));
    }

    #[test]
    fn inst_is_an_alias_for_instance() {
        let cli = EncodeCli::try_parse_from(["encode", "inst", "digit.png"]).unwrap();
        assert!(matches!(cli.command, EncodeCommand::Instance(_)));
    }
}
