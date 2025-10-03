use crate::actions;
use crate::display;
use clap::Args;
use console::style;
use ndarray::Array1;
use nrn::data::scalers::{Scaler, ScalerMethod};
use nrn::io::data::load_inputs;
use std::cmp::Ordering::Equal;
use std::error::Error;
use std::io::stdin;

#[derive(Args, Debug)]
pub struct PredictArgs {
    // Name of the dataset to predict on
    model: String,

    /// Specify the input data for prediction, if not provided, it will read from stdin
    #[arg(short, long)]
    input: Option<String>,

    /// Specify the scaler used for the dataset features
    #[arg(short, long, value_enum)]
    scaler: Option<String>,
}

impl PredictArgs {
    pub fn run(self) -> Result<(), Box<dyn Error>> {
        let model = actions::load_model(&self.model)?;

        let scaler: Option<ScalerMethod> = self
            .scaler
            .iter()
            .find_map(|s| actions::load_scaler(s).ok());

        let mut input = if let Some(input_file) = self.input {
            let input = load_inputs(&input_file)?;

            println!(
                "Input data loaded from {}",
                style(input_file).bright().blue().italic()
            );

            input
        } else {
            let mut inputs = Vec::with_capacity(model.input_size());

            loop {
                println!(
                    "{}[{}]:",
                    style("Input").bold().bright().blue(),
                    style(inputs.len()).yellow(),
                );

                let mut raw = String::new();
                stdin().read_line(&mut raw)?;

                match raw.trim().parse::<f32>() {
                    Ok(val) => inputs.push(val),
                    Err(err) => {
                        display::error(err.to_string().as_str());
                    }
                }

                if inputs.len() >= inputs.capacity() {
                    break;
                }
            }

            Array1::from_vec(inputs)
        };

        if let Some(ref scaler) = scaler {
            scaler.apply_single_inplace(input.view_mut());
        }

        let predictions = model.predict_single(input.view());

        let mut result: Vec<(usize, f32)> = if predictions.len() == 1 {
            vec![(0, 1.0 - predictions[0]), (1, predictions[0])]
        } else {
            predictions
                .iter()
                .enumerate()
                .map(|(index, &value)| (index, value))
                .collect::<Vec<_>>()
        };

        result.sort_by(|(_, p1), (_, p2)| p2.partial_cmp(&p1).unwrap_or(Equal));

        println!(
            "{} for {}\n|> {}",
            style("Predictions").bold().bright().green(),
            style(input).yellow(),
            result
                .iter()
                .map(|(index, value)| format!(
                    "{}: {:.2}%",
                    style(index).bright().blue(),
                    value * 100.0
                ))
                .collect::<Vec<_>>()
                .join("\n|> ")
        );

        Ok(())
    }
}
