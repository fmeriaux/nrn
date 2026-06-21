use crate::display::{error, loaded};
use clap::Args;
use console::style;
use ndarray::Array1;
use nrn::io::data::load_inputs;
use nrn::model::Predictor;
use std::cmp::Ordering::Equal;
use std::error::Error;
use std::io::stdin;

#[derive(Args, Debug)]
pub struct PredictArgs {
    /// Model directory to predict with (network plus its optional scaler)
    model: String,

    /// Specify the input data for prediction, if not provided, it will read from stdin
    #[arg(short, long)]
    input: Option<String>,
}

impl PredictArgs {
    pub fn run(self) -> Result<(), Box<dyn Error>> {
        let predictor = Predictor::load(&self.model)?;
        loaded(&predictor.network);
        if let Some(scaler) = &predictor.scaler {
            loaded(scaler);
        }

        let input = if let Some(input_file) = self.input {
            let input = load_inputs(&input_file)?;

            println!(
                "Input data loaded from {}",
                style(input_file).bright().blue().italic()
            );

            input
        } else {
            let mut inputs = Vec::with_capacity(predictor.network.input_size());

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
                        error!("{err}");
                    }
                }

                if inputs.len() >= inputs.capacity() {
                    break;
                }
            }

            Array1::from_vec(inputs)
        };

        let predictions = predictor.predict_single(input.view());

        let mut result: Vec<(usize, f32)> = if predictions.len() == 1 {
            vec![(0, 1.0 - predictions[0]), (1, predictions[0])]
        } else {
            predictions
                .iter()
                .enumerate()
                .map(|(index, &value)| (index, value))
                .collect::<Vec<_>>()
        };

        result.sort_by(|(_, p1), (_, p2)| p2.partial_cmp(p1).unwrap_or(Equal));

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
