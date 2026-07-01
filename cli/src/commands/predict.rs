use crate::display::{error, evaluated, loaded, prompt};
use clap::Args;
use ndarray::Array1;
use nrn::data::Instance;
use nrn::model::Predictor;
use nrn::plot::DiagramOptions;
use std::error::Error;
use std::io::stdin;

#[derive(Args, Debug)]
pub struct PredictArgs {
    /// Model directory to predict with (network plus its optional scaler)
    model: String,

    /// Instance file to predict on; when omitted, the features are read from stdin
    #[arg(short, long)]
    instance: Option<String>,

    /// Print the forward-pass activation diagram before the classification
    #[arg(short, long, default_value_t = false)]
    activations: bool,
}

impl PredictArgs {
    pub fn run(self) -> Result<(), Box<dyn Error>> {
        let predictor = Predictor::load(&self.model)?;
        loaded(&predictor);

        let input_size = predictor.network.input_size();

        let instance = match self.instance {
            Some(instance_file) => {
                let instance = Instance::load(&instance_file)?;
                loaded(&instance);
                instance
            }
            None => read_instance(input_size)?,
        };

        if self.activations {
            let diagram =
                predictor.activation_diagram(instance.view(), &DiagramOptions::default())?;
            println!("{}", diagram.to_console());
        }

        let classification = predictor.classify_single(instance.view())?;
        evaluated(&classification);

        Ok(())
    }
}

/// Reads exactly `input_size` feature values from stdin, one per line, reprompting
/// on an unparseable line and erroring on premature end of input.
fn read_instance(input_size: usize) -> Result<Instance, Box<dyn Error>> {
    let mut values = Vec::with_capacity(input_size);

    while values.len() < input_size {
        prompt(values.len());

        let mut raw = String::new();
        if stdin().read_line(&mut raw)? == 0 {
            return Err(format!(
                "unexpected end of input: read {} of {input_size} features",
                values.len()
            )
            .into());
        }

        match raw.trim().parse::<f32>() {
            Ok(value) => values.push(value),
            Err(err) => error!("{err}"),
        }
    }

    println!();

    Ok(Instance::new(Array1::from_vec(values)))
}
