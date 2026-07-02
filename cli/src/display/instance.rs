use super::{Describe, Named, error, prompt_feature, theme};
use ndarray::Array1;
use nrn::data::Instance;
use std::error::Error;
use std::io::stdin;

impl Named for Instance {
    const NAME: &'static str = "INSTANCE";
}

impl Describe for Instance {
    fn describe(&self) -> String {
        theme::value(self.values())
    }
}

/// Prompting for an [`Instance`] interactively from stdin.
pub(crate) trait PromptInstance {
    /// Prompts for exactly `input_size` feature values from stdin, one per line,
    /// reprompting on an unparseable line and erroring on premature end of input.
    fn prompt(input_size: usize) -> Result<Instance, Box<dyn Error>>;
}

impl PromptInstance for Instance {
    fn prompt(input_size: usize) -> Result<Instance, Box<dyn Error>> {
        let mut values = Vec::with_capacity(input_size);

        while values.len() < input_size {
            prompt_feature(values.len());

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
}
