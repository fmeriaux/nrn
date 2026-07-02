use crate::actions::acquire_instance;
use crate::display::{evaluated, loaded};
use clap::Args;
use nrn::model::Predictor;
use nrn::plot::DiagramOptions;
use std::error::Error;

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

        let instance = acquire_instance(self.instance, predictor.network.input_size())?;

        let classification = if self.activations {
            let diagram =
                predictor.activation_diagram(instance.view(), &DiagramOptions::default())?;
            println!("{}", diagram.to_console());
            diagram.classification
        } else {
            predictor.classify_single(instance.view())?
        };
        evaluated(&classification);

        Ok(())
    }
}
