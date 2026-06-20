use super::{Describe, Named, theme};
use nrn::model::NeuralNetwork;

impl Named for NeuralNetwork {
    const NAME: &'static str = "NEURAL NETWORK";
}

impl Describe for NeuralNetwork {
    fn describe(&self) -> String {
        theme::value(self.summary())
    }
}
