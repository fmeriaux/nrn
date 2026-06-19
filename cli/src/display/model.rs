use super::{Describe, Named, rows};
use nrn::model::NeuralNetwork;

impl Named for NeuralNetwork {
    const NAME: &'static str = "NEURAL NETWORK";
}

impl Describe for NeuralNetwork {
    fn describe(&self) -> String {
        rows(&[
            ("Architecture", self.summary()),
            ("Inputs", self.input_size().to_string()),
            ("Classes", self.n_classes().to_string()),
        ])
    }
}
