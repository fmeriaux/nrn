use super::{Describe, block};
use nrn::model::NeuralNetwork;

impl Describe for NeuralNetwork {
    fn describe(&self) -> String {
        block(
            "NEURAL NETWORK",
            &[
                ("Architecture", self.summary()),
                ("Inputs", self.input_size().to_string()),
                ("Classes", self.n_classes().to_string()),
            ],
        )
    }
}
