use super::{Describe, Named, rows};
use nrn::data::scalers::Scaler;
use nrn::model::Predictor;

impl Named for Predictor {
    const NAME: &'static str = "PREDICTOR";
}

/// The network architecture and the scaler applied to raw inputs (or `none`).
impl Describe for Predictor {
    fn describe(&self) -> String {
        let scaler = self.scaler.as_ref().map_or("none", Scaler::name);

        rows(&[
            ("Architecture", self.network.summary()),
            ("Scaler", scaler.to_string()),
        ])
    }
}
