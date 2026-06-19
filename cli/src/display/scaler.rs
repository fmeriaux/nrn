use super::{Describe, Named, rows};
use nrn::data::scalers::{Scaler, ScalerMethod};

impl Named for ScalerMethod {
    const NAME: &'static str = "SCALER";
}

impl Describe for ScalerMethod {
    fn describe(&self) -> String {
        rows(&[("Method", self.name().to_string())])
    }
}
