use super::{Describe, block};
use nrn::data::scalers::{Scaler, ScalerMethod};

impl Describe for ScalerMethod {
    fn describe(&self) -> String {
        block("SCALER", &[("Method", self.name().to_string())])
    }
}
