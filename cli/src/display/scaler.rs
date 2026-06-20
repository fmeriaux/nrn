use super::{Describe, Named, theme};
use nrn::data::scalers::{Scaler, ScalerMethod};

impl Named for ScalerMethod {
    const NAME: &'static str = "SCALER";
}

impl Describe for ScalerMethod {
    fn describe(&self) -> String {
        theme::value(self.name())
    }
}
