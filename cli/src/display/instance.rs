use super::{Describe, Named, theme};
use nrn::data::Instance;

impl Named for Instance {
    const NAME: &'static str = "INSTANCE";
}

impl Describe for Instance {
    fn describe(&self) -> String {
        theme::value(self.values())
    }
}
