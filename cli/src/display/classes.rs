use super::{Describe, Named, rows};
use nrn::data::Classes;

impl Named for Classes {
    const NAME: &'static str = "CLASSES";
}

impl Describe for Classes {
    /// One dotted-leader row per class, the class name labelling its integer
    /// label, ordered by name.
    fn describe(&self) -> String {
        let entries = self
            .iter()
            .map(|(name, label)| (name.as_str(), label.to_string()))
            .collect::<Vec<_>>();

        rows(&entries)
    }
}
