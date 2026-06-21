use super::{Describe, Named, rows};
use nrn::classification::Classification;

impl Named for Classification {
    const NAME: &'static str = "CLASSIFICATION";
}

/// One dotted-leader row per class, ranked most likely first, each showing its
/// probability as a percentage.
impl Describe for Classification {
    fn describe(&self) -> String {
        let entries: Vec<(String, String)> = self
            .ranking()
            .iter()
            .map(|(class, probability)| {
                (
                    format!("Class {class}"),
                    format!("{:.2}%", probability * 100.0),
                )
            })
            .collect();

        let borrowed: Vec<(&str, String)> = entries
            .iter()
            .map(|(label, value)| (label.as_str(), value.clone()))
            .collect();

        rows(&borrowed)
    }
}
