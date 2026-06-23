use super::{Describe, Named, rows};
use nrn::data::{Dataset, DatasetOrigin};

impl Named for Dataset {
    const NAME: &'static str = "DATASET";
}

impl Describe for Dataset {
    fn describe(&self) -> String {
        let mut entries = vec![
            ("Features", self.n_features().to_string()),
            ("Classes", self.n_classes().to_string()),
            ("Samples", self.n_samples().to_string()),
        ];

        match self.origin() {
            Some(DatasetOrigin::Synthetic { distribution, seed }) => {
                entries.push(("Origin", format!("synthetic {distribution} (seed {seed})")));
            }
            Some(DatasetOrigin::Encoded { source, seed }) => {
                entries.push(("Origin", format!("encoded from {source} (seed {seed})")));
            }
            None => {}
        }

        rows(&entries)
    }
}
