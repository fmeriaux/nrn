use super::{Describe, block};
use nrn::data::{Dataset, DatasetOrigin};

impl Describe for Dataset {
    fn describe(&self) -> String {
        let mut rows = vec![
            ("Features", self.n_features().to_string()),
            ("Classes", self.n_classes().to_string()),
            ("Samples", self.n_samples().to_string()),
        ];

        match self.origin() {
            Some(DatasetOrigin::Synthetic { distribution, seed }) => {
                rows.push(("Origin", format!("synthetic {distribution} (seed {seed})")));
            }
            Some(DatasetOrigin::Encoded { source }) => {
                rows.push(("Origin", format!("encoded from {source}")));
            }
            None => {}
        }

        block("DATASET", &rows)
    }
}
