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
            Some(DatasetOrigin::Encoded { source }) => {
                entries.push(("Origin", format!("encoded from {source}")));
            }
            None => {}
        }

        rows(&entries)
    }
}

#[cfg(test)]
mod tests {
    use crate::display::Describe;
    use ndarray::{Array2, array};
    use nrn::data::{Dataset, DatasetOrigin};

    fn dataset(origin: Option<DatasetOrigin>) -> Dataset {
        // 4 samples (rows), 2 features, 2 classes.
        Dataset::tabular(Array2::zeros((4, 2)), array![0.0f32, 1.0, 0.0, 1.0], origin).unwrap()
    }

    #[test]
    fn describes_shape_and_omits_the_origin_row_when_absent() {
        let described = dataset(None).describe();
        assert!(described.contains("Features"));
        assert!(described.contains("Classes"));
        assert!(described.contains("Samples"));
        assert!(!described.contains("Origin"));
    }

    #[test]
    fn describes_a_synthetic_origin() {
        let described = dataset(Some(DatasetOrigin::Synthetic {
            distribution: "ring".to_string(),
            seed: 42,
        }))
        .describe();
        assert!(described.contains("synthetic ring (seed 42)"));
    }

    #[test]
    fn describes_an_encoded_origin() {
        let described = dataset(Some(DatasetOrigin::Encoded {
            source: "imgs".to_string(),
        }))
        .describe();
        assert!(described.contains("encoded from imgs"));
    }
}
