use super::{Describe, Named, rows};
use nrn::data::{Dataset, Targets};

impl Named for Dataset {
    const NAME: &'static str = "DATASET";
}

impl Describe for Dataset {
    fn describe(&self) -> String {
        let mut entries = vec![("Features", self.feature_size().to_string())];

        if let Targets::ClassLabel(label) = self.targets() {
            entries.push(("Classes", label.class_count().to_string()));
        }

        entries.push(("Samples", self.sample_size().to_string()));

        if let Some(description) = self.info().and_then(|info| info.description.as_ref()) {
            entries.push(("Description", description.clone()));
        }

        rows(&entries)
    }
}

#[cfg(test)]
mod tests {
    use crate::display::Describe;
    use ndarray::{Array2, array};
    use nrn::data::{Dataset, DatasetInfo, Targets};

    fn dataset(info: Option<DatasetInfo>) -> Dataset {
        // 4 samples (rows), 2 features, 2 classes.
        Dataset::tabular(
            Array2::zeros((4, 2)),
            Targets::class_label(array![0u32, 1, 0, 1], None).unwrap(),
            info,
        )
        .unwrap()
    }

    #[test]
    fn describes_shape_and_omits_the_description_row_when_absent() {
        let described = dataset(None).describe();
        assert!(described.contains("Features"));
        assert!(described.contains("Classes"));
        assert!(described.contains("Samples"));
        assert!(!described.contains("Description"));
    }

    #[test]
    fn describes_the_dataset_information_when_present() {
        let described = dataset(Some(DatasetInfo {
            description: Some("synthetic ring (seed 42)".to_string()),
        }))
        .describe();
        assert!(described.contains("synthetic ring (seed 42)"));
    }
}
