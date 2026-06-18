/// Where a dataset came from: which producer built it.
///
/// A pure value object — the I/O layer owns how it is persisted. It records only
/// what the data cannot reveal; sizing (features, samples, classes) stays
/// derivable from the dataset itself.
#[derive(Debug, Clone, PartialEq)]
pub enum DatasetOrigin {
    /// Produced by a synthetic generator, reproducible from `distribution` + `seed`.
    Synthetic { distribution: String, seed: u64 },
    /// Encoded from a directory of images at `source`.
    Encoded { source: String },
}

impl DatasetOrigin {
    /// A deterministic slug naming this origin: the distribution and seed for a
    /// synthetic dataset, the source name for an encoded one.
    pub fn label(&self) -> String {
        match self {
            DatasetOrigin::Synthetic { distribution, seed } => format!("{distribution}-seed{seed}"),
            DatasetOrigin::Encoded { source } => source.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_label_carries_distribution_and_seed() {
        let origin = DatasetOrigin::Synthetic {
            distribution: "spiral".to_string(),
            seed: 42,
        };
        assert_eq!(origin.label(), "spiral-seed42");
    }

    #[test]
    fn encoded_label_is_the_source_name() {
        let origin = DatasetOrigin::Encoded {
            source: "mnist".to_string(),
        };
        assert_eq!(origin.label(), "mnist");
    }
}
