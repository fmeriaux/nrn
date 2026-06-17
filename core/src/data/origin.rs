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
