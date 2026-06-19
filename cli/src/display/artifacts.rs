use super::{Describe, Named, rows};
use crate::path::PathExt;
use std::path::PathBuf;

/// The files a command wrote to disk, each a `(label, path)` pair.
pub(crate) struct Artifacts {
    items: Vec<(&'static str, PathBuf)>,
}

impl Artifacts {
    /// Build an `Artifacts` holding a single `(label, path)` pair.
    pub(crate) fn single(label: &'static str, path: PathBuf) -> Self {
        Self {
            items: vec![(label, path)],
        }
    }

    /// Append a `(label, path)` pair, returning `&mut Self` to chain further calls.
    pub(crate) fn add(&mut self, label: &'static str, path: PathBuf) -> &mut Self {
        self.items.push((label, path));
        self
    }
}

impl<const N: usize> From<[(&'static str, PathBuf); N]> for Artifacts {
    fn from(items: [(&'static str, PathBuf); N]) -> Self {
        Self {
            items: items.into(),
        }
    }
}

impl Named for Artifacts {
    const NAME: &'static str = "ARTIFACTS";
}

impl Describe for Artifacts {
    fn describe(&self) -> String {
        let entries: Vec<(&str, String)> = self
            .items
            .iter()
            .map(|(label, path)| (*label, path.to_relative().display().to_string()))
            .collect();

        rows(&entries)
    }
}
