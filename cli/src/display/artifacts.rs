use super::{Describe, Named, column_width, theme};
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
    /// A lone artifact is its bare path; several are one file per line — a
    /// bullet, the written path padded to a column, then its role as a dim
    /// lowercase caption.
    fn describe(&self) -> String {
        if let [(_, path)] = self.items.as_slice() {
            return theme::value(path.to_relative().display());
        }

        let paths: Vec<String> = self
            .items
            .iter()
            .map(|(_, path)| path.to_relative().display().to_string())
            .collect();
        let width = column_width(paths.iter().map(String::as_str));

        self.items
            .iter()
            .zip(&paths)
            .map(|((label, _), path)| {
                format!(
                    "   {} {}{}  {}",
                    theme::caption("›"),
                    theme::value(path),
                    " ".repeat(width - path.len()),
                    theme::caption(label.to_lowercase()),
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}
