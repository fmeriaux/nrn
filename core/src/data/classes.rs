use std::collections::BTreeMap;

/// A set of named classes, each mapped to a contiguous 0-indexed label.
///
/// A pure value object holding the name → label mapping, ordered by class name.
/// The I/O layer owns how the classes are discovered (see [`Classes::scan`],
/// behind the `io` feature).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Classes(BTreeMap<String, usize>);

impl Classes {
    /// Builds the mapping from `(name, label)` pairs.
    pub fn new(classes: BTreeMap<String, usize>) -> Self {
        Self(classes)
    }

    /// The number of classes.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether no class is present.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Iterates over the `(name, label)` pairs, ordered by class name.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &usize)> {
        self.0.iter()
    }

    /// The name mapped to `label`.
    pub fn name_of(&self, id: usize) -> Option<&str> {
        self.0
            .iter()
            .find(|&(_, &l)| l == id)
            .map(|(name, _)| name.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Classes {
        Classes::new(BTreeMap::from([
            ("bird".to_string(), 0),
            ("cat".to_string(), 1),
        ]))
    }

    #[test]
    fn len_and_emptiness_reflect_the_mapping() {
        assert_eq!(sample().len(), 2);
        assert!(!sample().is_empty());
        assert!(Classes::new(BTreeMap::new()).is_empty());
    }

    #[test]
    fn iter_yields_pairs_ordered_by_name() {
        let pairs: Vec<(String, usize)> = sample()
            .iter()
            .map(|(name, &label)| (name.clone(), label))
            .collect();
        assert_eq!(pairs, vec![("bird".to_string(), 0), ("cat".to_string(), 1)]);
    }

    #[test]
    fn name_of_finds_the_matching_label() {
        assert_eq!(sample().name_of(0), Some("bird"));
        assert_eq!(sample().name_of(1), Some("cat"));
    }

    #[test]
    fn name_of_is_none_for_an_unmapped_label() {
        assert_eq!(sample().name_of(2), None);
    }
}
