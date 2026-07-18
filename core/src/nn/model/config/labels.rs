//! [`Labels`]: a task's name vocabulary for its classes or multi-label positions.

use crate::data::Targets;

/// A name vocabulary for a task's classes or multi-label positions, indexed by id.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Labels(Box<[String]>);

impl Labels {
    /// Wraps `names`, indexed by id.
    pub fn new(names: Vec<String>) -> Self {
        Self(names.into_boxed_slice())
    }

    /// The name vocabulary carried by `targets`, when named class labels are present.
    pub fn from_targets(targets: &Targets) -> Option<Self> {
        match targets {
            Targets::ClassLabel(label) => label.names().map(|names| Labels::new(names.to_vec())),
            Targets::Value(_) => None,
        }
    }

    /// The number of names in the vocabulary.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the vocabulary names nothing.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The full vocabulary, indexed by id.
    pub fn names(&self) -> &[String] {
        &self.0
    }

    /// The name at `id`, or `"Class {id}"` when it doesn't exist.
    pub fn get_or_default(&self, id: usize) -> String {
        self.0
            .get(id)
            .cloned()
            .unwrap_or_else(|| format!("Class {id}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn from_targets_reads_the_class_label_vocabulary() {
        let names = vec!["cat".to_string(), "dog".to_string()];
        let targets = Targets::class_label(array![0u32, 1], Some(names.clone())).unwrap();
        assert_eq!(Labels::from_targets(&targets), Some(Labels::new(names)));
    }

    #[test]
    fn from_targets_is_none_without_names() {
        let targets = Targets::class_label(array![0u32, 1], None).unwrap();
        assert_eq!(Labels::from_targets(&targets), None);
    }

    #[test]
    fn from_targets_is_none_for_value_targets() {
        let targets = Targets::value(array![[0.0f32], [1.0]].into_dyn()).unwrap();
        assert_eq!(Labels::from_targets(&targets), None);
    }

    #[test]
    fn get_or_default_reads_the_name_at_its_id() {
        let labels = Labels::new(vec!["cat".to_string(), "dog".to_string()]);
        assert_eq!(labels.get_or_default(0), "cat");
        assert_eq!(labels.get_or_default(1), "dog");
    }

    #[test]
    fn get_or_default_falls_back_past_the_last_id() {
        let labels = Labels::new(vec!["cat".to_string()]);
        assert_eq!(labels.get_or_default(1), "Class 1");
    }

    #[test]
    fn default_is_an_empty_vocabulary() {
        assert_eq!(Labels::default(), Labels::new(vec![]));
    }

    #[test]
    fn len_and_is_empty_reflect_the_vocabulary_size() {
        assert_eq!(Labels::new(vec![]).len(), 0);
        assert!(Labels::new(vec![]).is_empty());
        assert_eq!(Labels::new(vec!["cat".to_string()]).len(), 1);
    }

    #[test]
    fn names_returns_the_full_vocabulary_in_id_order() {
        let labels = Labels::new(vec!["cat".to_string(), "dog".to_string()]);
        assert_eq!(labels.names(), ["cat".to_string(), "dog".to_string()]);
    }
}
