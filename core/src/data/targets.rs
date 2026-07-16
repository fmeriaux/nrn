//! [`Targets`]: the dtype-carrying representation of a dataset's targets.

use crate::data::Classes;
use ndarray::{Array1, ArrayD, Axis};
use std::collections::HashSet;
use std::error::Error;

/// Class ids, contiguous over `0..n_classes`, with their names when known.
#[derive(Debug, Clone, PartialEq)]
pub struct ClassLabel {
    ids: Array1<u32>,
    names: Option<Classes>,
}

impl ClassLabel {
    /// Pairs class `ids` with their `names`, validating that the ids are
    /// contiguous over `0..n_classes` with at least two classes, and that
    /// `names`, when given, has exactly one name per class.
    ///
    /// # Errors
    /// - [`ClassLabelError::TooFewClasses`] when fewer than two distinct ids
    ///   are present.
    /// - [`ClassLabelError::NonContiguousIds`] when the ids are not
    ///   contiguous over `0..n_classes`.
    /// - [`ClassLabelError::NameCountMismatch`] when `names` is given but its
    ///   count differs from the number of classes.
    pub fn new(ids: Array1<u32>, names: Option<Classes>) -> Result<Self, ClassLabelError> {
        let n_classes = ids.iter().copied().collect::<HashSet<u32>>().len();
        if n_classes < 2 {
            return Err(ClassLabelError::TooFewClasses(n_classes));
        }
        let max_id = ids
            .iter()
            .copied()
            .max()
            .expect("n_classes >= 2 guarantees at least one id");
        if max_id as usize != n_classes - 1 {
            return Err(ClassLabelError::NonContiguousIds);
        }
        if let Some(names) = &names
            && names.len() != n_classes
        {
            return Err(ClassLabelError::NameCountMismatch {
                classes: n_classes,
                names: names.len(),
            });
        }
        Ok(Self { ids, names })
    }

    /// The class id of each sample.
    pub fn ids(&self) -> &Array1<u32> {
        &self.ids
    }

    /// The class names, when known.
    pub fn names(&self) -> Option<&Classes> {
        self.names.as_ref()
    }

    /// The number of distinct classes.
    pub fn n_classes(&self) -> usize {
        self.ids.iter().copied().collect::<HashSet<u32>>().len()
    }

    /// The sample indices whose class id equals `id`.
    pub fn indices_for(&self, id: u32) -> Vec<usize> {
        self.ids
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| (v == id).then_some(i))
            .collect()
    }
}

/// Errors returned when class ids and names cannot form a valid [`ClassLabel`].
#[derive(Debug, PartialEq, Eq)]
pub enum ClassLabelError {
    /// Class ids are not contiguous over `0..n_classes`.
    NonContiguousIds,
    /// Fewer than two distinct classes are present (carries the count found).
    TooFewClasses(usize),
    /// The number of class names differs from the number of classes.
    NameCountMismatch { classes: usize, names: usize },
}

impl std::fmt::Display for ClassLabelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClassLabelError::NonContiguousIds => {
                write!(f, "class ids must be contiguous over 0..n_classes")
            }
            ClassLabelError::TooFewClasses(n) => {
                write!(
                    f,
                    "class labels need at least 2 distinct classes, but found {n}"
                )
            }
            ClassLabelError::NameCountMismatch { classes, names } => write!(
                f,
                "{classes} classes were found, but {names} names were given"
            ),
        }
    }
}

impl Error for ClassLabelError {}

/// An array of finite values.
#[derive(Debug, Clone, PartialEq)]
pub struct Values(ArrayD<f32>);

impl Values {
    /// Wraps `values`, validating that every one is finite.
    ///
    /// # Errors
    /// [`NonFiniteValues`] when `values` contains a non-finite value.
    pub fn new(values: ArrayD<f32>) -> Result<Self, NonFiniteValues> {
        if !values.iter().all(|&v| v.is_finite()) {
            return Err(NonFiniteValues);
        }
        Ok(Self(values))
    }

    /// Borrows the underlying array.
    pub fn as_array(&self) -> &ArrayD<f32> {
        &self.0
    }
}

/// The reason [`Values::new`] rejected its input.
#[derive(Debug, PartialEq, Eq)]
pub struct NonFiniteValues;

impl std::fmt::Display for NonFiniteValues {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "values must all be finite")
    }
}

impl Error for NonFiniteValues {}

/// A dataset's targets: either class ids or real-valued outputs.
#[derive(Debug, Clone, PartialEq)]
pub enum Targets {
    ClassLabel(ClassLabel),
    Value(Values),
}

/// Which variant of [`Targets`] a dataset carries, independent of size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetKind {
    ClassLabel,
    Value,
}

impl std::fmt::Display for TargetKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TargetKind::ClassLabel => write!(f, "class-label"),
            TargetKind::Value => write!(f, "real-valued"),
        }
    }
}

impl Targets {
    /// Builds `ClassLabel` targets from `ids` and `names`.
    ///
    /// # Errors
    /// The same errors as [`ClassLabel::new`].
    pub fn class_label(ids: Array1<u32>, names: Option<Classes>) -> Result<Self, ClassLabelError> {
        Ok(Targets::ClassLabel(ClassLabel::new(ids, names)?))
    }

    /// Builds `Value` targets from `values`.
    ///
    /// # Errors
    /// The same errors as [`Values::new`].
    pub fn value(values: ArrayD<f32>) -> Result<Self, NonFiniteValues> {
        Ok(Targets::Value(Values::new(values)?))
    }

    /// The number of samples (the leading axis).
    pub fn n_samples(&self) -> usize {
        match self {
            Targets::ClassLabel(label) => label.ids.len(),
            Targets::Value(values) => values.as_array().len_of(Axis(0)),
        }
    }

    /// The per-sample shape, sample axis excluded: empty for `ClassLabel`, the
    /// trailing axes for `Value`.
    pub fn sample_shape(&self) -> &[usize] {
        match self {
            Targets::ClassLabel(_) => &[],
            Targets::Value(values) => &values.as_array().shape()[1..],
        }
    }

    /// Which variant these targets are.
    pub fn kind(&self) -> TargetKind {
        match self {
            Targets::ClassLabel(_) => TargetKind::ClassLabel,
            Targets::Value(_) => TargetKind::Value,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2, array};
    use std::collections::BTreeMap;

    fn classes(pairs: [(&str, usize); 2]) -> Classes {
        Classes::new(BTreeMap::from(
            pairs.map(|(name, id)| (name.to_string(), id)),
        ))
    }

    #[test]
    fn new_accepts_contiguous_ids_without_names() {
        let label = ClassLabel::new(array![0u32, 1, 2, 1], None).unwrap();
        assert_eq!(label.n_classes(), 3);
        assert_eq!(label.names(), None);
    }

    #[test]
    fn new_accepts_contiguous_ids_with_matching_names() {
        let names = classes([("cat", 0), ("dog", 1)]);
        let label = ClassLabel::new(array![0u32, 1], Some(names.clone())).unwrap();
        assert_eq!(label.names(), Some(&names));
    }

    #[test]
    fn new_rejects_a_single_class() {
        assert_eq!(
            ClassLabel::new(array![0u32, 0, 0], None),
            Err(ClassLabelError::TooFewClasses(1))
        );
    }

    #[test]
    fn new_rejects_non_contiguous_ids() {
        // {1, 2}: two classes, but max id 2 leaves a gap at 0.
        assert_eq!(
            ClassLabel::new(array![1u32, 2], None),
            Err(ClassLabelError::NonContiguousIds)
        );
    }

    #[test]
    fn new_rejects_a_name_count_mismatch() {
        let names = Classes::new(BTreeMap::from([("cat".to_string(), 0)]));
        assert_eq!(
            ClassLabel::new(array![0u32, 1], Some(names)),
            Err(ClassLabelError::NameCountMismatch {
                classes: 2,
                names: 1
            })
        );
    }

    #[test]
    fn indices_for_finds_matching_samples() {
        let label = ClassLabel::new(array![0u32, 1, 0, 2, 1], None).unwrap();
        assert_eq!(label.indices_for(0), vec![0, 2]);
        assert_eq!(label.indices_for(1), vec![1, 4]);
        assert_eq!(label.indices_for(2), vec![3]);
    }

    #[test]
    fn class_label_error_messages_are_descriptive() {
        assert_eq!(
            ClassLabelError::NonContiguousIds.to_string(),
            "class ids must be contiguous over 0..n_classes"
        );
        assert_eq!(
            ClassLabelError::TooFewClasses(1).to_string(),
            "class labels need at least 2 distinct classes, but found 1"
        );
        assert_eq!(
            ClassLabelError::NameCountMismatch {
                classes: 2,
                names: 1
            }
            .to_string(),
            "2 classes were found, but 1 names were given"
        );
    }

    #[test]
    fn class_label_reports_samples_and_sample_shape() {
        let targets = Targets::class_label(array![0u32, 1, 2, 1], None).unwrap();
        assert_eq!(targets.n_samples(), 4);
        assert_eq!(targets.sample_shape(), &[] as &[usize]);
        assert_eq!(targets.kind(), TargetKind::ClassLabel);
    }

    #[test]
    fn value_reports_samples_and_sample_shape() {
        let targets = Targets::value(Array2::<f32>::zeros((5, 3)).into_dyn()).unwrap();
        assert_eq!(targets.n_samples(), 5);
        assert_eq!(targets.sample_shape(), &[3]);
        assert_eq!(targets.kind(), TargetKind::Value);
    }

    #[test]
    fn value_rank1_has_an_empty_sample_shape() {
        let targets = Targets::value(Array1::<f32>::zeros(4).into_dyn()).unwrap();
        assert_eq!(targets.sample_shape(), &[] as &[usize]);
    }

    #[test]
    fn new_rejects_non_finite_values() {
        assert_eq!(
            Values::new(array![1.0f32, f32::NAN].into_dyn()),
            Err(NonFiniteValues)
        );
    }

    #[test]
    fn non_finite_values_error_message_is_descriptive() {
        assert_eq!(NonFiniteValues.to_string(), "values must all be finite");
    }

    #[test]
    fn kind_display_is_descriptive() {
        assert_eq!(TargetKind::ClassLabel.to_string(), "class-label");
        assert_eq!(TargetKind::Value.to_string(), "real-valued");
    }
}
