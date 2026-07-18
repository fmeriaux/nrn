use super::{Describe, Named, rows};
use ndarray::Axis;
use nrn::model::{Inference, Labels, Rank};

impl Named for Inference {
    const NAME: &'static str = "PREDICTION";
}

impl Describe for Inference {
    fn describe(&self) -> String {
        match self {
            Inference::Classification {
                rankings, labels, ..
            } => describe_classification(&rankings[0], labels.as_ref()),
            Inference::Values(activations) => {
                let output = activations.output();
                let sample = output.index_axis(Axis(output.ndim() - 1), 0);

                let entries: Vec<(String, String)> = sample
                    .iter()
                    .enumerate()
                    .map(|(index, value)| (format!("Output {index}"), format!("{value:.4}")))
                    .collect();

                rows(&entries)
            }
        }
    }
}

/// Ranked class rows, named by `labels` when known, with the winning class (rank 0) marked by
/// an arrow.
fn describe_classification(ranking: &[Rank], labels: Option<&Labels>) -> String {
    let entries: Vec<(String, String)> = ranking
        .iter()
        .enumerate()
        .map(|(rank, Rank { class_id, score })| {
            let percentage = format!("{:.2}%", score * 100.0);
            let value = if rank == 0 {
                format!("{percentage}  \u{25c0}")
            } else {
                percentage
            };
            let name = labels
                .unwrap_or(&Labels::default())
                .get_or_default(*class_id);
            (name, value)
        })
        .collect();

    rows(&entries)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array, IxDyn};
    use nrn::activations::RELU;
    use nrn::model::{ModelConfig, NetworkConfig, NeuralNetwork, Predictor};
    use nrn::task::Task;

    fn rank(class_id: usize, score: f32) -> Rank {
        Rank { class_id, score }
    }

    #[test]
    fn describe_marks_only_the_winning_class_with_an_arrow() {
        let text = describe_classification(&[rank(1, 0.7), rank(2, 0.2), rank(0, 0.1)], None);

        let winner = text.lines().find(|line| line.contains("Class 1")).unwrap();
        assert!(winner.contains('\u{25c0}'));
        let runner_up = text.lines().find(|line| line.contains("Class 2")).unwrap();
        assert!(!runner_up.contains('\u{25c0}'));
    }

    #[test]
    fn describe_names_classes_from_labels_when_known() {
        let labels = Labels::new(vec![
            "cat".to_string(),
            "dog".to_string(),
            "bird".to_string(),
        ]);
        let text =
            describe_classification(&[rank(1, 0.7), rank(0, 0.2), rank(2, 0.1)], Some(&labels));

        assert!(text.contains("dog"));
        assert!(text.contains("cat"));
        assert!(text.contains("bird"));
        assert!(!text.contains("Class"));
    }

    #[test]
    fn describe_falls_back_to_class_n_without_labels() {
        let text = describe_classification(&[rank(0, 0.9), rank(1, 0.1)], None);
        assert!(text.contains("Class 0"));
        assert!(text.contains("Class 1"));
    }

    #[test]
    fn describe_flattens_a_spatial_values_output_without_panicking() {
        let config = NetworkConfig::builder(vec![1, 4, 4])
            .conv2d(2, (3, 3), 1, 0, &RELU)
            .build();
        let model = NeuralNetwork::from_config(config, 0).unwrap();
        let task = Task::Regression {
            target_shape: vec![2, 2, 2],
        };
        let model_config = ModelConfig::new(task, None).unwrap();
        let predictor = Predictor::new(model, model_config, None);

        let instance = Array::from_shape_fn(IxDyn(&[1, 4, 4]), |idx| (idx[1] + idx[2]) as f32);
        let inference = predictor.infer_instance(instance.view()).unwrap();

        let text = inference.describe();
        assert_eq!(text.lines().count(), 8);
    }
}
