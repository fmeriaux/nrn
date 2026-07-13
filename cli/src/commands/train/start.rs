use super::DivergedRun;
use super::args::TrainArgs;
use super::callbacks::{ConsoleMonitor, ModelSaver};
use crate::display::{Spinner, initialized, loaded, recording, show};
use crate::path::PathExt;
use clap::Args;
use nrn::activations::{IDENTITY, RELU};
use nrn::data::Dataset;
use nrn::io::model::config::ModelConfigRecord;
use nrn::io::model::hyperparams::HyperParametersRecord;
use nrn::io::model::network::NetworkConfigRecord;
use nrn::io::model::run::{TrainingMeta, TrainingRun};
use nrn::io::model::scalers::ScalerRecord;
use nrn::model::{NetworkConfig, NeuralNetwork, Predictor};
use nrn::task::Task;
use nrn::training::Callbacks;
use std::error::Error;
use std::io::{Error as IoError, ErrorKind};
use std::path::Path;

#[derive(Args, Debug)]
pub struct StartArgs {
    /// Dataset to train on
    dataset: String,

    /// Load a pre-trained model to continue training (checkpoint count resets to 0)
    #[arg(short, long, conflicts_with = "layers")]
    model: Option<String>,

    /// Hidden layer sizes (comma-separated)
    #[arg(long, value_delimiter = ',', conflicts_with = "model")]
    layers: Option<Vec<usize>>,

    /// Overwrite an existing training run directory
    #[arg(long, default_value_t = false)]
    overwrite: bool,

    #[command(flatten)]
    hp: TrainArgs,
}

impl StartArgs {
    pub fn run(&self) -> Result<(), Box<dyn Error>> {
        let dataset_path = Path::new(&self.dataset);
        let dataset_name = dataset_path.file_stem_string();
        let run_dir = dataset_path.sibling("training-model");
        let model_name = format!("model-{dataset_name}");

        let dataset = Dataset::load(&self.dataset)?;
        loaded(&dataset);

        let task = Task::from_dataset(&dataset);
        task.validate_dataset(&dataset)?;
        show(&task);

        let hyperparameters = self.hp.to_hyperparameters(&task)?;

        let model = match &self.model {
            Some(path) => {
                let model = Predictor::load(path)?.network;
                loaded(&model);
                model
            }
            None => {
                let mut builder = NetworkConfig::builder(vec![dataset.n_features()]);
                for hidden in self.layers.clone().unwrap_or_default() {
                    builder = builder.dense(hidden, &RELU);
                }
                let config = builder.dense(task.output_size(), &IDENTITY).build();
                let model = NeuralNetwork::from_config(config, hyperparameters.seed())?;
                initialized(&model);
                model
            }
        };

        let spinner = Spinner::start("Preparing dataset");
        let data = hyperparameters.prepare(dataset, None)?;
        spinner.finish();

        let recorder = if hyperparameters.checkpoint_interval() > 0 {
            let meta = TrainingMeta {
                dataset: dataset_name,
                model: model_name.clone(),
                hyperparams: HyperParametersRecord::from(&hyperparameters),
            };
            let config = ModelConfigRecord {
                network: NetworkConfigRecord::from(&model),
                task: task.into(),
            };
            let scaler = data.scaler().cloned().map(ScalerRecord::from);
            let recorder =
                TrainingRun::create(&run_dir, &meta, &config, scaler.as_ref(), self.overwrite)
                    .map_err(overwrite_hint)?
                    .recorder();
            recording(&recorder);
            Some(recorder)
        } else {
            None
        };

        let callbacks = Callbacks::empty()
            .with(ConsoleMonitor::new(hyperparameters.clone(), None))
            .with(ModelSaver::new(
                &run_dir,
                &model_name,
                task,
                data.scaler().cloned(),
            ))
            .with_opt(recorder);

        hyperparameters
            .build(model, task, data, callbacks)?
            .train()?
            .into_result()
            .map_err(DivergedRun::from)?;

        Ok(())
    }
}

/// Adds the `--overwrite` remediation hint to an `AlreadyExists` error.
fn overwrite_hint(error: IoError) -> IoError {
    if error.kind() == ErrorKind::AlreadyExists {
        IoError::new(
            error.kind(),
            format!("{error}; use --overwrite to replace it"),
        )
    } else {
        error
    }
}

#[cfg(test)]
mod tests {
    use super::overwrite_hint;
    use std::io::{Error as IoError, ErrorKind};

    #[test]
    fn overwrite_hint_appends_remediation_for_already_exists() {
        let mapped = overwrite_hint(IoError::new(ErrorKind::AlreadyExists, "run dir exists"));
        assert_eq!(mapped.kind(), ErrorKind::AlreadyExists);
        assert!(mapped.to_string().contains("use --overwrite"));
    }

    #[test]
    fn overwrite_hint_passes_other_errors_through_unchanged() {
        let mapped = overwrite_hint(IoError::new(ErrorKind::PermissionDenied, "denied"));
        assert_eq!(mapped.kind(), ErrorKind::PermissionDenied);
        assert_eq!(mapped.to_string(), "denied");
    }
}
