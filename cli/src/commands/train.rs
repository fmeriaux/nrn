use std::error::Error;
use std::iter::once;
use std::sync::{Arc, Mutex};
use clap::Args;
use colored::Colorize;
use ndarray::ArrayView2;
use nrn::accuracies::{Accuracy, BINARY_ACCURACY, MULTI_CLASS_ACCURACY};
use nrn::activations::{RELU, SIGMOID, SOFTMAX};
use nrn::data::SplitDataset;
use nrn::loss_functions::{LossFunction, CROSS_ENTROPY_LOSS};
use nrn::model::{NeuralNetwork, NeuronLayerSpec};
use nrn::optimizers::{Optimizer, StochasticGradientDescent};
use nrn::training::History;
use crate::{actions, display_info, display_initialization, display_success};
use crate::progression::Progression;

#[derive(Args, Debug)]
pub struct TrainArgs {
    /// Name of the dataset to train on
    dataset: String,

    /// Provide a pre-trained model to continue training, if not provided, a new model will be initialized
    #[arg(short, long)]
    model: Option<String>,

    /// The number of epochs to train the model
    #[arg(short, long)]
    epochs: usize,

    #[arg(short = 'k', long, default_value_t = 10)]
    /// Specify the checkpoint interval for saving the model state,
    /// if set to 0, no checkpoints will be saved
    checkpoint_interval: usize,

    /// Specify the hidden layers of the model when a new model is initialized
    #[arg(long, value_delimiter = ',', conflicts_with = "model")]
    layers: Option<Vec<usize>>,

    /// Specify the learning rate for the training process
    #[arg(long, default_value_t = 0.001)]
    learning_rate: f32,

    /// Specify the maximum norm for gradient clipping
    #[arg(long, default_value_t = 1.0)]
    max_norm: f32,
}

impl TrainArgs {
    pub fn run(self) -> Result<(), Box<dyn Error>> {
        let loss_function: Arc<dyn LossFunction> = CROSS_ENTROPY_LOSS.clone();
        let optimizer: Arc<Mutex<dyn Optimizer>> =
            Arc::new(Mutex::new(StochasticGradientDescent::new(self.learning_rate)));

        let SplitDataset { train, test } = actions::load_dataset(&self.dataset)?;

        let accuracy: Arc<dyn Accuracy> = select_accuracy(train.max_label());

        // 🧠 NEURAL NETWORK INITIALIZATION
        let (train_inputs, train_targets) = train.to_model_shape();
        let (test_inputs, test_targets) = test.to_model_shape();
        let train_targets = train_targets.view();
        let test_targets = test_targets.view();

        let layer_specs: Vec<NeuronLayerSpec> = self.layers
            .unwrap_or_default()
            .into_iter()
            .map(|neurons| NeuronLayerSpec {
                neurons,
                activation: RELU.clone(),
            })
            .chain(once(create_output_layer(train.max_label())))
            .collect();

        let mut model = self.model
            .iter()
            .find_map(|file| actions::load_model(file).ok())
            .unwrap_or_else(|| initialize_model(train_inputs.nrows(), &layer_specs));

        // 👨‍🎓 TRAINING LOOP
        let mut history: Option<History> =
            History::by_interval(self.checkpoint_interval, self.epochs);

        display_info!(
                    "{} -- {} epochs, learning rate: {}",
                    "Training".bright_cyan(),
                    self.epochs.to_string().yellow(),
                    self.learning_rate.to_string().yellow()
                );

        // Closure to record the training history at each checkpoint
        let record_history =
            |model: &NeuralNetwork,
             history: &mut History,
             train_predictions: ArrayView2<f32>| {
                let test_predictions = model.predict(test_inputs);
                history.checkpoint(
                    model,
                    &loss_function,
                    &accuracy,
                    train_predictions,
                    train_targets,
                    test_predictions.view(),
                    test_targets,
                );
            };

        if let Some(ref mut history) = history {
            display_info!(
                        "{} -- {} checkpoints will be recorded, one every {} epochs",
                        "History".bright_cyan(),
                        history.model.capacity().to_string().yellow(),
                        history.interval.to_string().yellow()
                    );

            let train_activations = model.predict(train_inputs);
            record_history(&model, history, train_activations.view());
        };

        let progression = Progression::new(self.epochs, "Training");
        for epoch in progression.iter() {
            let train_predictions = model.train(
                train_inputs,
                train_targets,
                &loss_function,
                &optimizer,
                self.max_norm,
            );

            if let Some(ref mut history) = history {
                let epoch_number = epoch + 1;
                if epoch_number % history.interval == 0 || epoch_number == self.epochs {
                    record_history(&model, history, train_predictions.view());
                }
            }
        }

        let train_predictions = model.predict(train_inputs);
        let train_predictions = train_predictions.view();

        display_success!(
                    "{} -- Loss: {} -- Train Accuracy: {} -- Test Accuracy: {}",
                    "Training completed".bright_green(),
                    loss_function
                        .compute(train_predictions, train_targets)
                        .to_string()
                        .yellow(),
                    accuracy
                        .compute(train_predictions, train_targets)
                        .to_string()
                        .yellow(),
                    accuracy
                        .compute(model.predict(test_inputs).view(), test_targets)
                        .to_string()
                        .yellow()
                );

        // 🗂️ SAVE THE TRAINED NETWORK
        let model_file = format!("model-{}", self.dataset);
        actions::save_model(&model_file, &model)?;

        // 🗂️ SAVE THE TRAINING HISTORY
        if let Some(ref history) = history {
            actions::save_training_history(&format!("training-{}", model_file), history)?;
        }

        Ok(())
    }
}

fn select_accuracy(max_label: usize) -> Arc<dyn Accuracy> {
    if max_label > 1 {
        MULTI_CLASS_ACCURACY.clone()
    } else {
        BINARY_ACCURACY.clone()
    }
}

fn create_output_layer(max_label: usize) -> NeuronLayerSpec {
    if max_label > 1 {
        NeuronLayerSpec {
            neurons: max_label + 1,
            activation: SOFTMAX.clone(),
        }
    } else {
        NeuronLayerSpec {
            neurons: 1,
            activation: SIGMOID.clone(),
        }
    }
}

fn initialize_model(input_size: usize, layer_specs: &Vec<NeuronLayerSpec>) -> NeuralNetwork {
    let model = NeuralNetwork::initialization(input_size, &layer_specs);

    display_initialization!("Neural network initialized ({})", model.summary().yellow());

    model
}