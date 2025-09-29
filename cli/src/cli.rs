use crate::commands::Command::*;
use crate::commands::EncodeCommand::{Img, ImgDir};
use crate::commands::{Command, DistributionOption};
use crate::plot::chart;
use crate::plot::gif::DecisionBoundaryView;
use crate::progression::Progression;
use crate::{display_info, display_initialization, display_success, display_warning};
use colored::Colorize;
use ndarray::{Array1, ArrayView2};
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand::prelude::StdRng;
use nrn::accuracies::{Accuracy, BINARY_ACCURACY, MULTI_CLASS_ACCURACY};
use nrn::activations::{RELU, SIGMOID, SOFTMAX};
use nrn::data::scalers::{Scaler, ScalerMethod};
use nrn::data::synth::{DatasetGenerator, RingDataset, UniformDataset};
use nrn::data::vectorizers::{ImageEncoder, VectorEncoder};
use nrn::data::{Dataset, SplitDataset};
use nrn::io::bytes::secure_read;
use nrn::io::data::{SplitDatasetExt, load_inputs, save_inputs};
use nrn::io::scalers::ScalerRecord;
use nrn::loss_functions::{CROSS_ENTROPY_LOSS, LossFunction};
use nrn::model::{NeuralNetwork, NeuronLayerSpec};
use nrn::optimizers::{Optimizer, StochasticGradientDescent};
use nrn::training::History;
use nrn::io::classes::extract_classes;
use std::cmp::Ordering::Equal;
use std::error::Error;
use std::fs::read_dir;
use std::io::stdin;
use std::iter::once;
use std::path::Path;
use std::sync::{Arc, Mutex};

fn load_dataset(filename: &str) -> Result<SplitDataset, Box<dyn Error>> {
    let dataset = SplitDataset::load(filename)?;

    display_initialization!(
        "Dataset loaded ({} features, {} training samples, {} test samples)",
        dataset.train.n_features().to_string().yellow(),
        dataset.train.n_samples().to_string().yellow(),
        dataset.test.n_samples().to_string().yellow()
    );

    Ok(dataset)
}

/// Saves the dataset to a file with the specified filename.
fn save_dataset(dataset: &SplitDataset, filename: &str) -> Result<(), Box<dyn Error>> {
    dataset.save(&filename)?;

    display_success!(
        "{} at {} {}",
        "Dataset saved".bright_green(),
        filename.bright_blue().italic(),
        "(HDF5)".italic().dimmed()
    );

    Ok(())
}

fn save_scaler(scaler: ScalerMethod, filename: &str) -> Result<(), Box<dyn Error>> {
    let record: ScalerRecord = scaler.into();
    record.save(filename)?;

    display_success!(
        "{} at {} {}",
        "Scaler saved".bright_green(),
        filename.bright_blue().italic(),
        "(JSON)".italic().dimmed()
    );

    Ok(())
}

fn load_scaler(filename: &str) -> Result<ScalerMethod, Box<dyn Error>> {
    let scaler: ScalerMethod = ScalerRecord::load(filename)?.into();

    display_initialization!("Scaler loaded ({})", scaler.name().yellow());

    Ok(scaler)
}

/// Plots the dataset if it has exactly two features and saves the plot to a file.
fn plot_dataset(
    filename: &str,
    dataset: &Dataset,
    width: u32,
    height: u32,
) -> Result<(), Box<dyn Error>> {
    if dataset.n_features() == 2 {
        chart::of_data(&filename, &dataset, width, height)?;
        display_success!(
            "{} at {} {}",
            "Plot saved".bright_green(),
            filename.bright_blue().italic(),
            "(PNG)".italic().dimmed()
        );
    } else {
        display_warning!("Plotting is only available for datasets with exactly two features");
    }

    Ok(())
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

fn select_accuracy(max_label: usize) -> Arc<dyn Accuracy> {
    if max_label > 1 {
        MULTI_CLASS_ACCURACY.clone()
    } else {
        BINARY_ACCURACY.clone()
    }
}

fn initialize_model(input_size: usize, layer_specs: &Vec<NeuronLayerSpec>) -> NeuralNetwork {
    let model = NeuralNetwork::initialization(input_size, &layer_specs);

    display_initialization!("Neural network initialized ({})", model.summary().yellow());

    model
}

fn load_model(filename: &str) -> Result<NeuralNetwork, Box<dyn Error>> {
    let model = NeuralNetwork::load(filename)?;

    display_initialization!("Neural network loaded ({})", model.summary().yellow());

    Ok(model)
}

fn save_model(filename: &str, model: &NeuralNetwork) -> Result<(), Box<dyn Error>> {
    model.save(filename)?;

    display_success!(
        "{} at {} {}",
        "Model saved".bright_green(),
        filename.bright_blue().italic(),
        "(HDF5)".italic().dimmed()
    );

    Ok(())
}

fn load_training_history(filename: &str) -> Result<History, Box<dyn Error>> {
    let history = History::load(filename)?;

    assert!(
        history.model.len() > 2,
        "Training history must contain more than two checkpoints to plot."
    );

    display_initialization!(
        "Training history loaded ({} checkpoints, one every {} epochs)",
        history.model.len().to_string().yellow(),
        history.interval.to_string().yellow()
    );

    Ok(history)
}

fn save_training_history(filename: &str, history: &History) -> Result<(), Box<dyn Error>> {
    history.save(filename)?;

    display_success!(
        "{} at {} {}",
        "Training history saved".bright_green(),
        filename.bright_blue().italic(),
        "(HDF5)".italic().dimmed()
    );

    Ok(())
}

pub(crate) fn handle(command: Command) -> Result<(), Box<dyn Error>> {
    match command {
        // 🗂️ DATASET GENERATION
        Synth {
            seed,
            distribution,
            samples,
            features,
            clusters,
            train_ratio,
            min: feature_min,
            max: feature_max,
            plot,
        } => {
            // 🗂️ GENERATE THE DATASET
            let generator: Arc<dyn DatasetGenerator> = match distribution {
                DistributionOption::Uniform => Arc::new(UniformDataset {
                    n_samples: samples,
                    n_features: features,
                    n_clusters: clusters,
                    feature_min,
                    feature_max,
                }),
                DistributionOption::Ring => Arc::new(RingDataset {
                    n_samples: samples,
                    n_features: features,
                    n_clusters: clusters,
                    feature_min,
                    feature_max,
                }),
            };

            let dataset = generator.generate(seed);

            let split_dataset = dataset.split(train_ratio);

            display_success!(
                "{} ({} features, {} samples -> {} training, {} test)",
                "Dataset generated".bright_green(),
                dataset.n_features().to_string().yellow(),
                dataset.n_samples().to_string().yellow(),
                split_dataset.train.n_samples().to_string().yellow(),
                split_dataset.test.n_samples().to_string().yellow()
            );

            let filename = format!(
                "{}-c{}-f{}-n{}-seed{}",
                distribution.to_string(),
                clusters,
                dataset.n_features(),
                dataset.n_samples(),
                seed
            );

            save_dataset(&split_dataset, &filename)?;

            if plot {
                plot_dataset(&filename, &dataset, 800, 600)?;
            }
        }
        Encode { subcommand } => {
            match subcommand {
                // 🖼️ IMAGE DATASET ENCODING
                ImgDir {
                    input,
                    output,
                    seed,
                    grayscale,
                    train_ratio,
                    shape,
                } => {
                    // Define categories by iterating over directories
                    let root = Path::new(&input);

                    let classes = extract_classes(&root)?;

                    display_info!(
                        "{}: {}",
                        "Classes found".bright_cyan(),
                        classes
                            .iter()
                            .map(|(name, label)| format!(
                                "{} (as {})",
                                name.bright_blue(),
                                label.to_string().yellow()
                            ))
                            .collect::<Vec<_>>()
                            .join(", ")
                    );

                    let mut data = Vec::new();
                    let mut labels = Vec::new();

                    for (i, (category, label)) in classes.iter().enumerate() {
                        let total_img = read_dir(&root.join(category))?
                            .filter_map(Result::ok)
                            .count();

                        let progression = Progression::new(
                            total_img,
                            format!(
                                "Encoding category [{}/{}]: {}",
                                i + 1,
                                classes.len(),
                                category.bright_blue()
                            ),
                        );

                        let encoder = ImageEncoder {
                            img_shape: (shape, shape),
                            grayscale,
                        };

                        for entry in read_dir(&root.join(category))?.filter_map(Result::ok) {
                            progression.inc();

                            let img = secure_read(entry.path())?;

                            if let Ok(img) = encoder.encode(&img) {
                                data.push(img);
                                labels.push(label.to_owned());
                            }
                        }

                        progression.done();
                    }

                    if data.is_empty() {
                        return Err("No images found in the specified directory".into());
                    }

                    let mut rng = StdRng::seed_from_u64(seed);
                    let dataset = Dataset::from_vec(&mut rng, data, labels)?;

                    let split_dataset = dataset.split(train_ratio);

                    display_success!("{}", "Image dataset encoded".bright_green());

                    save_dataset(&split_dataset, &output)?;
                }
                // 🖼️ SINGLE IMAGE ENCODING
                Img {
                    input,
                    output,
                    grayscale,
                    shape,
                } => {
                    let encoder = ImageEncoder {
                        img_shape: (shape, shape),
                        grayscale,
                    };

                    let image = encoder.encode(&secure_read(Path::new(&input))?)?;

                    save_inputs(&output, &image)?;

                    display_success!("{}", "Image encoded".bright_green());
                }
            }
        }
        // 📊 DATASET SCALING
        Scale {
            dataset: filename,
            scaling,
            plot,
        } => {
            let mut split_dataset = load_dataset(&filename)?;

            let scaler = scaling.fit(split_dataset.train.features.view());
            split_dataset.scale_inplace(&scaler);

            display_success!(
                "{} with {}",
                "Dataset scaled".bright_green(),
                scaler.name().yellow()
            );

            save_dataset(&split_dataset, &format!("scaled-{}", filename))?;
            save_scaler(
                scaler,
                &format!("scaler-{}", filename.trim_end_matches(".h5")),
            )?;

            if plot {
                plot_dataset(
                    &format!("scaled-{}", filename),
                    &split_dataset.unsplit(),
                    800,
                    600,
                )?;
            }
        }
        // 🧠 NEURAL NETWORK TRAINING
        Train {
            dataset,
            epochs,
            checkpoint_interval,
            model: model_file,
            layers: hidden_layers,
            learning_rate,
            max_norm,
        } => {
            let loss_function: Arc<dyn LossFunction> = CROSS_ENTROPY_LOSS.clone();
            let optimizer: Arc<Mutex<dyn Optimizer>> =
                Arc::new(Mutex::new(StochasticGradientDescent::new(learning_rate)));

            let SplitDataset { train, test } = load_dataset(&dataset)?;

            let accuracy: Arc<dyn Accuracy> = select_accuracy(train.max_label());

            // 🧠 NEURAL NETWORK INITIALIZATION
            let (train_inputs, train_targets) = train.to_model_shape();
            let (test_inputs, test_targets) = test.to_model_shape();
            let train_targets = train_targets.view();
            let test_targets = test_targets.view();

            let layer_specs: Vec<NeuronLayerSpec> = hidden_layers
                .unwrap_or_default()
                .into_iter()
                .map(|neurons| NeuronLayerSpec {
                    neurons,
                    activation: RELU.clone(),
                })
                .chain(once(create_output_layer(train.max_label())))
                .collect();

            let mut model = model_file
                .iter()
                .find_map(|file| load_model(file).ok())
                .unwrap_or_else(|| initialize_model(train_inputs.nrows(), &layer_specs));

            // 👨‍🎓 TRAINING LOOP
            let mut history: Option<History> = History::by_interval(checkpoint_interval, epochs);

            display_info!(
                "{} -- {} epochs, learning rate: {}",
                "Training".bright_cyan(),
                epochs.to_string().yellow(),
                learning_rate.to_string().yellow()
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

            let progression = Progression::new(epochs, "Training");
            for epoch in progression.iter() {
                let train_predictions = model.train(
                    train_inputs,
                    train_targets,
                    &loss_function,
                    &optimizer,
                    max_norm,
                );

                if let Some(ref mut history) = history {
                    let epoch_number = epoch + 1;
                    if epoch_number % history.interval == 0 || epoch_number == epochs {
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
            let model_file = format!("model-{}", dataset);
            save_model(&model_file, &model)?;

            // 🗂️ SAVE THE TRAINING HISTORY
            if let Some(ref history) = history {
                save_training_history(&format!("training-{}", model_file), history)?;
            }
        }
        // 🔮 PREDICTION
        Predict {
            model,
            input: input_file,
            scaler,
        } => {
            let model = load_model(&model)?;

            let scaler: Option<ScalerMethod> = scaler.iter().find_map(|s| load_scaler(s).ok());

            let mut input = if let Some(input_file) = input_file {
                let input = load_inputs(&input_file)?;

                display_initialization!(
                    "Input data loaded from {}",
                    input_file.bright_blue().italic()
                );

                input
            } else {
                let mut inputs = Vec::with_capacity(model.input_size());

                loop {
                    println!(
                        "{}[{}]:",
                        "Input".bold().bright_blue(),
                        inputs.len().to_string().yellow(),
                    );

                    let mut raw = String::new();
                    stdin().read_line(&mut raw)?;

                    match raw.trim().parse::<f32>() {
                        Ok(val) => inputs.push(val),
                        Err(err) => {
                            eprintln!("{}", err.to_string().red());
                        }
                    }

                    if inputs.len() >= inputs.capacity() {
                        break;
                    }
                }

                Array1::from_vec(inputs)
            };

            if let Some(ref scaler) = scaler {
                scaler.apply_single_inplace(input.view_mut());
            }

            let predictions = model.predict_single(input.view());

            let mut result: Vec<(usize, f32)> = if predictions.len() == 1 {
                vec![(0, 1.0 - predictions[0]), (1, predictions[0])]
            } else {
                predictions
                    .iter()
                    .enumerate()
                    .map(|(index, &value)| (index, value))
                    .collect::<Vec<_>>()
            };

            result.sort_by(|(_, p1), (_, p2)| p2.partial_cmp(&p1).unwrap_or(Equal));

            println!(
                "{} for {}\n|> {}",
                "Predictions".bold().bright_green(),
                input.to_string().yellow(),
                result
                    .iter()
                    .map(|(index, value)| format!(
                        "{}: {:.2}%",
                        index.to_string().bright_blue(),
                        value * 100.0
                    ))
                    .collect::<Vec<_>>()
                    .join("\n|> ")
            )
        }
        // 📊 PLOT TRAINING HISTORY
        Plot {
            history: history_file,
            dataset,
            frames,
            width,
            height,
        } => {
            let history = load_training_history(&history_file)?;

            chart::of_history(
                &format!("loss-{}", history_file),
                width,
                height,
                &[("Loss", &history.loss)],
            )?;

            chart::of_history(
                &format!("accuracy-{}", history_file),
                width,
                height,
                &[
                    ("Train Accuracy", &history.train_accuracy),
                    ("Test Accuracy", &history.test_accuracy),
                ],
            )?;

            display_success!(
                "{} at {} {}",
                "Training history plots saved".bright_green(),
                history_file.bright_blue().italic(),
                "(PNG)".italic().dimmed()
            );

            if let Some(dataset) = dataset {
                let dataset = SplitDataset::load(&dataset)?;

                if dataset.train.n_features() != 2 {
                    display_warning!(
                        "Decision boundary visualization is only available for datasets with exactly two features"
                    );
                    return Ok(());
                }

                let interval = history.model.len() / history.model.len().min(frames.into());

                let progression = Progression::new(
                    history.model.len(),
                    "Generating decision boundary animation",
                );

                let mut decision_boundaries =
                    DecisionBoundaryView::new(width, height, dataset.train);

                for step in progression.iter() {
                    let step_number = step + 1;

                    if step_number == 1
                        || step_number % interval == 0
                        || step_number == history.model.len()
                    {
                        decision_boundaries.add_frame(&history.model[step])?;
                    }
                }

                decision_boundaries.save(&format!("{}", history_file))?;

                display_success!(
                    "{} at {} {}",
                    "Decision boundary animation saved".bright_green(),
                    history_file.bright_blue().italic(),
                    "(GIF)".italic().dimmed()
                );
            }
        }
    }

    Ok(())
}
