use crate::commands::Command;
use crate::commands::Command::*;
use crate::commands::EncodeCommand::{Img, ImgDir};
use crate::core::activation::ActivationMethod::{ReLU, Sigmoid, Softmax};
use crate::core::encoder::{encode_image, extract_categories};
use crate::core::neuron_network::{NeuronLayerSpec, NeuronNetwork, accuracy, log_loss};
use crate::core::scaling::Scaler;
use crate::core::training_history::TrainingHistory;
use crate::hdf5::{load_inputs, save_inputs};
use crate::plot::DecisionBoundaryView;
use crate::progression::Progression;
use crate::synth::{Dataset, SplitDataset};
use crate::{display_info, display_initialization, display_success, display_warning, plot};
use colored::Colorize;
use ndarray::Array1;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand::prelude::StdRng;
use std::cmp::Ordering::Equal;
use std::fs::read_dir;
use std::io::stdin;
use std::iter::once;
use std::path::Path;

fn load_dataset(filename: &str) -> Result<SplitDataset, Box<dyn std::error::Error>> {
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
fn save_dataset(dataset: &SplitDataset, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
    dataset.save(&filename)?;

    display_success!(
        "{} at {} {}",
        "Dataset saved".bright_green(),
        filename.bright_blue().italic(),
        "(HDF5)".italic().dimmed()
    );

    Ok(())
}

fn save_scaler(scaler: &Scaler, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
    scaler.save(filename)?;

    display_success!(
        "{} at {} {}",
        "Scaler saved".bright_green(),
        filename.bright_blue().italic(),
        "(JSON)".italic().dimmed()
    );

    Ok(())
}

fn load_scaler(filename: &str) -> Result<Scaler, Box<dyn std::error::Error>> {
    let scaler = Scaler::load(filename)?;

    display_initialization!(
        "Scaler loaded ({})",
        scaler.to_method().to_string().yellow()
    );

    Ok(scaler)
}

/// Plots the dataset if it has exactly two features and saves the plot to a file.
fn plot_dataset(
    filename: &str,
    dataset: &Dataset,
    width: u32,
    height: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    if dataset.n_features() == 2 {
        plot::of_data(&filename, &dataset, width, height)?;
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
            activation: Softmax,
        }
    } else {
        NeuronLayerSpec {
            neurons: 1,
            activation: Sigmoid,
        }
    }
}

fn initialize_model(input_size: usize, layer_specs: &Vec<NeuronLayerSpec>) -> NeuronNetwork {
    let model = NeuronNetwork::initialization(input_size, &layer_specs);

    display_initialization!("Neural network initialized ({})", model.summary().yellow());

    model
}

fn load_model(filename: &str) -> Result<NeuronNetwork, Box<dyn std::error::Error>> {
    let model = NeuronNetwork::load(filename)?;

    display_initialization!("Neural network loaded ({})", model.summary().yellow());

    Ok(model)
}

fn save_model(filename: &str, model: &NeuronNetwork) -> Result<(), Box<dyn std::error::Error>> {
    model.save(filename)?;

    display_success!(
        "{} at {} {}",
        "Model saved".bright_green(),
        filename.bright_blue().italic(),
        "(HDF5)".italic().dimmed()
    );

    Ok(())
}

fn load_training_history(filename: &str) -> Result<TrainingHistory, Box<dyn std::error::Error>> {
    let history = TrainingHistory::load(filename)?;

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

fn save_training_history(
    filename: &str,
    history: &TrainingHistory,
) -> Result<(), Box<dyn std::error::Error>> {
    history.save(filename)?;

    display_success!(
        "{} at {} {}",
        "Training history saved".bright_green(),
        filename.bright_blue().italic(),
        "(HDF5)".italic().dimmed()
    );

    Ok(())
}

pub(crate) fn handle(command: Command) -> Result<(), Box<dyn std::error::Error>> {
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
            let dataset = Dataset::new(
                &distribution,
                seed,
                samples,
                features,
                clusters,
                feature_min,
                feature_max,
            );

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

                    let categories = extract_categories(&root)?;

                    display_info!(
                        "{}: {}",
                        "Categories found".bright_cyan(),
                        categories
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

                    for (i, (category, label)) in categories.iter().enumerate() {
                        let total_img = read_dir(&root.join(category))?
                            .filter_map(Result::ok)
                            .count();

                        let progression = Progression::new(
                            total_img,
                            format!(
                                "Encoding category [{}/{}]: {}",
                                i + 1,
                                categories.len(),
                                category.bright_blue()
                            ),
                        );

                        for entry in read_dir(&root.join(category))?.filter_map(Result::ok) {
                            progression.inc();

                            if let Ok(img) = encode_image((shape, shape), grayscale, &entry.path())
                            {
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
                    let dataset = Dataset::from_image_vec(&mut rng, data, labels)?;

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
                    let image = encode_image((shape, shape), grayscale, &input)?;

                    save_inputs(&output, &image.map(|&v| v as f32))?;

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

            let scaler = Scaler::fit_from_name(&scaling, &split_dataset.train.features);
            split_dataset.scale(&scaler);

            display_success!(
                "{} with {}",
                "Dataset scaled".bright_green(),
                scaling.to_string().yellow()
            );

            save_dataset(&split_dataset, &format!("scaled-{}", filename))?;
            save_scaler(
                &scaler,
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
            let SplitDataset { train, test } = load_dataset(&dataset)?;

            // 🧠 NEURAL NETWORK INITIALIZATION
            let (train_inputs, train_expectations) = train.to_model_shape();
            let (test_inputs, test_expectations) = test.to_model_shape();

            let layer_specs: Vec<NeuronLayerSpec> = hidden_layers
                .unwrap_or_default()
                .into_iter()
                .map(|neurons| NeuronLayerSpec {
                    neurons,
                    activation: ReLU,
                })
                .chain(once(create_output_layer(train.max_label())))
                .collect();

            let mut model = model_file
                .iter()
                .find_map(|file| load_model(file).ok())
                .unwrap_or_else(|| initialize_model(train_inputs.nrows(), &layer_specs));

            // 👨‍🎓 TRAINING LOOP
            let mut history: Option<TrainingHistory> =
                TrainingHistory::by_interval(checkpoint_interval, epochs);

            display_info!(
                "{} -- {} epochs, learning rate: {}",
                "Training".bright_cyan(),
                epochs.to_string().yellow(),
                learning_rate.to_string().yellow()
            );

            if let Some(ref history) = history {
                display_info!(
                    "{} -- {} checkpoints will be recorded, one every {} epochs",
                    "History".bright_cyan(),
                    history.model.capacity().to_string().yellow(),
                    history.interval.to_string().yellow()
                );
            };

            let progression = Progression::new(epochs, "Training");

            for epoch in progression.iter() {
                let train_activations =
                    model.train(&train_inputs, &train_expectations, learning_rate, max_norm);

                if let Some(ref mut history) = history {
                    let epoch_number = epoch + 1;
                    if epoch_number % history.interval == 0 || epoch_number == epochs {
                        let test_activations = model.predict(&test_inputs);

                        history.checkpoint(
                            &model,
                            &train_activations,
                            &train_expectations,
                            &test_activations,
                            &test_expectations,
                        );
                    }
                }
            }

            display_success!(
                "{} -- Loss: {} -- Train Accuracy: {} -- Test Accuracy: {}",
                "Training completed".bright_green(),
                log_loss(&model.predict(&train_inputs), &train_expectations)
                    .to_string()
                    .yellow(),
                accuracy(&model.predict(&train_inputs), &train_expectations)
                    .to_string()
                    .yellow(),
                accuracy(&model.predict(&test_inputs), &test_expectations)
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

            let scaler: Option<Scaler> = scaler.iter().find_map(|s| load_scaler(s).ok());

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
                input = scaler.apply(&input);
            }

            let predictions = model.predict_single(&input);

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

            plot::of_history(
                &format!("loss-{}", history_file),
                width,
                height,
                &[("Loss", &history.loss)],
            )?;

            plot::of_history(
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

                    if step_number % interval == 0 || step_number == history.model.len() {
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
