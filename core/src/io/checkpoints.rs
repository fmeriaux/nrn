use crate::checkpoints::Checkpoints;
use crate::evaluation::{EvaluationSet, Evaluation};
use crate::io::h5;
use crate::model::NeuralNetwork;
use hdf5_metno::Group;
use std::io::Result;
use std::path::{Path, PathBuf};

impl Evaluation {
    fn save_to(&self, group: &Group) -> Result<()> {
        group
            .new_attr::<f32>()
            .create("accuracy")?
            .write_scalar(&self.accuracy)?;
        group
            .new_attr::<f32>()
            .create("loss")?
            .write_scalar(&self.loss)?;
        Ok(())
    }

    fn load_from(group: &Group) -> Result<Self> {
        let accuracy: f32 = group.attr("accuracy")?.read_scalar()?;
        let loss: f32 = group.attr("loss")?.read_scalar()?;
        Ok(Evaluation { accuracy, loss })
    }
}

impl EvaluationSet {
    fn save_to(&self, group: &Group) -> Result<()> {
        let train_group = group.create_group("train")?;
        self.train.save_to(&train_group)?;

        if let Some(validation) = &self.validation {
            let validation_group = group.create_group("validation")?;
            validation.save_to(&validation_group)?;
        }

        let test_group = group.create_group("test")?;
        self.test.save_to(&test_group)?;

        Ok(())
    }

    fn load_from(group: &Group) -> Result<Self> {
        let train_group = group.group("train")?;
        let train = Evaluation::load_from(&train_group)?;

        let validation = match group.group("validation") {
            Ok(validation_group) => Some(Evaluation::load_from(&validation_group)?),
            Err(_) => None,
        };

        let test_group = group.group("test")?;
        let test = Evaluation::load_from(&test_group)?;

        Ok(EvaluationSet {
            train,
            validation,
            test,
        })
    }
}

impl Checkpoints {
    /// Saves the checkpoints to an HDF5 file.
    /// # Arguments
    /// - `path`: The path to the file where the checkpoints will be saved.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<PathBuf> {
        let file = h5::create_file(path)?;

        file.new_attr::<usize>()
            .create("interval")?
            .write_scalar(&self.interval)?;

        let evaluations_group = file.create_group("evaluations")?;
        for (i, eval) in self.evaluations.iter().enumerate() {
            let checkpoint_group = evaluations_group.create_group(&format!("checkpoint{}", i))?;
            eval.save_to(&checkpoint_group)?;
        }

        let snapshots_group = file.create_group("snapshots")?;
        for (i, snapshot) in self.snapshots.iter().enumerate() {
            let checkpoint_group = snapshots_group.create_group(&format!("checkpoint{}", i))?;
            snapshot.save_to(&checkpoint_group)?;
        }

        Ok(PathBuf::from(file.filename()))
    }

    /// Loads the checkpoints from an HDF5 file.
    /// # Arguments
    /// - `path`: The path to the file to load the checkpoints from.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = h5::load_file(path)?;

        let interval: usize = file.attr("interval")?.read_scalar()?;

        let evaluations_group = file.group("evaluations")?;

        let mut evaluations = Vec::new();

        loop {
            let checkpoint_name = format!("checkpoint{}", evaluations.len());
            match evaluations_group.group(&checkpoint_name) {
                Ok(checkpoint_group) => {
                    evaluations.push(EvaluationSet::load_from(&checkpoint_group)?);
                }
                Err(_) => break,
            }
        }

        let snapshots_group = file.group("snapshots")?;

        let mut snapshots = Vec::new();

        loop {
            let checkpoint_name = format!("checkpoint{}", snapshots.len());
            match snapshots_group.group(&checkpoint_name) {
                Ok(checkpoint_group) => {
                    snapshots.push(NeuralNetwork::load_from_group(&checkpoint_group)?);
                }
                Err(_) => break,
            }
        }

        Ok(Checkpoints {
            interval,
            snapshots,
            evaluations,
        })
    }
}
