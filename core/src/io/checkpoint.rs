use crate::evaluation::{Evaluation, EvaluationSet};
use crate::evaluation_history::{EpochEvaluation, EvaluationHistory};
use crate::io::json;
use crate::io::optimizer as optimizer_io;
use crate::io::path::PathExt;
use crate::io::scheduler as scheduler_io;
use crate::model::NeuralNetwork;
use crate::optimizers::{Optimizer, OptimizerState};
use crate::schedulers::{Scheduler, SchedulerState};
use crate::training::{CallbackResult, TrainerCallback};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::ErrorKind::InvalidData;
use std::io::{Error, Result};
use std::path::{Path, PathBuf};
use std::result::Result as StdResult;

#[derive(Serialize, Deserialize)]
struct CheckpointEvaluationSet {
    train: CheckpointEvaluation,
    #[serde(skip_serializing_if = "Option::is_none")]
    validation: Option<CheckpointEvaluation>,
    test: CheckpointEvaluation,
}

#[derive(Serialize, Deserialize)]
struct CheckpointEvaluation {
    loss: f32,
    accuracy: f32,
}

impl From<Evaluation> for CheckpointEvaluation {
    fn from(eval: Evaluation) -> Self {
        CheckpointEvaluation {
            loss: eval.loss,
            accuracy: eval.accuracy,
        }
    }
}

impl From<CheckpointEvaluation> for Evaluation {
    fn from(pair: CheckpointEvaluation) -> Self {
        Evaluation {
            loss: pair.loss,
            accuracy: pair.accuracy,
        }
    }
}

impl From<&EvaluationSet> for CheckpointEvaluationSet {
    fn from(eval: &EvaluationSet) -> Self {
        CheckpointEvaluationSet {
            train: eval.train.into(),
            validation: eval.validation.map(Into::into),
            test: eval.test.into(),
        }
    }
}

impl From<CheckpointEvaluationSet> for EvaluationSet {
    fn from(evals: CheckpointEvaluationSet) -> Self {
        EvaluationSet {
            train: evals.train.into(),
            validation: evals.validation.map(Into::into),
            test: evals.test.into(),
        }
    }
}

/// A reference to a checkpoint subdirectory, named `checkpoint-{epoch:06}`.
pub(super) struct CheckpointRef {
    pub(super) epoch: usize,
    pub(super) dir: PathBuf,
}

/// Scans `dir` for `checkpoint-*` subdirectories, sorted by their numeric epoch
/// (not lexically, so 10+ checkpoints sort correctly). Reads no other files.
pub(super) fn scan_checkpoints(dir: &Path) -> Result<Vec<CheckpointRef>> {
    let mut checkpoints: Vec<CheckpointRef> = fs::read_dir(dir)?
        .filter_map(StdResult::ok)
        .filter_map(|entry| {
            let path = entry.path();
            let name = path.file_name()?.to_str()?;
            if path.is_dir() {
                let epoch = name.strip_prefix("checkpoint-")?.parse::<usize>().ok()?;
                Some(CheckpointRef { epoch, dir: path })
            } else {
                None
            }
        })
        .collect();

    checkpoints.sort_by_key(|s| s.epoch);
    Ok(checkpoints)
}

/// Writes model checkpoints to a directory.
///
/// Each call to [`write`](CheckpointRecorder::write) creates a subdirectory
/// `checkpoint-{epoch:06}/` containing `model.safetensors`, `evaluations.json`,
/// and — when the optimizer or scheduler carry internal state — `optimizer.safetensors`
/// and/or `scheduler.json` respectively.
/// Implements [`TrainerCallback`]: [`on_checkpoint`](TrainerCallback::on_checkpoint)
/// writes a checkpoint. Obtained via [`TrainingRun::recorder`](crate::io::run::TrainingRun::recorder).
#[derive(Debug)]
pub struct CheckpointRecorder {
    dir: PathBuf,
}

impl CheckpointRecorder {
    /// Creates a recorder writing into `dir`. Obtained via
    /// [`TrainingRun::recorder`](crate::io::run::TrainingRun::recorder).
    pub(super) fn new(dir: PathBuf) -> Self {
        CheckpointRecorder { dir }
    }

    /// Writes `checkpoint-{epoch:06}/` containing the model weights, evaluations,
    /// and any optimizer/scheduler state.
    pub fn write(
        &self,
        model: &NeuralNetwork,
        optimizer: &dyn Optimizer,
        scheduler: &dyn Scheduler,
        evaluation: &EvaluationSet,
        epoch: usize,
    ) -> Result<()> {
        let checkpoint_dir = self.dir.join(format!("checkpoint-{epoch:06}"));
        fs::create_dir_all(&checkpoint_dir)?;

        model.save(checkpoint_dir.join("model"))?;

        json::save(
            &CheckpointEvaluationSet::from(evaluation),
            checkpoint_dir.join("evaluations"),
        )?;

        if let Some(state) = optimizer.to_state() {
            optimizer_io::save(&state, checkpoint_dir.join("optimizer"))?;
        }
        if let Some(state) = scheduler.to_state() {
            scheduler_io::save(&state, checkpoint_dir.join("scheduler"))?;
        }

        Ok(())
    }
}

impl TrainerCallback for CheckpointRecorder {
    fn on_checkpoint(
        &mut self,
        model: &NeuralNetwork,
        optimizer: &dyn Optimizer,
        scheduler: &dyn Scheduler,
        eval: &EvaluationSet,
        epoch: usize,
    ) -> CallbackResult {
        self.write(model, optimizer, scheduler, eval, epoch)
            .map_err(Into::into)
    }
}

/// Lazy read access to a directory of checkpoints written by [`CheckpointRecorder`].
///
/// [`CheckpointArchive::load`] only scans directory names — no files are read until
/// [`model_at`](CheckpointArchive::model_at) or
/// [`evaluation_history`](CheckpointArchive::evaluation_history) is called.
pub struct CheckpointArchive {
    entries: Vec<CheckpointRef>,
}

impl CheckpointArchive {
    /// Scans `dir` for `checkpoint-*` subdirectories, sorted by epoch.
    pub fn load<P: AsRef<Path>>(dir: P) -> Result<Self> {
        let dir = Path::combine_safe_with_cwd(dir)?;
        Ok(CheckpointArchive {
            entries: scan_checkpoints(&dir)?,
        })
    }

    /// Returns the number of checkpoints found.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns true if no checkpoints were found.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns the absolute epoch number of the checkpoint at `index`, from the
    /// scanned directory name (zero I/O).
    pub fn epoch_at(&self, index: usize) -> Option<usize> {
        self.entries.get(index).map(|s| s.epoch)
    }

    /// Returns the checkpoint entry at `index`, or an error if out of range.
    fn entry_at(&self, index: usize) -> Result<&CheckpointRef> {
        self.entries.get(index).ok_or_else(|| {
            Error::new(
                InvalidData,
                format!(
                    "checkpoint index {index} out of range (archive has {} checkpoints)",
                    self.entries.len()
                ),
            )
        })
    }

    /// Resolves an optional checkpoint state file. Checks that `name.ext` exists
    /// in the checkpoint at `index` and, if so, returns the loader-ready base
    /// path (`name`, without extension); `Ok(None)` when the file is absent.
    fn optional_file(&self, index: usize, name: &str, ext: &str) -> Result<Option<PathBuf>> {
        let base = self.entry_at(index)?.dir.join(name);
        Ok(base.with_extension(ext).exists().then_some(base))
    }

    /// Loads the model at position `index` from its checkpoint directory.
    pub fn model_at(&self, index: usize) -> Result<NeuralNetwork> {
        NeuralNetwork::load(self.entry_at(index)?.dir.join("model"))
    }

    /// Loads the optimizer state at position `index`, or `Ok(None)` if the
    /// checkpoint has no `optimizer.safetensors` (stateless optimizer, or a
    /// checkpoint written before this state was tracked).
    pub fn optimizer_at(&self, index: usize) -> Result<Option<OptimizerState>> {
        self.optional_file(index, "optimizer", "safetensors")?
            .map(optimizer_io::load)
            .transpose()
    }

    /// Loads the scheduler state at position `index`, or `Ok(None)` if the
    /// checkpoint has no `scheduler.json` (stateless scheduler, or a checkpoint
    /// written before this state was tracked).
    pub fn scheduler_at(&self, index: usize) -> Result<Option<SchedulerState>> {
        self.optional_file(index, "scheduler", "json")?
            .map(scheduler_io::load)
            .transpose()
    }

    /// Reads all `evaluations.json` files into a pure [`EvaluationHistory`].
    pub fn evaluation_history(&self) -> Result<EvaluationHistory> {
        let mut history = Vec::with_capacity(self.entries.len());

        for entry in &self.entries {
            let evals: CheckpointEvaluationSet = json::load(entry.dir.join("evaluations"))?;
            history.push(EpochEvaluation {
                epoch: entry.epoch,
                evaluation: evals.into(),
            });
        }

        Ok(EvaluationHistory::new(history))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::RELU;
    use crate::io::hyperparams::HyperParametersRecord;
    use crate::io::run::{TrainingMeta, TrainingRun};
    use crate::model::NeuronLayerSpec;
    use crate::optimizers::{Adam, StochasticGradientDescent};
    use crate::schedulers::{ConstantScheduler, Scheduler, StepDecay};
    use ndarray::Array2;

    fn sample_optimizer() -> Adam {
        Adam::with_defaults(0.01.try_into().unwrap())
    }

    fn sample_scheduler() -> ConstantScheduler {
        ConstantScheduler::new(0.01.try_into().unwrap())
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = PathBuf::from(format!(
            "target/nrn_checkpoint_{}_{}",
            tag,
            std::process::id()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn cleanup(dir: &Path) {
        let _ = fs::remove_dir_all(dir);
    }

    fn create_run<P: AsRef<Path>>(path: P, dataset: &str, overwrite: bool) -> CheckpointRecorder {
        TrainingRun::create(
            path,
            &TrainingMeta {
                dataset: dataset.to_string(),
                model: format!("model-{dataset}"),
                hyperparams: HyperParametersRecord::sample(),
            },
            overwrite,
        )
        .unwrap()
        .recorder()
    }

    fn sample_model() -> NeuralNetwork {
        let specs = NeuronLayerSpec::network_for(vec![3], &*RELU, 2);
        NeuralNetwork::initialization(2, &specs, 0)
    }

    fn make_eval(loss: f32, with_validation: bool) -> EvaluationSet {
        EvaluationSet {
            train: Evaluation {
                loss,
                accuracy: 0.5,
            },
            validation: with_validation.then_some(Evaluation {
                loss: loss + 1.0,
                accuracy: 0.1,
            }),
            test: Evaluation {
                loss: loss + 100.0,
                accuracy: 0.9,
            },
        }
    }

    #[test]
    fn write_names_checkpoint_dir_by_epoch() {
        let dir = temp_dir("write_epoch");
        let recorder = create_run(&dir, "ds", false);
        recorder
            .write(
                &sample_model(),
                &sample_optimizer(),
                &sample_scheduler(),
                &make_eval(0.0, false),
                37,
            )
            .unwrap();

        let exists = dir.join("checkpoint-000037").is_dir();
        cleanup(&dir);

        assert!(exists);
    }

    #[test]
    fn roundtrip_with_validation() {
        let dir = temp_dir("roundtrip_val");
        let recorder = create_run(&dir, "ds", false);
        for i in 0..3 {
            recorder
                .write(
                    &sample_model(),
                    &sample_optimizer(),
                    &sample_scheduler(),
                    &make_eval(i as f32, true),
                    i * 10,
                )
                .unwrap();
        }

        let history = CheckpointArchive::load(&dir)
            .unwrap()
            .evaluation_history()
            .unwrap();
        cleanup(&dir);

        assert_eq!(history.len(), 3);
        for (i, loss) in history.train_losses().iter().enumerate() {
            assert_eq!(*loss, i as f32);
        }
        assert!(!history.validation_losses().is_empty());
    }

    #[test]
    fn roundtrip_without_validation() {
        let dir = temp_dir("roundtrip_noval");
        let recorder = create_run(&dir, "ds", false);
        for i in 0..3 {
            recorder
                .write(
                    &sample_model(),
                    &sample_optimizer(),
                    &sample_scheduler(),
                    &make_eval(i as f32, false),
                    i * 10,
                )
                .unwrap();
        }

        let history = CheckpointArchive::load(&dir)
            .unwrap()
            .evaluation_history()
            .unwrap();
        cleanup(&dir);

        assert_eq!(history.len(), 3);
        assert!(history.validation_losses().is_empty());
    }

    #[test]
    fn predictions_survive_roundtrip() {
        let dir = temp_dir("predictions");
        let model = sample_model();
        let inputs = Array2::from_shape_fn((2, 4), |(i, j)| (i + j) as f32 * 0.3);

        let recorder = create_run(&dir, "ds", false);
        recorder
            .write(
                &model,
                &sample_optimizer(),
                &sample_scheduler(),
                &make_eval(0.0, false),
                0,
            )
            .unwrap();

        let archive = CheckpointArchive::load(&dir).unwrap();
        let loaded = archive.model_at(0).unwrap();
        cleanup(&dir);

        assert_eq!(model.predict(inputs.view()), loaded.predict(inputs.view()));
    }

    #[test]
    fn numeric_sort_beats_lexical() {
        let dir = temp_dir("sort");
        let model = sample_model();
        let recorder = create_run(&dir, "ds", false);
        for i in 0..12 {
            recorder
                .write(
                    &model,
                    &sample_optimizer(),
                    &sample_scheduler(),
                    &make_eval(i as f32, false),
                    i,
                )
                .unwrap();
        }

        let archive = CheckpointArchive::load(&dir).unwrap();
        cleanup(&dir);

        assert_eq!(archive.len(), 12);
        for i in 0..12 {
            assert_eq!(archive.epoch_at(i), Some(i));
        }
    }

    #[test]
    fn empty_dir_archive_is_empty() {
        let dir = temp_dir("empty");
        let archive = CheckpointArchive::load(&dir).unwrap();
        cleanup(&dir);

        assert!(archive.is_empty());
        assert_eq!(archive.len(), 0);
    }

    #[test]
    fn load_ignores_non_directory_checkpoint_file() {
        let dir = temp_dir("non_dir_snap");
        fs::write(dir.join("checkpoint-000000"), b"not a dir").unwrap();
        let archive = CheckpointArchive::load(&dir).unwrap();
        cleanup(&dir);

        assert!(archive.is_empty());
    }

    #[test]
    fn load_ignores_checkpoint_dir_with_non_numeric_suffix() {
        let dir = temp_dir("non_numeric_snap");
        fs::create_dir_all(dir.join("checkpoint-abc")).unwrap();
        let archive = CheckpointArchive::load(&dir).unwrap();
        cleanup(&dir);

        assert!(archive.is_empty());
    }

    #[test]
    fn load_ignores_unrelated_directory() {
        // A directory whose name lacks the `checkpoint-` prefix is skipped.
        let dir = temp_dir("unrelated_dir");
        fs::create_dir_all(dir.join("scratch")).unwrap();
        let archive = CheckpointArchive::load(&dir).unwrap();
        cleanup(&dir);

        assert!(archive.is_empty());
    }

    #[test]
    fn model_at_out_of_range_gives_range_error() {
        let dir = temp_dir("model_at_oob");
        let recorder = create_run(&dir, "ds", false);
        recorder
            .write(
                &sample_model(),
                &sample_optimizer(),
                &sample_scheduler(),
                &make_eval(0.0, false),
                0,
            )
            .unwrap();

        let archive = CheckpointArchive::load(&dir).unwrap();
        let msg = archive.model_at(99).unwrap_err().to_string();
        cleanup(&dir);

        assert!(msg.contains("out of range"), "got: {msg}");
    }

    #[test]
    fn optimizer_and_scheduler_at_out_of_range_error() {
        // `optimizer_at` / `scheduler_at` reach `entry_at` through `optional_file`,
        // a different call site than `model_at`, so its range check is covered here.
        let dir = temp_dir("state_at_oob");
        let recorder = create_run(&dir, "ds", false);
        recorder
            .write(
                &sample_model(),
                &sample_optimizer(),
                &sample_scheduler(),
                &make_eval(0.0, false),
                0,
            )
            .unwrap();

        let archive = CheckpointArchive::load(&dir).unwrap();
        let opt_err = archive.optimizer_at(99).unwrap_err().to_string();
        let sched_err = archive.scheduler_at(99).unwrap_err().to_string();
        cleanup(&dir);

        assert!(opt_err.contains("out of range"), "got: {opt_err}");
        assert!(sched_err.contains("out of range"), "got: {sched_err}");
    }

    #[test]
    fn model_at_missing_model_file_fails() {
        let dir = temp_dir("model_at_missing");
        let recorder = create_run(&dir, "ds", false);
        recorder
            .write(
                &sample_model(),
                &sample_optimizer(),
                &sample_scheduler(),
                &make_eval(0.0, false),
                0,
            )
            .unwrap();
        fs::remove_file(dir.join("checkpoint-000000").join("model.safetensors")).unwrap();

        let archive = CheckpointArchive::load(&dir).unwrap();
        let result = archive.model_at(0);
        cleanup(&dir);

        assert!(result.is_err());
    }

    #[test]
    fn model_at_corrupt_model_file_fails() {
        let dir = temp_dir("model_at_corrupt");
        let recorder = create_run(&dir, "ds", false);
        recorder
            .write(
                &sample_model(),
                &sample_optimizer(),
                &sample_scheduler(),
                &make_eval(0.0, false),
                0,
            )
            .unwrap();
        fs::write(
            dir.join("checkpoint-000000").join("model.safetensors"),
            b"garbage",
        )
        .unwrap();

        let archive = CheckpointArchive::load(&dir).unwrap();
        let result = archive.model_at(0);
        cleanup(&dir);

        assert!(result.is_err());
    }

    #[test]
    fn evaluation_history_corrupted_evaluations_json_fails() {
        let dir = temp_dir("corrupt_evals");
        let recorder = create_run(&dir, "ds", false);
        recorder
            .write(
                &sample_model(),
                &sample_optimizer(),
                &sample_scheduler(),
                &make_eval(0.0, false),
                0,
            )
            .unwrap();
        fs::write(
            dir.join("checkpoint-000000").join("evaluations.json"),
            b"not valid json",
        )
        .unwrap();

        let archive = CheckpointArchive::load(&dir).unwrap();
        let result = archive.evaluation_history();
        cleanup(&dir);

        assert!(result.is_err());
    }

    #[test]
    fn on_checkpoint_writes_a_checkpoint() {
        let dir = temp_dir("on_checkpoint");
        let mut recorder = create_run(&dir, "ds", false);

        recorder
            .on_checkpoint(
                &sample_model(),
                &sample_optimizer(),
                &sample_scheduler(),
                &make_eval(0.0, false),
                5,
            )
            .unwrap();

        let archive = CheckpointArchive::load(&dir).unwrap();
        cleanup(&dir);

        assert_eq!(archive.len(), 1);
        assert_eq!(archive.epoch_at(0), Some(5));
    }

    #[test]
    fn write_with_adam_writes_optimizer_state() {
        let dir = temp_dir("optimizer_adam");
        let recorder = create_run(&dir, "ds", false);

        let mut optimizer = sample_optimizer();
        let mut trained_model = sample_model();
        let gradients = crate::gradients::Gradients {
            dw: Array2::from_elem(trained_model.layers[0].weights.dim(), 0.1),
            db: ndarray::Array1::from_elem(trained_model.layers[0].biases.dim(), 0.1),
        };
        optimizer.update(0, &mut trained_model.layers[0], &gradients);
        optimizer.step();

        recorder
            .write(
                &trained_model,
                &optimizer,
                &sample_scheduler(),
                &make_eval(0.0, false),
                0,
            )
            .unwrap();

        let exists = dir.join("checkpoint-000000/optimizer.safetensors").exists();
        let archive = CheckpointArchive::load(&dir).unwrap();
        let state = archive.optimizer_at(0).unwrap();
        cleanup(&dir);

        assert!(exists);
        let state = state.unwrap();
        assert!(!state.tensors.is_empty());
        assert_eq!(
            state.metadata.get("time_step").map(String::as_str),
            Some("2")
        );
    }

    #[test]
    fn write_with_sgd_writes_no_optimizer_file() {
        let dir = temp_dir("optimizer_sgd");
        let recorder = create_run(&dir, "ds", false);

        let optimizer = StochasticGradientDescent::new(0.01.try_into().unwrap());
        recorder
            .write(
                &sample_model(),
                &optimizer,
                &sample_scheduler(),
                &make_eval(0.0, false),
                0,
            )
            .unwrap();

        let exists = dir.join("checkpoint-000000/optimizer.safetensors").exists();
        let archive = CheckpointArchive::load(&dir).unwrap();
        let state = archive.optimizer_at(0).unwrap();
        cleanup(&dir);

        assert!(!exists);
        assert!(state.is_none());
    }

    #[test]
    fn write_with_stateful_scheduler_writes_scheduler_state() {
        let dir = temp_dir("scheduler_step");
        let recorder = create_run(&dir, "ds", false);

        let mut scheduler = StepDecay::from_values(0.1, 2, 0.5).unwrap();
        scheduler.step();
        scheduler.step();
        scheduler.step();

        recorder
            .write(
                &sample_model(),
                &sample_optimizer(),
                &scheduler,
                &make_eval(0.0, false),
                0,
            )
            .unwrap();

        let exists = dir.join("checkpoint-000000/scheduler.json").exists();
        let archive = CheckpointArchive::load(&dir).unwrap();
        let state = archive.scheduler_at(0).unwrap();
        cleanup(&dir);

        assert!(exists);
        assert_eq!(state.unwrap().current_step, 3);
    }

    #[test]
    fn write_with_stateless_scheduler_writes_no_scheduler_file() {
        let dir = temp_dir("scheduler_constant");
        let recorder = create_run(&dir, "ds", false);

        recorder
            .write(
                &sample_model(),
                &sample_optimizer(),
                &sample_scheduler(),
                &make_eval(0.0, false),
                0,
            )
            .unwrap();

        let exists = dir.join("checkpoint-000000/scheduler.json").exists();
        let archive = CheckpointArchive::load(&dir).unwrap();
        let state = archive.scheduler_at(0).unwrap();
        cleanup(&dir);

        assert!(!exists);
        assert!(state.is_none());
    }

    #[test]
    fn write_fails_when_evaluations_path_is_a_directory() {
        let dir = temp_dir("write_evals_dir_conflict");
        let recorder = create_run(&dir, "ds", false);

        // Pre-create "evaluations.json" as a directory so json::save's fs::write fails.
        fs::create_dir_all(dir.join("checkpoint-000000").join("evaluations.json")).unwrap();

        let result = recorder.write(
            &sample_model(),
            &sample_optimizer(),
            &sample_scheduler(),
            &make_eval(0.0, false),
            0,
        );
        cleanup(&dir);

        assert!(result.is_err());
    }

    #[test]
    fn load_rejects_path_traversal() {
        let result = CheckpointArchive::load("../../nrn_traversal_test");
        assert!(result.is_err());
    }
}
