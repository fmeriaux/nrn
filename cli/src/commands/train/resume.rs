use super::DivergedRun;
use super::args::ResumeOverrides;
use super::callbacks::{ConsoleMonitor, ModelSaver};
use crate::display::{Spinner, loaded, recording, show, warning};
use clap::Args;
use nrn::data::Dataset;
use nrn::io::run::TrainingRun;
use nrn::objectives::Objective;
use nrn::training::{Callbacks, HyperParameters};
use std::error::Error;
use std::path::Path;

#[derive(Args, Debug)]
pub struct ResumeArgs {
    /// Training run directory to resume from
    run_dir: String,

    /// Epoch to resume from; must match a recorded checkpoint (default: last checkpoint)
    #[arg(long)]
    from: Option<usize>,

    #[command(flatten)]
    overrides: ResumeOverrides,
}

impl ResumeArgs {
    pub fn run(&self) -> Result<(), Box<dyn Error>> {
        let run_dir = Path::new(&self.run_dir);
        let run = TrainingRun::open(run_dir)?;
        let meta = run.meta();

        let dataset = Dataset::load(&meta.dataset)?;
        loaded(&dataset);

        // Re-inferred from the dataset in this increment; will be read from the persisted run
        // metadata once the objective is part of the run config.
        let objective = Objective::from_dataset(&dataset);
        show(&objective);

        let previous = HyperParameters::try_from(meta.hyperparams.clone())?;
        let scaler = meta.scaler.clone().map(Into::into);

        let mut record = meta.hyperparams.clone();
        self.overrides.apply(&mut record);
        let hyperparameters = HyperParameters::try_from(record)?;

        let archive = run.archive()?;
        let checkpoint_idx = archive.resolve_epoch(self.from)?;

        let model = archive.model_at(checkpoint_idx)?;
        loaded(&model);

        let from_epoch = archive
            .epoch_at(checkpoint_idx)
            .expect("checkpoint_idx was just resolved against the archive");

        let scheduler_state = archive.scheduler_at(checkpoint_idx)?;
        let optimizer_state = archive.optimizer_at(checkpoint_idx)?;

        let recorder = if hyperparameters.checkpoint_interval() > 0 {
            let trimmed = run.trim_after(from_epoch)?;
            if trimmed > 0 {
                warning!("Removed {trimmed} checkpoint(s) after epoch {from_epoch}");
            }
            let recorder = run.recorder();
            recording(&recorder);
            Some(recorder)
        } else {
            None
        };

        let spinner = Spinner::start("Preparing dataset");
        let data = hyperparameters.prepare(dataset, scaler)?;
        spinner.finish();

        let callbacks = Callbacks::empty()
            .with(ConsoleMonitor::new(hyperparameters.clone(), Some(previous)))
            .with(ModelSaver::new(
                run_dir,
                &meta.model,
                data.scaler().cloned(),
            ))
            .with_opt(recorder);

        let mut trainer = hyperparameters.build(model, objective, data, callbacks)?;
        trainer.restore(from_epoch, optimizer_state, scheduler_state)?;
        trainer.train()?.into_result().map_err(DivergedRun::from)?;

        Ok(())
    }
}
