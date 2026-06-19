use super::{Describe, Named, rows};
use nrn::io::checkpoint::CheckpointArchive;

impl Named for CheckpointArchive {
    const NAME: &'static str = "TRAINING RUN";
}

impl Describe for CheckpointArchive {
    fn describe(&self) -> String {
        let first = self.epoch_at(0).unwrap_or(0);
        let last = self.epoch_at(self.len().saturating_sub(1)).unwrap_or(0);

        rows(&[
            ("Checkpoints", self.len().to_string()),
            ("Epochs", format!("{first}..{last}")),
        ])
    }
}
