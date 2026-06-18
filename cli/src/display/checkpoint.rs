use super::{Describe, block};
use nrn::io::checkpoint::CheckpointArchive;

impl Describe for CheckpointArchive {
    fn describe(&self) -> String {
        let first = self.epoch_at(0).unwrap_or(0);
        let last = self.epoch_at(self.len().saturating_sub(1)).unwrap_or(0);

        block(
            "TRAINING RUN",
            &[
                ("Checkpoints", self.len().to_string()),
                ("Epochs", format!("{first}..{last}")),
            ],
        )
    }
}
