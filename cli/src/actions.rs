//! Orchestration shared across commands: small bits of glue that resolve CLI
//! arguments into domain values, tracing what happened as they go.

use crate::display::{ReadInstance, loaded};
use nrn::data::Instance;
use std::error::Error;

/// Resolves an instance argument: loads it from `source` when given (tracing it
/// as loaded), otherwise reads `input_size` feature values from stdin.
pub(crate) fn load_or_read_instance(
    source: Option<String>,
    input_size: usize,
) -> Result<Instance, Box<dyn Error>> {
    match source {
        Some(file) => {
            let instance = Instance::load(&file)?;
            loaded(&instance);
            Ok(instance)
        }
        None => Instance::read(input_size),
    }
}
