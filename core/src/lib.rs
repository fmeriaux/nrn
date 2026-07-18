mod nn;
pub use nn::*;

pub mod analysis;
pub mod data;
pub mod plot;

#[cfg(feature = "io")]
pub mod io;

// Links the selected BLAS backend; ndarray's `dot` then resolves to its `sgemm`.
#[cfg(feature = "blas")]
use blas_src as _;

#[cfg(test)]
pub(crate) mod testing;
