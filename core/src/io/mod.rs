pub mod bytes;
pub mod checkpoint;
pub mod classes;
pub mod data;
pub mod hyperparams;
pub mod instance;
pub mod json;
pub mod model;
pub mod optimizer;
pub mod path;
pub mod predictor;
pub mod run;
pub mod scalers;
pub mod scheduler;
pub mod task;
pub mod tensors;

// Figure persistence — only available alongside the raster renderer.
#[cfg(feature = "raster")]
pub mod gif;
#[cfg(feature = "raster")]
pub mod png;
