//! Visualization: a backend-neutral [`Figure`] model and its renderers.
//!
//! [`scene`] defines the figure model and [`build`] derives figures from domain objects;
//! both are always compiled. Renderers live behind feature flags — [`image`] rasterizes
//! to RGB with `plotters`.

mod build;
mod scene;

pub use scene::*;

#[cfg(feature = "raster")]
mod image;
#[cfg(feature = "raster")]
pub use image::*;
