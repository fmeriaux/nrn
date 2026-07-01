//! Visualization: a backend-neutral [`Figure`] model and its renderers.
//!
//! [`scene`] defines the figure model and [`build`] derives figures from domain objects;
//! [`activations`] models a forward pass as a neuron-and-connection diagram. All are
//! always compiled. Renderers live behind feature flags — [`image`] rasterizes figures and
//! diagrams to RGB with `plotters`, [`console`] draws them as text. The activation diagram
//! renders vertically (nodes only) to the console and as a horizontal node-link graph to an image.

mod activations;
mod build;
mod scene;

pub use activations::*;
pub use scene::*;

#[cfg(feature = "raster")]
mod image;
#[cfg(feature = "raster")]
pub use image::*;

#[cfg(feature = "console")]
mod console;
#[cfg(feature = "console")]
pub use console::*;
