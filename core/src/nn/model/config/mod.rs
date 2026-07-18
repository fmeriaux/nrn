//! Layer and network configuration: the declarative description of a network's architecture.
//! [`NetworkConfig`] bundles the per-sample input shape with an ordered stack of
//! [`LayerConfig`]s, one per layer kind.

mod layer;
mod network;

pub use layer::*;
pub use network::*;
