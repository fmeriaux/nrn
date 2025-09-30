mod scale;
mod synth;
mod encode;
mod train;
mod predict;
mod plot;

pub use encode::EncodeArgs;
pub use scale::ScaleArgs;
pub use synth::SynthArgs;
pub use train::TrainArgs;
pub use predict::PredictArgs;
pub use plot::PlotArgs;