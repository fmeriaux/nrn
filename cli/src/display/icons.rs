//! The emoji glyphs prefixing console lines, each with an ASCII fallback for
//! terminals without emoji support.

use console::Emoji;

pub(crate) const LOAD_ICON: Emoji = Emoji("📥", "[L]");
pub(crate) const GEN_ICON: Emoji = Emoji("🌱", "[G]");
pub(crate) const INIT_ICON: Emoji = Emoji("🚀", "[I]");
pub(crate) const MODEL_ICON: Emoji = Emoji("🧠", "[M]");
pub(crate) const SCALER_ICON: Emoji = Emoji("📏", "[S]");
pub(crate) const DATASET_ICON: Emoji = Emoji("📁", "[D]");
pub(crate) const PLOT_ICON: Emoji = Emoji("🧊", "[P]");
pub(crate) const RUN_ICON: Emoji = Emoji("📈", "[T]");
pub(crate) const ANIMATION_ICON: Emoji = Emoji("🎬", "[A]");
pub(crate) const WARN_ICON: Emoji = Emoji("⚠️", "[!]");
pub(crate) const ERROR_ICON: Emoji = Emoji("❌", "[X]");
pub(crate) const SUCCESS_ICON: Emoji = Emoji("✅", "[✓]");
pub(crate) const TRACE_ICON: Emoji = Emoji("🔍", "[*]");
