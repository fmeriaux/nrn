//! The emoji glyphs prefixing console lines, each with an ASCII fallback for
//! terminals without emoji support.

use console::Emoji;

pub(crate) const LOAD_ICON: Emoji = Emoji("📥", "[L]");
pub(crate) const GEN_ICON: Emoji = Emoji("🌱", "[G]");
pub(crate) const INIT_ICON: Emoji = Emoji("🚀", "[I]");
pub(crate) const SAVE_ICON: Emoji = Emoji("💾", "[w]");
pub(crate) const RECORD_ICON: Emoji = Emoji("⏺️", "[R]");
pub(crate) const WARN_ICON: Emoji = Emoji("⚠️", "[!]");
pub(crate) const ERROR_ICON: Emoji = Emoji("❌", "[X]");
pub(crate) const SUCCESS_ICON: Emoji = Emoji("✅", "[✓]");
pub(crate) const TRACE_ICON: Emoji = Emoji("🔍", "[*]");
pub(crate) const EVAL_ICON: Emoji = Emoji("📊", "[=]");
