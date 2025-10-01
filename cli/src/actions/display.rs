use std::env;
use std::path::{Path, PathBuf};
use console::{style, Emoji};
use pathdiff::diff_paths;

const SAVE_ICON: Emoji = Emoji("✔", "[SAVE]");

fn to_relative_path<P: AsRef<Path>>(path: P) -> PathBuf {
    env::current_dir()
        .ok()
        .and_then(|cwd| diff_paths(&path, cwd))
        .unwrap_or_else(|| path.as_ref().to_path_buf())
}

pub fn saved_at<P: AsRef<Path>>(path: P, file_type: &str) {
    let relative_path = to_relative_path(&path);
    println!(
        "{} {} saved at {}",
        style(SAVE_ICON).bright().green(),
        file_type,
        style(relative_path.display()).bright().magenta().italic()
    );
}