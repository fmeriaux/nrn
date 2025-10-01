use crate::io::path::PathExt;
use gif::{Encoder, Frame, Repeat};
use std::error::Error;
use std::fs::File;
use std::path::{Path, PathBuf};

pub fn save_gif<P: AsRef<Path>>(
    frames: Vec<Frame>,
    width: u16,
    height: u16,
    path: P,
) -> Result<PathBuf, Box<dyn Error>> {
    let filepath = path.as_ref().with_extension("gif");
    let filepath = Path::combine_safe_with_cwd(filepath)?;
    filepath.create_parents()?;

    let file = File::create(&filepath)?;

    let mut encoder = Encoder::new(file, width, height, &[])?;

    encoder.set_repeat(Repeat::Infinite)?;

    for frame in frames.iter() {
        encoder.write_frame(&frame)?;
    }

    Ok(filepath.to_path_buf())
}

pub fn save_gif_from_rgb<P: AsRef<Path>>(
    rgb_frames: Vec<Vec<u8>>,
    width: u16,
    height: u16,
    frame_delay: u16,
    path: P,
) -> Result<PathBuf, Box<dyn Error>> {
    let frames = rgb_frames
        .into_iter()
        .map(|rgb_frame| {
            let mut frame = Frame::from_rgb(width, height, &rgb_frame);
            frame.delay = frame_delay;
            frame
        })
        .collect();

    save_gif(frames, width, height, path)
}
