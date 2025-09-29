use crate::io::path::PathExt;
use std::io::Result;
use std::path::Path;

/// Reads the entire contents of a file into a byte vector after validating the file path.
///
/// This function performs a safety check on the given `path` to prevent path traversal attacks by ensuring
/// the resolved absolute path stays within an allowed working directory or base path.
///
/// # Arguments
///
/// * `path` - A reference to a path-like object (`impl AsRef<Path>`) representing the file to read.
///
/// # Returns
///
/// * `Ok(Vec<u8>)` containing the file content in bytes if the path is valid and the file can be read.
/// * `Err(std::io::Error)` if the path check fails or if reading the file results in an IO error.
///
/// # Errors
///
/// This function will return an error if:
/// - The path is invalid or deemed unsafe (e.g., attempts path traversal).
/// - The file does not exist or cannot be opened.
/// - There are insufficient permissions to read the file.
///
/// # Examples
///
/// ```
/// use std::path::Path;
///
/// let file_path = Path::new("safe_dir/data.bin");
/// let data = io::bytes::secure_read(file_path)?;
/// println!("Read {} bytes", data.len());
/// # Ok::<(), std::io::Error>(())
/// ```
///
/// # Panics
///
/// This function does not panic.
///
/// # Safety
///
/// This function adds a layer of security by preventing attempts to read files outside the allowed directory.
///
/// # Notes
///
/// The specifics of the path validation (e.g., what base directory is allowed) depend on the implementation of
/// the path-checking logic.
///
///
pub fn secure_read<P: AsRef<Path>>(path: P) -> Result<Vec<u8>> {
    let path = Path::combine_safe_with_cwd(path)?;
    std::fs::read(path)
}
