//! End-to-end coverage of the `encode` command: a directory of per-class image
//! folders into a dataset (`encode dataset` / `ds`), and a single image into an
//! instance (`encode instance` / `inst`), plus default output naming, the
//! non-image skip, and the empty-input guard.
//!
//! Fixtures are tiny PNGs written into a temp dir created *under* the crate
//! directory, so both this process and the `nrn` subprocess resolve them within
//! the path-safety boundary (paths outside the cwd are rejected).

use assert_cmd::Command;
use image::{Rgb, RgbImage};
use nrn::data::{Dataset, Instance, Targets};
use predicates::str::contains;
use std::fs;
use std::path::Path;
use tempfile::TempDir;

/// A temp dir under the crate directory (within the path-safety boundary).
fn workspace() -> TempDir {
    TempDir::new_in(".").unwrap()
}

/// Writes a 2x2 solid-color PNG at `path` (the encoder resizes, so size is moot).
fn write_png(path: impl AsRef<Path>, color: [u8; 3]) {
    let mut img = RgbImage::new(2, 2);
    for pixel in img.pixels_mut() {
        *pixel = Rgb(color);
    }
    img.save(path).unwrap();
}

/// A `root/<class>/...` tree of solid-color PNGs, one entry per `(class, file)`.
fn class_dirs(root: &Path, classes: &[(&str, &[&str])]) {
    for (class, files) in classes {
        let dir = root.join(class);
        fs::create_dir_all(&dir).unwrap();
        for (i, file) in files.iter().enumerate() {
            write_png(dir.join(file), [10 * i as u8, 0, 0]);
        }
    }
}

/// A fresh `nrn` invocation rooted at `dir`.
fn nrn(dir: &Path) -> Command {
    let mut cmd = Command::cargo_bin("nrn").unwrap();
    cmd.current_dir(dir);
    cmd
}

#[test]
fn encodes_a_directory_into_a_dataset() {
    let tmp = workspace();
    class_dirs(
        &tmp.path().join("imgs"),
        &[("cat", &["0.png", "1.png"]), ("dog", &["0.png"])],
    );

    nrn(tmp.path())
        .args(["encode", "dataset", "imgs", "-o", "out", "--shape", "4"])
        .assert()
        .success()
        .stdout(contains("ENCODED"));

    // RGB 4x4 = 48 features, 3 samples, 2 classes.
    let dataset = Dataset::load(tmp.path().join("out")).unwrap();
    assert_eq!(dataset.feature_size(), 48);
    assert_eq!(dataset.sample_size(), 3);
    let Targets::ClassLabel(label) = dataset.targets() else {
        panic!("expected ClassLabel targets");
    };
    assert_eq!(label.class_count(), 2);
}

#[test]
fn skips_non_image_files_when_encoding_a_dataset() {
    let tmp = workspace();
    let root = tmp.path().join("imgs");
    class_dirs(&root, &[("cat", &["0.png"]), ("dog", &["0.png"])]);
    // A stray non-image file in a class folder fails to encode and is skipped.
    fs::write(root.join("cat").join("notes.txt"), b"not an image").unwrap();

    nrn(tmp.path())
        .args([
            "encode",
            "dataset",
            "imgs",
            "-o",
            "out",
            "--shape",
            "2",
            "--grayscale",
        ])
        .assert()
        .success()
        .stderr(contains(
            "1 image(s) in 'cat' failed to encode and were skipped",
        ));

    // Grayscale 2x2 = 4 features, only the 2 PNGs counted (the .txt is dropped).
    let dataset = Dataset::load(tmp.path().join("out")).unwrap();
    assert_eq!(dataset.feature_size(), 4);
    assert_eq!(dataset.sample_size(), 2);
}

#[test]
fn ds_alias_defaults_the_output_name_to_the_dataset_id() {
    let tmp = workspace();
    class_dirs(
        &tmp.path().join("imgs"),
        &[("cat", &["0.png"]), ("dog", &["0.png"])],
    );

    nrn(tmp.path())
        .args(["encode", "ds", "imgs", "--shape", "2", "--grayscale"])
        .assert()
        .success();

    // No `-o`: the dataset is saved under its default name.
    assert!(tmp.path().join("imgs-c2-f4-n2.parquet").exists());
}

#[test]
fn errors_when_a_class_folder_has_no_images() {
    let tmp = workspace();
    class_dirs(&tmp.path().join("imgs"), &[("cat", &[]), ("dog", &[])]);

    nrn(tmp.path())
        .args(["encode", "dataset", "imgs"])
        .assert()
        .failure();
}

#[test]
fn warns_before_failing_when_every_image_in_a_class_fails_to_encode() {
    let tmp = workspace();
    let root = tmp.path().join("imgs");
    class_dirs(&root, &[("cat", &["0.png"]), ("dog", &["0.png"])]);
    // Every "image" in `cat` fails to decode, so its label never appears —
    // the warning should still explain why before the contiguity error hits.
    fs::remove_file(root.join("cat").join("0.png")).unwrap();
    fs::write(root.join("cat").join("0.png"), b"not an image").unwrap();

    nrn(tmp.path())
        .args(["encode", "dataset", "imgs"])
        .assert()
        .failure()
        .stderr(contains(
            "1 image(s) in 'cat' failed to encode and were skipped",
        ));
}

#[test]
fn encodes_a_single_image_into_an_instance() {
    let tmp = workspace();
    write_png(tmp.path().join("digit.png"), [128, 64, 32]);

    nrn(tmp.path())
        .args(["encode", "instance", "digit.png", "--shape", "4"])
        .assert()
        .success()
        .stdout(contains("Encoding completed"));

    // No `-o`: the instance is saved under the image's file stem.
    let instance = Instance::load(tmp.path().join("digit")).unwrap();
    assert_eq!(instance.len(), 48); // RGB 4x4
}

#[test]
fn inst_alias_honours_the_output_override() {
    let tmp = workspace();
    write_png(tmp.path().join("digit.png"), [128, 64, 32]);

    nrn(tmp.path())
        .args([
            "encode",
            "inst",
            "digit.png",
            "-o",
            "vec",
            "--shape",
            "2",
            "--grayscale",
        ])
        .assert()
        .success();

    let instance = Instance::load(tmp.path().join("vec")).unwrap();
    assert_eq!(instance.len(), 4); // grayscale 2x2
}
