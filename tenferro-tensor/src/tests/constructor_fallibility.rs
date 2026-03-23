use std::fs;
use std::path::PathBuf;

use tenferro_device::{Error, LogicalMemorySpace};

use super::*;

const CPU: LogicalMemorySpace = LogicalMemorySpace::MainMemory;

#[cfg(feature = "cuda")]
const GPU0: LogicalMemorySpace = LogicalMemorySpace::GpuMemory { device_id: 0 };

fn constructors_source(relative: &str) -> String {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push(relative);
    fs::read_to_string(path).unwrap_or_else(|err| {
        panic!("failed to read {relative} for source-level regression: {err}")
    })
}

#[test]
fn cpu_empty_strided_rejects_invalid_layouts_without_panicking() {
    let err = Tensor::<f64>::empty_strided(&[2, 2], &[1, 2], -1, CPU)
        .expect_err("empty_strided should return an Error, not panic");

    assert!(
        matches!(err, Error::StrideError(_) | Error::InvalidArgument(_)),
        "expected invalid empty_strided layout to return an error, got {err:?}"
    );
}

#[test]
fn cpu_arange_rejects_invalid_inputs_without_panicking() {
    let err = Tensor::<f64>::arange(0.0, 5.0, 0.0, CPU, MemoryOrder::ColumnMajor)
        .expect_err("arange should return an Error, not panic");

    assert!(
        matches!(err, Error::InvalidArgument(_)),
        "expected invalid arange input to return an error, got {err:?}"
    );
}

#[test]
fn cpu_linspace_rejects_invalid_inputs_without_panicking() {
    let err = Tensor::<f64>::linspace(0.0, 1.0, -3, CPU, MemoryOrder::ColumnMajor)
        .expect_err("linspace should return an Error, not panic");

    assert!(
        matches!(err, Error::InvalidArgument(_)),
        "expected invalid linspace input to return an error, got {err:?}"
    );
}

#[test]
fn cpu_eye_rejects_overflow_without_panicking() {
    let err = Tensor::<f64>::eye(usize::MAX, CPU, MemoryOrder::ColumnMajor)
        .expect_err("eye should return an Error on overflow, not panic");

    assert!(
        matches!(err, Error::StrideError(_)),
        "expected eye overflow to return a stride error, got {err:?}"
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_empty_strided_rejects_invalid_layouts_without_panicking() {
    let err = Tensor::<f64>::empty_strided(&[2, 2], &[1, 2], -1, GPU0)
        .expect_err("empty_strided should return an Error, not panic");

    assert!(
        matches!(err, Error::StrideError(_) | Error::InvalidArgument(_)),
        "expected invalid empty_strided layout to return an error, got {err:?}"
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_arange_rejects_invalid_inputs_without_panicking() {
    let err = Tensor::<f64>::arange(0.0, 5.0, 0.0, GPU0, MemoryOrder::ColumnMajor)
        .expect_err("arange should return an Error, not panic");

    assert!(
        matches!(err, Error::InvalidArgument(_)),
        "expected invalid arange input to return an error, got {err:?}"
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_linspace_rejects_invalid_inputs_without_panicking() {
    let err = Tensor::<f64>::linspace(0.0, 1.0, -3, GPU0, MemoryOrder::ColumnMajor)
        .expect_err("linspace should return an Error, not panic");

    assert!(
        matches!(err, Error::InvalidArgument(_)),
        "expected invalid linspace input to return an error, got {err:?}"
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_eye_rejects_overflow_without_panicking() {
    let err = Tensor::<f64>::eye(usize::MAX, GPU0, MemoryOrder::ColumnMajor)
        .expect_err("eye should return an Error on overflow, not panic");

    assert!(
        matches!(err, Error::StrideError(_)),
        "expected eye overflow to return a stride error, got {err:?}"
    );
}

#[test]
fn constructors_source_no_longer_contains_panic_based_public_paths() {
    for relative in [
        "src/tensor/constructors.rs",
        "src/tensor/constructors_special.rs",
    ] {
        let source = constructors_source(relative);
        let forbidden = [
            "unwrap_or_else(|err| panic!",
            "panic!(\"tensor allocation",
            "panic!(\"eye:",
        ];

        for needle in forbidden {
            assert!(
                !source.contains(needle),
                "{relative} still contains forbidden panic path: {needle}"
            );
        }
    }
}
