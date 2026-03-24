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

fn constructor_body<'a>(source: &'a str, name: &str) -> &'a str {
    let signature = format!("pub fn {name}");
    let start = source
        .find(&signature)
        .unwrap_or_else(|| panic!("missing public constructor {name}"));
    let tail = &source[start..];
    let next = tail
        .find("\n    ///")
        .or_else(|| tail.find("\nimpl<"))
        .or_else(|| tail.find("\nfn "))
        .unwrap_or(tail.len());
    &tail[..next]
}

fn assert_constructor_body_is_fallible(source: &str, name: &str) {
    let body = constructor_body(source, name);
    for needle in ["panic!(", "unwrap_or_else(|err| panic!"] {
        assert!(
            !body.contains(needle),
            "public constructor {name} still contains forbidden panic path: {needle}"
        );
    }
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
    let regular = constructors_source("src/tensor/constructors/deterministic.rs");
    for name in [
        "empty",
        "zeros",
        "ones",
        "full",
        "empty_like",
        "zeros_like",
        "ones_like",
        "full_like",
    ] {
        assert_constructor_body_is_fallible(&regular, name);
    }

    let rng = constructors_source("src/tensor/constructors/rng.rs");
    for name in [
        "rand",
        "randn",
        "rand_like",
        "randn_like",
        "randint",
        "randint_like",
    ] {
        assert_constructor_body_is_fallible(&rng, name);
    }

    let special = constructors_source("src/tensor/constructors_special.rs");
    for name in ["eye", "arange", "linspace"] {
        assert_constructor_body_is_fallible(&special, name);
    }
}
