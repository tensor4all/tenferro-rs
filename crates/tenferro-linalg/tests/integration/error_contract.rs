//! One linalg error type, one rendering: the decomposition names live in the
//! `op` payload, so every failure a decomposition reports is a `linalg::Error`
//! variant and the sentence a caller sees is fixed here.
//!
//! These tests are the snapshot. Changing an `#[error(...)]` attribute or
//! dropping a variant breaks them, which is the point: log lines, test
//! expectations, and downstream `contains` checks read these sentences.

use std::error::Error as _;

use tenferro_linalg::{Error as LinalgError, TensorLinalgExt};
use tenferro_tensor::{DType, ErrorKind};

/// Every decomposition reports through the same four variants, and the
/// operation it came from is the `op` payload rather than the type.
#[test]
fn every_decomposition_renders_through_the_same_variants() {
    let rendered = [
        (
            LinalgError::NonConvergence { op: "svd" },
            "svd did not converge",
        ),
        (
            LinalgError::NonFinite {
                op: "eigh",
                role: "input",
            },
            "eigh encountered non-finite input",
        ),
        (LinalgError::Singular { op: "solve" }, "solve is singular"),
        (
            LinalgError::UnsupportedDType {
                op: "qr",
                dtype: DType::I32,
            },
            "qr does not support dtype I32",
        ),
    ];

    for (error, expected) in rendered {
        assert_eq!(error.to_string(), expected);
    }
}

/// The coarse classification each variant reports is part of the boundary
/// contract, so it is pinned next to the text.
#[test]
fn every_variant_keeps_its_coarse_kind() {
    let classified = [
        (
            LinalgError::NonConvergence { op: "eig" },
            ErrorKind::NumericalFailure,
        ),
        (
            LinalgError::NonFinite {
                op: "eig",
                role: "eigenvalue",
            },
            ErrorKind::NumericalFailure,
        ),
        (
            LinalgError::Singular { op: "cholesky" },
            ErrorKind::NumericalFailure,
        ),
        (
            LinalgError::UnsupportedDType {
                op: "lu",
                dtype: DType::Bool,
            },
            ErrorKind::Unsupported,
        ),
    ];

    for (error, expected) in classified {
        assert_eq!(error.kind(), expected, "{error}");
    }
}

/// Crossing the tensor boundary keeps the linalg value as the typed source
/// instead of flattening it into a message. The wrapper prefixes the operation
/// and the extension family exactly once, and the sentence it carries is the
/// one the linalg variant renders.
#[test]
fn a_failed_factorization_survives_the_tensor_boundary() {
    let mut host = tenferro_cpu::CpuEngine::serial();
    // [[1, 2], [2, 1]] has eigenvalues 3 and -1, so it is not positive definite.
    let input =
        tenferro_tensor::Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 2.0, 1.0])
            .expect("2x2 column-major payload");

    let error = input
        .cholesky(&mut host)
        .expect_err("a non positive-definite matrix has no Cholesky factor");

    assert_eq!(error.kind(), ErrorKind::NumericalFailure, "{error}");

    let source = error
        .source()
        .and_then(|source| source.downcast_ref::<LinalgError>())
        .expect("the linalg failure stays the typed source across the boundary");
    assert!(
        matches!(source, LinalgError::NonConvergence { op: "cholesky" }),
        "{source}"
    );
    assert_eq!(
        error.to_string(),
        "cholesky: extension tenferro-linalg.linalg.v1 failed: cholesky did not converge"
    );
    assert!(error.to_string().ends_with(&source.to_string()), "{error}");
}
