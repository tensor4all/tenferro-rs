//! Device-free tests of the Metal module's tables and refusals.
//!
//! Anything that needs a device lives in `tests/metal_engine.rs`, which is a `[[test]]`
//! target with `required-features = ["metal"]` so a headless Linux CI job never builds it.

use super::capability::METAL_BACKEND;
use super::metal_capabilities;

#[test]
fn the_capability_table_names_this_backend_and_never_claims_f64() {
    let table = metal_capabilities();
    assert!(!table.is_empty());
    for row in table {
        assert_eq!(row.backend, METAL_BACKEND);
        assert!(
            !matches!(
                row.dtype,
                tenferro_tensor::DType::F64 | tenferro_tensor::DType::C64
            ),
            "{:?} claims a double-precision Metal row; MSL rejects `double`",
            row.op
        );
    }
}

#[test]
fn every_operation_has_one_row_per_dtype() {
    let table = metal_capabilities();
    let mut seen = Vec::new();
    for row in table {
        let key = (row.op, row.dtype, row.output_dtype);
        assert!(!seen.contains(&key), "duplicate capability row for {key:?}");
        seen.push(key);
    }
}

#[test]
fn a_complex_magnitude_row_reports_a_real_output_dtype() {
    let row = metal_capabilities()
        .iter()
        .find(|row| {
            row.op == tenferro_core_ops::PrimitiveOpKind::Abs
                && row.dtype == tenferro_tensor::DType::C32
        })
        .expect("abs of Complex32 has a row");
    assert_eq!(row.output_dtype, tenferro_tensor::DType::F32);
}
