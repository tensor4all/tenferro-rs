//! Every `cpu_capabilities()` entry against what `CpuEngine` does (issue #2477).
//!
//! The descriptor is a declaration, not a gate; this test is what makes the declaration
//! trustworthy. Operations come from the table, so a new entry is covered without editing the
//! test, and `capability_cases::run_case` panics for an operation it has no fixture for.

use tenferro_core_ops::PrimitiveOpKind;
use tenferro_cpu::{cpu_capabilities, CpuEngine};
use tenferro_tensor::{
    BackendId, CapabilityAxis, CapabilityQuery, DType, SupportLevel, Tensor,
    TensorBackendCapability,
};

#[path = "../support/capability_cases.rs"]
mod capability_cases;

use capability_cases::{assert_entry_matches, catalog_admits, run_case, DTYPES};

/// The table's operations in table order, each once.
fn table_ops() -> Vec<PrimitiveOpKind> {
    let mut ops: Vec<PrimitiveOpKind> = Vec::new();
    for entry in cpu_capabilities() {
        if !ops.contains(&entry.op) {
            ops.push(entry.op);
        }
    }
    ops
}

#[test]
fn every_descriptor_entry_matches_what_the_cpu_engine_does() {
    let mut engine = CpuEngine::serial();
    let mut place =
        |_engine: &mut CpuEngine, tensor: &Tensor| tensor.duplicate().expect("host copy");
    let mut checked = 0_usize;

    for op in table_ops() {
        for dtype in DTYPES {
            let entry = engine.capability(CapabilityQuery::new(op, dtype));
            if !catalog_admits(op, dtype) {
                assert!(
                    entry.is_none(),
                    "CPU {op:?}/{dtype:?} is outside the catalog's dtype policy but the \
                     descriptor names it"
                );
                continue;
            }
            let outcome = run_case(&mut engine, op, dtype, &mut place);
            assert_entry_matches(BackendId::Cpu, op, dtype, entry, outcome);
            checked += 1;
        }
    }

    assert!(
        checked >= cpu_capabilities().len(),
        "checked {checked} operation/dtype pairs against {} descriptor entries",
        cpu_capabilities().len()
    );
}

#[test]
fn the_descriptor_names_this_backend_and_its_axes() {
    let engine = CpuEngine::serial();
    assert_eq!(engine.backend_id(), BackendId::Cpu);

    let add_i64 = engine
        .capability(CapabilityQuery::new(PrimitiveOpKind::Add, DType::I64))
        .expect("CPU add/i64 should be described");
    assert_eq!(add_i64.result, SupportLevel::Native);
    assert_eq!(add_i64.read_inputs, SupportLevel::Native);
    assert_eq!(add_i64.write_output, SupportLevel::FallbackCopy);
    assert_eq!(add_i64.output_dtype, DType::I64);

    let dot_f64 = engine
        .capability(CapabilityQuery::new(
            PrimitiveOpKind::DotGeneral,
            DType::F64,
        ))
        .expect("CPU dot_general/f64 should be described");
    for axis in [
        CapabilityAxis::OwnedResult,
        CapabilityAxis::ReadInputs,
        CapabilityAxis::WriteOutput,
        CapabilityAxis::Accumulation,
    ] {
        assert_eq!(dot_f64.axis(axis), SupportLevel::Native, "{axis:?}");
    }

    assert!(
        engine
            .capability(CapabilityQuery::new(PrimitiveOpKind::Add, DType::Bool))
            .is_none(),
        "missing descriptor entries stay explicit rather than guessed"
    );

    // The axes below are declared-only (no test holds them to the engine), so they are pinned
    // here at least against accidental edits.
    for (op, dtype, write_output) in [
        (PrimitiveOpKind::Neg, DType::I32, SupportLevel::FallbackCopy),
        (PrimitiveOpKind::Div, DType::I32, SupportLevel::FallbackCopy),
        (PrimitiveOpKind::Rem, DType::I64, SupportLevel::Unsupported),
    ] {
        let entry = engine
            .capability(CapabilityQuery::new(op, dtype))
            .unwrap_or_else(|| panic!("CPU {op:?}/{dtype:?} should be described"));
        assert_eq!(entry.result, SupportLevel::Native, "{op:?}/{dtype:?}");
        assert_eq!(entry.write_output, write_output, "{op:?}/{dtype:?}");
    }
}
