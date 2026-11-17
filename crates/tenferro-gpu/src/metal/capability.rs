//! The Metal capability table.
//!
//! The table is plain `const` data over `tenferro-tensor`, so it compiles without the `metal`
//! feature and stays the one place a reader learns what the Metal backend serves. Rows exist
//! only for the operations U4 slice 1 wires; a later slice extends the table as it extends the
//! engine, and [`super::MetalEngine`]'s refusals name the same slices the table omits.

use tenferro_core_ops::PrimitiveOpKind;
use tenferro_tensor::{BackendId, DType, OperationCapability, SupportLevel};

const U: SupportLevel = SupportLevel::Unsupported;
const N: SupportLevel = SupportLevel::Native;

/// The backend identifier the Metal rows carry.
///
/// `BackendId` has no `Metal` variant, and adding one is a public enum change; the seam
/// spells a backend that is not one of the enumerated vendors as `Other`.
pub(crate) const METAL_BACKEND: BackendId = BackendId::Other("metal");

/// A Metal row: an owned result from in-place read operands, no borrowed or strided
/// destination, no accumulation.
const fn row(
    op: PrimitiveOpKind,
    dtype: DType,
    output_dtype: DType,
    level: SupportLevel,
) -> OperationCapability {
    OperationCapability {
        backend: METAL_BACKEND,
        op,
        dtype,
        output_dtype,
        result: level,
        read_inputs: level,
        write_output: U,
        strided_output: U,
        accumulation: U,
    }
}

const fn same(op: PrimitiveOpKind, dtype: DType, level: SupportLevel) -> OperationCapability {
    row(op, dtype, dtype, level)
}

const METAL_CAPABILITIES: &[OperationCapability] = &[
    same(PrimitiveOpKind::Add, DType::F32, N),
    same(PrimitiveOpKind::Add, DType::C32, N),
    same(PrimitiveOpKind::Sub, DType::F32, N),
    same(PrimitiveOpKind::Sub, DType::C32, N),
    same(PrimitiveOpKind::Mul, DType::F32, N),
    same(PrimitiveOpKind::Mul, DType::C32, N),
    // A complex divide is a scaled Gauss pair, not one `Binary` launch; it arrives with the
    // complex divide entry in U4 slice 2.
    same(PrimitiveOpKind::Div, DType::F32, N),
    same(PrimitiveOpKind::Div, DType::C32, U),
    same(PrimitiveOpKind::Neg, DType::F32, N),
    same(PrimitiveOpKind::Neg, DType::C32, N),
    same(PrimitiveOpKind::Conj, DType::F32, N),
    same(PrimitiveOpKind::Conj, DType::C32, N),
    same(PrimitiveOpKind::Sign, DType::F32, N),
    same(PrimitiveOpKind::Sign, DType::C32, N),
    // `abs` of a complex value is a real magnitude: the output dtype differs from the input.
    same(PrimitiveOpKind::Abs, DType::F32, N),
    row(PrimitiveOpKind::Abs, DType::C32, DType::F32, N),
];

/// The Metal backend's capability table.
#[must_use]
pub fn metal_capabilities() -> &'static [OperationCapability] {
    METAL_CAPABILITIES
}
