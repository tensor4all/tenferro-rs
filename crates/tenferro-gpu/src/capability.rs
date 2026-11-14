//! The CUDA backend capability table and its guide-table renderer.
//!
//! The table is plain `const` data over `tenferro-tensor` / `tenferro-core-ops`, so it
//! compiles without the `cuda` feature: the table in `docs/guides/devices-and-gpu.md` is
//! rendered from it and checked by a default-features test (issue #2477). Only the table's
//! readers — the `TensorBackendCapability` impl for `CudaEngine` and the public
//! `tenferro_gpu::cuda::cuda_capabilities()` re-export — stay behind the feature.

use tenferro_core_ops::PrimitiveOpKind;
use tenferro_tensor::{BackendId, DType, OperationCapability, SupportLevel};

const U: SupportLevel = SupportLevel::Unsupported;
const F: SupportLevel = SupportLevel::FallbackCopy;
const N: SupportLevel = SupportLevel::Native;

/// The CUDA capability table.
///
/// `crate::cuda::cuda_capabilities()` is its public spelling and exists only with the `cuda`
/// feature, so it cannot be a doc link here.
pub(crate) const fn cuda_capability_table() -> &'static [OperationCapability] {
    CUDA_CAPABILITIES
}

const fn owned_level(
    op: PrimitiveOpKind,
    dtype: DType,
    output_dtype: DType,
    level: SupportLevel,
) -> OperationCapability {
    OperationCapability {
        backend: BackendId::Cuda,
        op,
        dtype,
        output_dtype,
        result: level,
        read_inputs: if matches!(level, SupportLevel::Native) {
            F
        } else {
            U
        },
        write_output: U,
        strided_output: U,
        accumulation: U,
    }
}

const fn same_dtype(op: PrimitiveOpKind, dtype: DType, level: SupportLevel) -> OperationCapability {
    owned_level(op, dtype, dtype, level)
}

const fn dot_native(dtype: DType) -> OperationCapability {
    OperationCapability {
        backend: BackendId::Cuda,
        op: PrimitiveOpKind::DotGeneral,
        dtype,
        output_dtype: dtype,
        result: N,
        // A borrowed operand reaches cuBLAS through a compact copy: cuBLAS takes no strides.
        read_inputs: F,
        write_output: N,
        // The dotted result is written into an owned output only; a strided destination region
        // is refused by both `dot_general_read_into` entries.
        strided_output: U,
        accumulation: N,
    }
}

const CUDA_CAPABILITIES: &[OperationCapability] = &[
    same_dtype(PrimitiveOpKind::Add, DType::F32, N),
    same_dtype(PrimitiveOpKind::Add, DType::F64, N),
    same_dtype(PrimitiveOpKind::Add, DType::I32, N),
    same_dtype(PrimitiveOpKind::Add, DType::I64, N),
    same_dtype(PrimitiveOpKind::Add, DType::C32, N),
    same_dtype(PrimitiveOpKind::Add, DType::C64, N),
    same_dtype(PrimitiveOpKind::Sub, DType::F32, N),
    same_dtype(PrimitiveOpKind::Sub, DType::F64, N),
    same_dtype(PrimitiveOpKind::Sub, DType::I32, N),
    same_dtype(PrimitiveOpKind::Sub, DType::I64, N),
    same_dtype(PrimitiveOpKind::Sub, DType::C32, N),
    same_dtype(PrimitiveOpKind::Sub, DType::C64, N),
    same_dtype(PrimitiveOpKind::Mul, DType::F32, N),
    same_dtype(PrimitiveOpKind::Mul, DType::F64, N),
    same_dtype(PrimitiveOpKind::Mul, DType::I32, N),
    same_dtype(PrimitiveOpKind::Mul, DType::I64, N),
    same_dtype(PrimitiveOpKind::Mul, DType::C32, N),
    same_dtype(PrimitiveOpKind::Mul, DType::C64, N),
    same_dtype(PrimitiveOpKind::Neg, DType::F32, N),
    same_dtype(PrimitiveOpKind::Neg, DType::F64, N),
    same_dtype(PrimitiveOpKind::Neg, DType::I32, N),
    same_dtype(PrimitiveOpKind::Neg, DType::I64, N),
    same_dtype(PrimitiveOpKind::Neg, DType::C32, N),
    same_dtype(PrimitiveOpKind::Neg, DType::C64, N),
    same_dtype(PrimitiveOpKind::Conj, DType::F32, N),
    same_dtype(PrimitiveOpKind::Conj, DType::F64, N),
    same_dtype(PrimitiveOpKind::Conj, DType::C32, N),
    same_dtype(PrimitiveOpKind::Conj, DType::C64, N),
    same_dtype(PrimitiveOpKind::Div, DType::F32, N),
    same_dtype(PrimitiveOpKind::Div, DType::F64, N),
    same_dtype(PrimitiveOpKind::Div, DType::I32, N),
    same_dtype(PrimitiveOpKind::Div, DType::I64, N),
    same_dtype(PrimitiveOpKind::Div, DType::C32, N),
    same_dtype(PrimitiveOpKind::Div, DType::C64, N),
    same_dtype(PrimitiveOpKind::Rem, DType::F32, N),
    same_dtype(PrimitiveOpKind::Rem, DType::F64, N),
    same_dtype(PrimitiveOpKind::Rem, DType::I32, N),
    same_dtype(PrimitiveOpKind::Rem, DType::I64, N),
    same_dtype(PrimitiveOpKind::Abs, DType::F32, N),
    same_dtype(PrimitiveOpKind::Abs, DType::F64, N),
    same_dtype(PrimitiveOpKind::Abs, DType::I32, N),
    same_dtype(PrimitiveOpKind::Abs, DType::I64, N),
    owned_level(PrimitiveOpKind::Abs, DType::C32, DType::F32, N),
    owned_level(PrimitiveOpKind::Abs, DType::C64, DType::F64, N),
    same_dtype(PrimitiveOpKind::Sign, DType::F32, N),
    same_dtype(PrimitiveOpKind::Sign, DType::F64, N),
    same_dtype(PrimitiveOpKind::Sign, DType::I32, N),
    same_dtype(PrimitiveOpKind::Sign, DType::I64, N),
    same_dtype(PrimitiveOpKind::Sign, DType::C32, N),
    same_dtype(PrimitiveOpKind::Sign, DType::C64, N),
    same_dtype(PrimitiveOpKind::Maximum, DType::F32, N),
    same_dtype(PrimitiveOpKind::Maximum, DType::F64, N),
    same_dtype(PrimitiveOpKind::Maximum, DType::I32, N),
    same_dtype(PrimitiveOpKind::Maximum, DType::I64, N),
    same_dtype(PrimitiveOpKind::Minimum, DType::F32, N),
    same_dtype(PrimitiveOpKind::Minimum, DType::F64, N),
    same_dtype(PrimitiveOpKind::Minimum, DType::I32, N),
    same_dtype(PrimitiveOpKind::Minimum, DType::I64, N),
    owned_level(PrimitiveOpKind::Compare, DType::F32, DType::Bool, N),
    owned_level(PrimitiveOpKind::Compare, DType::F64, DType::Bool, N),
    owned_level(PrimitiveOpKind::Compare, DType::I32, DType::Bool, N),
    owned_level(PrimitiveOpKind::Compare, DType::I64, DType::Bool, N),
    owned_level(PrimitiveOpKind::Compare, DType::Bool, DType::Bool, U),
    same_dtype(PrimitiveOpKind::Select, DType::F32, N),
    same_dtype(PrimitiveOpKind::Select, DType::F64, N),
    same_dtype(PrimitiveOpKind::Select, DType::I32, N),
    same_dtype(PrimitiveOpKind::Select, DType::I64, N),
    same_dtype(PrimitiveOpKind::Select, DType::Bool, U),
    same_dtype(PrimitiveOpKind::Select, DType::C32, U),
    same_dtype(PrimitiveOpKind::Select, DType::C64, U),
    same_dtype(PrimitiveOpKind::Clamp, DType::F32, N),
    same_dtype(PrimitiveOpKind::Clamp, DType::F64, N),
    same_dtype(PrimitiveOpKind::Exp, DType::F32, N),
    same_dtype(PrimitiveOpKind::Exp, DType::F64, N),
    same_dtype(PrimitiveOpKind::Exp, DType::C32, U),
    same_dtype(PrimitiveOpKind::Exp, DType::C64, U),
    same_dtype(PrimitiveOpKind::Log, DType::F32, N),
    same_dtype(PrimitiveOpKind::Log, DType::F64, N),
    same_dtype(PrimitiveOpKind::Log, DType::C32, U),
    same_dtype(PrimitiveOpKind::Log, DType::C64, U),
    same_dtype(PrimitiveOpKind::Sin, DType::F32, N),
    same_dtype(PrimitiveOpKind::Sin, DType::F64, N),
    same_dtype(PrimitiveOpKind::Sin, DType::C32, U),
    same_dtype(PrimitiveOpKind::Sin, DType::C64, U),
    same_dtype(PrimitiveOpKind::Cos, DType::F32, N),
    same_dtype(PrimitiveOpKind::Cos, DType::F64, N),
    same_dtype(PrimitiveOpKind::Cos, DType::C32, U),
    same_dtype(PrimitiveOpKind::Cos, DType::C64, U),
    same_dtype(PrimitiveOpKind::Tanh, DType::F32, N),
    same_dtype(PrimitiveOpKind::Tanh, DType::F64, N),
    same_dtype(PrimitiveOpKind::Tanh, DType::C32, U),
    same_dtype(PrimitiveOpKind::Tanh, DType::C64, U),
    same_dtype(PrimitiveOpKind::Sqrt, DType::F32, N),
    same_dtype(PrimitiveOpKind::Sqrt, DType::F64, N),
    same_dtype(PrimitiveOpKind::Sqrt, DType::C32, U),
    same_dtype(PrimitiveOpKind::Sqrt, DType::C64, U),
    same_dtype(PrimitiveOpKind::Rsqrt, DType::F32, N),
    same_dtype(PrimitiveOpKind::Rsqrt, DType::F64, N),
    same_dtype(PrimitiveOpKind::Rsqrt, DType::C32, U),
    same_dtype(PrimitiveOpKind::Rsqrt, DType::C64, U),
    same_dtype(PrimitiveOpKind::Pow, DType::F32, N),
    same_dtype(PrimitiveOpKind::Pow, DType::F64, N),
    same_dtype(PrimitiveOpKind::Pow, DType::I32, N),
    same_dtype(PrimitiveOpKind::Pow, DType::I64, N),
    same_dtype(PrimitiveOpKind::Pow, DType::C32, U),
    same_dtype(PrimitiveOpKind::Pow, DType::C64, U),
    same_dtype(PrimitiveOpKind::Expm1, DType::F32, N),
    same_dtype(PrimitiveOpKind::Expm1, DType::F64, N),
    same_dtype(PrimitiveOpKind::Expm1, DType::C32, U),
    same_dtype(PrimitiveOpKind::Expm1, DType::C64, U),
    same_dtype(PrimitiveOpKind::Log1p, DType::F32, N),
    same_dtype(PrimitiveOpKind::Log1p, DType::F64, N),
    same_dtype(PrimitiveOpKind::Log1p, DType::C32, U),
    same_dtype(PrimitiveOpKind::Log1p, DType::C64, U),
    same_dtype(PrimitiveOpKind::Erf, DType::F32, N),
    same_dtype(PrimitiveOpKind::Erf, DType::F64, N),
    dot_native(DType::F32),
    dot_native(DType::F64),
    dot_native(DType::C32),
    dot_native(DType::C64),
    same_dtype(PrimitiveOpKind::ReduceSum, DType::F32, N),
    same_dtype(PrimitiveOpKind::ReduceSum, DType::F64, N),
    same_dtype(PrimitiveOpKind::ReduceSum, DType::I32, N),
    same_dtype(PrimitiveOpKind::ReduceSum, DType::I64, N),
    same_dtype(PrimitiveOpKind::ReduceSum, DType::C32, N),
    same_dtype(PrimitiveOpKind::ReduceSum, DType::C64, N),
    same_dtype(PrimitiveOpKind::ReduceSumSquares, DType::F32, N),
    same_dtype(PrimitiveOpKind::ReduceSumSquares, DType::F64, N),
    same_dtype(PrimitiveOpKind::ReduceProd, DType::F32, N),
    same_dtype(PrimitiveOpKind::ReduceProd, DType::F64, N),
    same_dtype(PrimitiveOpKind::ReduceProd, DType::I32, N),
    same_dtype(PrimitiveOpKind::ReduceProd, DType::I64, N),
    same_dtype(PrimitiveOpKind::ReduceProd, DType::C32, N),
    same_dtype(PrimitiveOpKind::ReduceProd, DType::C64, N),
    same_dtype(PrimitiveOpKind::ReduceMax, DType::F32, N),
    same_dtype(PrimitiveOpKind::ReduceMax, DType::F64, N),
    same_dtype(PrimitiveOpKind::ReduceMax, DType::I32, N),
    same_dtype(PrimitiveOpKind::ReduceMax, DType::I64, N),
    same_dtype(PrimitiveOpKind::ReduceMin, DType::F32, N),
    same_dtype(PrimitiveOpKind::ReduceMin, DType::F64, N),
    same_dtype(PrimitiveOpKind::ReduceMin, DType::I32, N),
    same_dtype(PrimitiveOpKind::ReduceMin, DType::I64, N),
];

/// Render the guide's core-primitive table from the capability table.
///
/// Rows follow the table's order, one row per operation; the axis levels of an operation's
/// native entries must agree, because the guide states one axis line per row.
#[cfg(test)]
fn render_capability_markdown() -> String {
    const HEADER: &str = "| Primitive op | Native CUDA dtypes | Unsupported descriptor dtypes | Output dtype | Axis levels for native dtypes |";
    const RULE: &str = "| --- | --- | --- | --- | --- |";

    fn dtypes(entries: &[&OperationCapability]) -> String {
        if entries.is_empty() {
            return "none".to_string();
        }
        entries
            .iter()
            .map(|entry| format!("`{:?}`", entry.dtype))
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn axis_levels(entry: &OperationCapability) -> String {
        format!(
            "result {:?}; read {:?}; write {:?}; strided {:?}; accumulation {:?}",
            entry.result,
            entry.read_inputs,
            entry.write_output,
            entry.strided_output,
            entry.accumulation
        )
    }

    let mut ops: Vec<PrimitiveOpKind> = Vec::new();
    for entry in CUDA_CAPABILITIES {
        if !ops.contains(&entry.op) {
            ops.push(entry.op);
        }
    }

    let mut out = String::new();
    out.push_str(HEADER);
    out.push('\n');
    out.push_str(RULE);
    out.push('\n');
    for op in ops {
        let rows: Vec<&OperationCapability> = CUDA_CAPABILITIES
            .iter()
            .filter(|entry| entry.op == op)
            .collect();
        let native: Vec<&OperationCapability> = rows
            .iter()
            .copied()
            .filter(|entry| entry.result.is_supported())
            .collect();
        let refused: Vec<&OperationCapability> = rows
            .iter()
            .copied()
            .filter(|entry| !entry.result.is_supported())
            .collect();

        let output_dtype = if native.iter().all(|entry| entry.output_dtype == entry.dtype) {
            "same as input".to_string()
        } else if let Some(first) = native.first() {
            if native
                .iter()
                .all(|entry| entry.output_dtype == first.output_dtype)
            {
                format!("`{:?}`", first.output_dtype)
            } else {
                native
                    .iter()
                    .map(|entry| format!("`{:?}->{:?}`", entry.dtype, entry.output_dtype))
                    .collect::<Vec<_>>()
                    .join(", ")
            }
        } else {
            "none".to_string()
        };

        let first_native = native
            .first()
            .expect("every catalogued operation has at least one native dtype");
        assert!(
            native
                .iter()
                .all(|entry| axis_levels(entry) == axis_levels(first_native)),
            "{} declares different axis levels per dtype; the guide row states one line, so \
             the renderer must group the dtypes before this table can be regenerated",
            tenferro_core_ops::descriptor(op).name
        );

        out.push_str(&format!(
            "| `{}` | {} | {} | {} | {} |\n",
            tenferro_core_ops::descriptor(op).name,
            dtypes(&native),
            dtypes(&refused),
            output_dtype,
            axis_levels(first_native)
        ));
    }
    out
}

#[cfg(test)]
mod tests;
