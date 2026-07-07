use tenferro_core_ops::{descriptor as primitive_descriptor, DTypePolicy, PrimitiveOpKind};

/// Host-side launch family used by the CubeCL primitive dispatcher.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GpuLaunchKind {
    BinaryFloatComplex,
    BinaryFloatComplexInt,
    BinaryFloatInt,
    BinaryFloatOnly,
    UnaryFloatComplex,
    UnaryFloatComplexInt,
    UnaryFloatInt,
    UnaryFloatOnly,
    CompareFloatIntToBool,
    SelectBoolFloatInt,
    ClampFloat,
    Reduction,
}

/// GPU-specific primitive metadata derived from the core operation catalog.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GpuOpDescriptor {
    pub(crate) kind: PrimitiveOpKind,
    pub(crate) name: &'static str,
    pub(crate) dtype_policy: DTypePolicy,
    pub(crate) launch: GpuLaunchKind,
}

pub(crate) fn gpu_descriptor(kind: PrimitiveOpKind) -> Option<GpuOpDescriptor> {
    let launch = match kind {
        PrimitiveOpKind::Add | PrimitiveOpKind::Mul => GpuLaunchKind::BinaryFloatComplexInt,
        PrimitiveOpKind::Div => GpuLaunchKind::BinaryFloatComplex,
        PrimitiveOpKind::Maximum | PrimitiveOpKind::Minimum => GpuLaunchKind::BinaryFloatInt,
        PrimitiveOpKind::Pow => GpuLaunchKind::BinaryFloatOnly,
        PrimitiveOpKind::Neg => GpuLaunchKind::UnaryFloatComplexInt,
        PrimitiveOpKind::Conj => GpuLaunchKind::UnaryFloatComplex,
        PrimitiveOpKind::Abs | PrimitiveOpKind::Sign => GpuLaunchKind::UnaryFloatInt,
        PrimitiveOpKind::Exp
        | PrimitiveOpKind::Log
        | PrimitiveOpKind::Sin
        | PrimitiveOpKind::Cos
        | PrimitiveOpKind::Tanh
        | PrimitiveOpKind::Sqrt
        | PrimitiveOpKind::Rsqrt
        | PrimitiveOpKind::Expm1
        | PrimitiveOpKind::Log1p => GpuLaunchKind::UnaryFloatOnly,
        PrimitiveOpKind::Compare => GpuLaunchKind::CompareFloatIntToBool,
        PrimitiveOpKind::Select => GpuLaunchKind::SelectBoolFloatInt,
        PrimitiveOpKind::Clamp => GpuLaunchKind::ClampFloat,
        PrimitiveOpKind::ReduceSum
        | PrimitiveOpKind::ReduceProd
        | PrimitiveOpKind::ReduceMax
        | PrimitiveOpKind::ReduceMin => GpuLaunchKind::Reduction,
        _ => return None,
    };

    let descriptor = primitive_descriptor(kind);
    Some(GpuOpDescriptor {
        kind,
        name: descriptor.name,
        dtype_policy: descriptor.dtype_policy,
        launch,
    })
}

pub(crate) fn require_gpu_descriptor(
    kind: PrimitiveOpKind,
    expected: GpuLaunchKind,
) -> crate::Result<GpuOpDescriptor> {
    let descriptor = gpu_descriptor(kind).ok_or_else(|| {
        crate::Error::backend_failure(
            "gpu_dispatch",
            format!("primitive {kind:?} is not implemented by CubeCL dispatch"),
        )
    })?;
    debug_assert_eq!(descriptor.launch, expected);
    Ok(descriptor)
}

#[cfg(test)]
mod tests;
