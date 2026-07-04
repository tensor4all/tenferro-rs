use std::any::Any;
use std::hash::Hasher;
use std::sync::Arc;

use tenferro_extension_macros::define_extension_runtime;
use tenferro_ops::SymDim;
use tenferro_runtime::extension::{ExtensionExecutionContext, ExtensionOp, HostReference};
use tenferro_tensor::{
    DType, DeviceKind, Error, GpuBackendKind, MemoryKind, Placement, Tensor, TensorRead,
};

use crate::backend::LinalgBackend;

#[cfg(all(test, not(feature = "cuda")))]
mod tests;

pub const LINALG_EXTENSION_FAMILY_ID: &str = "tenferro-linalg.linalg.v1";

pub(crate) const DEFAULT_DECOMPOSITION_AD_EPS: f64 = 1e-12;

#[derive(Clone, Copy, Debug, PartialEq)]
#[doc(hidden)]
pub(crate) enum LinalgOp {
    Cholesky,
    Lu,
    LuFactor,
    LuSolvePrepared {
        transpose_a: bool,
        conjugate_a: bool,
    },
    FullPivLu,
    FullPivLuSolve {
        transpose_a: bool,
    },
    Svd {
        eps: f64,
    },
    SvdVals {
        eps: f64,
    },
    Qr,
    Eigh {
        eps: f64,
    },
    EighVals {
        eps: f64,
    },
    Eig {
        input_dtype: DType,
    },
    EigVals {
        input_dtype: DType,
    },
    TriangularSolve {
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
    },
}

impl LinalgOp {
    fn output_count(self) -> usize {
        match self {
            Self::Cholesky
            | Self::EighVals { .. }
            | Self::EigVals { .. }
            | Self::FullPivLuSolve { .. }
            | Self::LuSolvePrepared { .. }
            | Self::SvdVals { .. }
            | Self::TriangularSolve { .. } => 1,
            Self::Svd { .. } => 3,
            Self::Qr | Self::Eigh { .. } | Self::Eig { .. } => 2,
            Self::LuFactor => 3,
            Self::Lu => 4,
            Self::FullPivLu => 5,
        }
    }

    fn input_count(self) -> usize {
        match self {
            Self::FullPivLuSolve { .. } | Self::TriangularSolve { .. } => 2,
            Self::LuSolvePrepared { .. } => 4,
            _ => 1,
        }
    }

    fn tag(self) -> u8 {
        match self {
            Self::Cholesky => 0,
            Self::Lu => 1,
            Self::FullPivLu => 2,
            Self::FullPivLuSolve { .. } => 3,
            Self::Svd { .. } => 4,
            Self::Qr => 5,
            Self::Eigh { .. } => 6,
            Self::Eig { .. } => 7,
            Self::TriangularSolve { .. } => 9,
            Self::LuFactor => 10,
            Self::LuSolvePrepared { .. } => 11,
            Self::SvdVals { .. } => 12,
            Self::EighVals { .. } => 13,
            Self::EigVals { .. } => 14,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
#[doc(hidden)]
pub(crate) struct LinalgExtensionOp {
    op: LinalgOp,
}

impl LinalgExtensionOp {
    pub(crate) fn new(op: LinalgOp) -> Self {
        Self { op }
    }

    pub(crate) fn op(&self) -> LinalgOp {
        self.op
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EagerLinalgDevice {
    Cpu,
    Cuda(usize),
}

fn tensor_placement(input: &Tensor) -> &Placement {
    input.placement()
}

fn input_eager_device(input: &Tensor) -> tenferro_tensor::Result<EagerLinalgDevice> {
    let placement = tensor_placement(input);
    match (&placement.memory_kind, placement.device.as_ref()) {
        (MemoryKind::Device, Some(device)) => match &device.kind {
            DeviceKind::Gpu(GpuBackendKind::Cuda) => Ok(EagerLinalgDevice::Cuda(device.ordinal)),
            DeviceKind::Gpu(kind) => Err(Error::backend_failure(
                "linalg_host_reference",
                format!("unsupported GPU backend {kind:?} for eager linalg"),
            )),
            kind => Err(Error::backend_failure(
                "linalg_host_reference",
                format!("unsupported device kind {kind:?} for eager linalg"),
            )),
        },
        (MemoryKind::Device, None) => Err(Error::backend_failure(
            "linalg_host_reference",
            "device tensor is missing placement device metadata",
        )),
        _ => Ok(EagerLinalgDevice::Cpu),
    }
}

fn eager_linalg_device(inputs: &[&Tensor]) -> tenferro_tensor::Result<EagerLinalgDevice> {
    let mut selected = None;
    for input in inputs {
        let device = input_eager_device(input)?;
        match (selected, device) {
            (None, next) => selected = Some(next),
            (Some(EagerLinalgDevice::Cpu), EagerLinalgDevice::Cpu) => {}
            (Some(EagerLinalgDevice::Cuda(lhs)), EagerLinalgDevice::Cuda(rhs)) if lhs == rhs => {}
            (Some(lhs), rhs) => {
                return Err(Error::backend_failure(
                    "linalg_host_reference",
                    format!("all eager linalg inputs must be on the same device, got {lhs:?} and {rhs:?}"),
                ));
            }
        }
    }
    Ok(selected.unwrap_or(EagerLinalgDevice::Cpu))
}

#[cfg(feature = "cuda")]
fn execute_cuda_eager_linalg(
    op: LinalgOp,
    inputs: &[&Tensor],
    device_ordinal: usize,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    let mut backend = tenferro_gpu::CudaBackend::new(device_ordinal)?;
    execute_linalg(op, inputs, &mut backend)
}

#[cfg(not(feature = "cuda"))]
fn execute_cuda_eager_linalg(
    _op: LinalgOp,
    _inputs: &[&Tensor],
    device_ordinal: usize,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    Err(Error::backend_failure(
        "linalg_host_reference",
        format!(
            "received CUDA tensor on cuda:{device_ordinal}, but tenferro-linalg was built \
             without the cuda feature; enable the cuda feature or download the tensor to CPU \
             before eager linalg"
        ),
    ))
}

impl ExtensionOp for LinalgExtensionOp {
    fn family_id(&self) -> &'static str {
        LINALG_EXTENSION_FAMILY_ID
    }

    fn payload_hash(&self, hasher: &mut dyn Hasher) {
        hasher.write_u8(self.op.tag());
        match self.op {
            LinalgOp::Svd { eps }
            | LinalgOp::SvdVals { eps }
            | LinalgOp::Eigh { eps }
            | LinalgOp::EighVals { eps } => hasher.write_u64(eps.to_bits()),
            LinalgOp::Eig { input_dtype } | LinalgOp::EigVals { input_dtype } => {
                hash_dtype(hasher, input_dtype);
            }
            LinalgOp::FullPivLuSolve { transpose_a } => {
                hasher.write_u8(u8::from(transpose_a));
            }
            LinalgOp::LuSolvePrepared {
                transpose_a,
                conjugate_a,
            } => {
                hasher.write_u8(u8::from(transpose_a));
                hasher.write_u8(u8::from(conjugate_a));
            }
            LinalgOp::TriangularSolve {
                left_side,
                lower,
                transpose_a,
                unit_diagonal,
            } => {
                hasher.write_u8(u8::from(left_side));
                hasher.write_u8(u8::from(lower));
                hasher.write_u8(u8::from(transpose_a));
                hasher.write_u8(u8::from(unit_diagonal));
            }
            LinalgOp::Cholesky
            | LinalgOp::Lu
            | LinalgOp::LuFactor
            | LinalgOp::FullPivLu
            | LinalgOp::Qr => {}
        }
    }

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|that| self == that)
    }

    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn input_count(&self) -> usize {
        self.op.input_count()
    }

    fn output_count(&self) -> usize {
        self.op.output_count()
    }

    fn infer_output_meta(
        &self,
        input_dtypes: &[DType],
        input_shapes: &[&[SymDim]],
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
        if input_dtypes.len() != self.input_count() || input_shapes.len() != self.input_count() {
            return Err(Error::InvalidConfig {
                op: "tenferro-linalg",
                message: format!(
                    "expected {} input metadata entries, got dtypes={} shapes={}",
                    self.input_count(),
                    input_dtypes.len(),
                    input_shapes.len()
                ),
            });
        }
        let metas = match self.op {
            LinalgOp::Cholesky => {
                require_matrix_meta("tenferro-linalg.cholesky", input_shapes[0])?;
                vec![(promote_dtypes(input_dtypes), input_shapes[0].to_vec())]
            }
            LinalgOp::FullPivLuSolve { .. } => {
                require_matrix_meta("tenferro-linalg.full_piv_lu_solve", input_shapes[0])?;
                require_matrix_meta("tenferro-linalg.full_piv_lu_solve", input_shapes[1])?;
                vec![(promote_dtypes(input_dtypes), input_shapes[1].to_vec())]
            }
            LinalgOp::TriangularSolve { .. } => {
                require_matrix_meta("tenferro-linalg.triangular_solve", input_shapes[0])?;
                require_matrix_meta("tenferro-linalg.triangular_solve", input_shapes[1])?;
                vec![(promote_dtypes(input_dtypes), input_shapes[1].to_vec())]
            }
            LinalgOp::LuSolvePrepared { .. } => {
                require_matrix_meta("tenferro-linalg.lu_solve_prepared_lu", input_shapes[0])?;
                require_matrix_meta("tenferro-linalg.lu_solve_prepared_rhs", input_shapes[3])?;
                vec![(
                    promote_dtypes(&[input_dtypes[0], input_dtypes[3]]),
                    input_shapes[3].to_vec(),
                )]
            }
            LinalgOp::Lu => lu_meta(input_dtypes[0], input_shapes[0])?,
            LinalgOp::LuFactor => lu_factor_meta(input_dtypes[0], input_shapes[0])?,
            LinalgOp::FullPivLu => full_piv_lu_meta(input_dtypes[0], input_shapes[0])?,
            LinalgOp::Svd { .. } => svd_meta(input_dtypes[0], input_shapes[0])?,
            LinalgOp::SvdVals { .. } => {
                vec![svd_values_meta(input_dtypes[0], input_shapes[0])?]
            }
            LinalgOp::Qr => qr_meta(input_dtypes[0], input_shapes[0])?,
            LinalgOp::Eigh { .. } => eigh_meta(input_dtypes[0], input_shapes[0])?,
            LinalgOp::EighVals { .. } => vec![eigh_values_meta(input_dtypes[0], input_shapes[0])?],
            LinalgOp::Eig { input_dtype } => eig_meta(input_dtype, input_shapes[0])?,
            LinalgOp::EigVals { input_dtype } => {
                vec![eig_values_meta(input_dtype, input_shapes[0])?]
            }
        };
        Ok(metas)
    }

    fn host_reference(&self) -> Option<&dyn HostReference> {
        Some(self)
    }
}

impl HostReference for LinalgExtensionOp {
    fn execute(&self, inputs: &[&Tensor]) -> tenferro_tensor::Result<Vec<Tensor>> {
        let expected = self.input_count();
        if inputs.len() != expected {
            return Err(Error::InvalidConfig {
                op: "linalg_host_reference",
                message: format!(
                    "expected {expected} inputs for {:?}, got {}",
                    self.op,
                    inputs.len()
                ),
            });
        }

        match eager_linalg_device(inputs)? {
            EagerLinalgDevice::Cpu => {
                let mut backend = tenferro_cpu::CpuBackend::new();
                execute_linalg(self.op, inputs, &mut backend)
            }
            EagerLinalgDevice::Cuda(device_ordinal) => {
                execute_cuda_eager_linalg(self.op, inputs, device_ordinal)
            }
        }
    }
}

fn execute_linalg_extension<B: LinalgBackend + 'static>(
    op: &LinalgExtensionOp,
    inputs: &[&Tensor],
    ctx: &mut ExtensionExecutionContext<'_, B>,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    execute_linalg(op.op(), inputs, ctx.backend_mut())
}

fn execute_linalg_extension_reads<B: LinalgBackend + 'static>(
    op: &LinalgExtensionOp,
    inputs: &[TensorRead<'_>],
    ctx: &mut ExtensionExecutionContext<'_, B>,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    // Linalg backends currently operate on compact tensors; materialization is
    // explicit here so borrowed views cannot silently bypass backend errors.
    let materialized_inputs: Vec<Tensor> = inputs
        .iter()
        .map(TensorRead::to_tensor)
        .collect::<tenferro_tensor::Result<_>>()?;
    let input_refs: Vec<&Tensor> = materialized_inputs.iter().collect();
    execute_linalg_extension(op, &input_refs, ctx)
}

define_extension_runtime! {
    runtime = LinalgRuntime,
    family_id = LINALG_EXTENSION_FAMILY_ID,
    op_type = LinalgExtensionOp,
    execute = execute_linalg_extension,
    execute_reads = execute_linalg_extension_reads,
    register_fn = register_runtime,
    backend_bound = LinalgBackend,
}

fn execute_linalg<B: LinalgBackend>(
    op: LinalgOp,
    inputs: &[&Tensor],
    backend: &mut B,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    match op {
        LinalgOp::Cholesky => Ok(vec![backend.cholesky(inputs[0])?]),
        LinalgOp::Lu => backend.lu(inputs[0]),
        LinalgOp::LuFactor => backend.lu_factor(inputs[0]),
        LinalgOp::LuSolvePrepared {
            transpose_a,
            conjugate_a,
        } => Ok(vec![backend.lu_solve_prepared(
            inputs[0],
            inputs[1],
            inputs[2],
            inputs[3],
            transpose_a,
            conjugate_a,
        )?]),
        LinalgOp::FullPivLu => backend.full_piv_lu(inputs[0]),
        LinalgOp::FullPivLuSolve { transpose_a } => Ok(vec![backend.full_piv_lu_solve(
            inputs[0],
            inputs[1],
            transpose_a,
        )?]),
        LinalgOp::Svd { .. } => backend.svd(inputs[0]),
        LinalgOp::SvdVals { .. } => Ok(vec![backend.svd_values(inputs[0])?]),
        LinalgOp::Qr => backend.qr(inputs[0]),
        LinalgOp::Eigh { .. } => backend.eigh(inputs[0]),
        LinalgOp::EighVals { .. } => Ok(vec![backend.eigh_values(inputs[0])?]),
        LinalgOp::Eig { .. } => backend.eig(inputs[0]),
        LinalgOp::EigVals { .. } => Ok(vec![backend.eig_values(inputs[0])?]),
        LinalgOp::TriangularSolve {
            left_side,
            lower,
            transpose_a,
            unit_diagonal,
        } => Ok(vec![backend.triangular_solve(
            inputs[0],
            inputs[1],
            left_side,
            lower,
            transpose_a,
            unit_diagonal,
        )?]),
    }
}

fn require_matrix_meta(op: &'static str, shape: &[SymDim]) -> tenferro_tensor::Result<()> {
    if shape.len() < 2 {
        return Err(Error::RankMismatch {
            op,
            expected: 2,
            actual: shape.len(),
        });
    }
    Ok(())
}

fn matrix_meta_parts<'a>(
    op: &'static str,
    shape: &'a [SymDim],
) -> tenferro_tensor::Result<(SymDim, SymDim, &'a [SymDim])> {
    require_matrix_meta(op, shape)?;
    Ok((shape[0].clone(), shape[1].clone(), &shape[2..]))
}

fn lu_meta(dtype: DType, shape: &[SymDim]) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
    let (m, n, batch) = matrix_meta_parts("tenferro-linalg.lu", shape)?;
    let k = m.clone().min(n.clone());
    Ok(vec![
        (dtype, matrix_shape(m.clone(), m, batch)),
        (dtype, matrix_shape(shape[0].clone(), k.clone(), batch)),
        (dtype, matrix_shape(k, n, batch)),
        (dtype, batch.to_vec()),
    ])
}

fn lu_factor_meta(
    dtype: DType,
    shape: &[SymDim],
) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
    let (m, n, batch) = matrix_meta_parts("tenferro-linalg.lu_factor", shape)?;
    let k = m.min(n);
    Ok(vec![
        (dtype, shape.to_vec()),
        (DType::I32, vector_shape(k, batch)),
        (dtype, batch.to_vec()),
    ])
}

fn full_piv_lu_meta(
    dtype: DType,
    shape: &[SymDim],
) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
    let (n, _, batch) = matrix_meta_parts("tenferro-linalg.full_piv_lu", shape)?;
    Ok(vec![
        (dtype, matrix_shape(n.clone(), n.clone(), batch)),
        (dtype, matrix_shape(n.clone(), n.clone(), batch)),
        (dtype, matrix_shape(n.clone(), n.clone(), batch)),
        (dtype, matrix_shape(n.clone(), n, batch)),
        (singular_values_dtype(dtype), batch.to_vec()),
    ])
}

fn svd_meta(dtype: DType, shape: &[SymDim]) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
    let (m, n, batch) = matrix_meta_parts("tenferro-linalg.svd", shape)?;
    let k = m.clone().min(n.clone());
    Ok(vec![
        (dtype, matrix_shape(m, k.clone(), batch)),
        (singular_values_dtype(dtype), vector_shape(k.clone(), batch)),
        (dtype, matrix_shape(k, n, batch)),
    ])
}

fn svd_values_meta(
    dtype: DType,
    shape: &[SymDim],
) -> tenferro_tensor::Result<(DType, Vec<SymDim>)> {
    let (m, n, batch) = matrix_meta_parts("tenferro-linalg.svd_values", shape)?;
    let k = m.min(n);
    Ok((singular_values_dtype(dtype), vector_shape(k, batch)))
}

fn qr_meta(dtype: DType, shape: &[SymDim]) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
    let (m, n, batch) = matrix_meta_parts("tenferro-linalg.qr", shape)?;
    let k = m.clone().min(n.clone());
    Ok(vec![
        (dtype, matrix_shape(m, k.clone(), batch)),
        (dtype, matrix_shape(k, n, batch)),
    ])
}

fn eigh_meta(dtype: DType, shape: &[SymDim]) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
    let (n, _, batch) = matrix_meta_parts("tenferro-linalg.eigh", shape)?;
    Ok(vec![
        (singular_values_dtype(dtype), vector_shape(n.clone(), batch)),
        (dtype, matrix_shape(n.clone(), n, batch)),
    ])
}

fn eigh_values_meta(
    dtype: DType,
    shape: &[SymDim],
) -> tenferro_tensor::Result<(DType, Vec<SymDim>)> {
    let (n, _, batch) = matrix_meta_parts("tenferro-linalg.eigh_values", shape)?;
    Ok((singular_values_dtype(dtype), vector_shape(n, batch)))
}

fn eig_meta(
    input_dtype: DType,
    shape: &[SymDim],
) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
    let dtype = eig_output_dtype(input_dtype);
    let (n, _, batch) = matrix_meta_parts("tenferro-linalg.eig", shape)?;
    Ok(vec![
        (dtype, vector_shape(n.clone(), batch)),
        (dtype, matrix_shape(n.clone(), n, batch)),
    ])
}

fn eig_values_meta(
    input_dtype: DType,
    shape: &[SymDim],
) -> tenferro_tensor::Result<(DType, Vec<SymDim>)> {
    let dtype = eig_output_dtype(input_dtype);
    let (n, _, batch) = matrix_meta_parts("tenferro-linalg.eig_values", shape)?;
    Ok((dtype, vector_shape(n, batch)))
}

fn matrix_shape(rows: SymDim, cols: SymDim, batch: &[SymDim]) -> Vec<SymDim> {
    let mut shape = vec![rows, cols];
    shape.extend_from_slice(batch);
    shape
}

fn vector_shape(len: SymDim, batch: &[SymDim]) -> Vec<SymDim> {
    let mut shape = vec![len];
    shape.extend_from_slice(batch);
    shape
}

fn eig_output_dtype(dtype: DType) -> DType {
    match dtype {
        DType::F64 | DType::C64 => DType::C64,
        DType::F32 | DType::C32 => DType::C32,
        DType::I32 | DType::I64 | DType::Bool => DType::C64,
    }
}

fn singular_values_dtype(dtype: DType) -> DType {
    match dtype {
        DType::C64 => DType::F64,
        DType::C32 => DType::F32,
        other => other,
    }
}

fn promote_dtypes(dtypes: &[DType]) -> DType {
    dtypes
        .iter()
        .copied()
        .reduce(tenferro_tensor::validate::promote_dtype)
        .unwrap_or(DType::F64)
}

fn hash_dtype(hasher: &mut dyn Hasher, dtype: DType) {
    let tag = match dtype {
        DType::F64 => 0,
        DType::F32 => 1,
        DType::I64 => 2,
        DType::C64 => 3,
        DType::C32 => 4,
        DType::I32 => 5,
        DType::Bool => 6,
    };
    hasher.write_u8(tag);
}
