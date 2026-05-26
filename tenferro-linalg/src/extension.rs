use std::any::Any;
use std::hash::Hasher;
use std::sync::Arc;

use tenferro_ops::SymDim;
use tenferro_runtime::extension::{
    ExtensionExecutionContext, ExtensionExecutor, ExtensionOpTrait, ExtensionRuntime,
    ExtensionRuntimeRegistryError,
};
use tenferro_tensor::{DType, Tensor, TensorBackend};

pub const LINALG_EXTENSION_FAMILY_ID: &str = "tenferro-linalg.linalg.v1";

#[derive(Clone, Copy, Debug, PartialEq)]
#[doc(hidden)]
pub enum LinalgOp {
    Cholesky,
    Lu,
    FullPivLu,
    FullPivLuSolve {
        transpose_a: bool,
    },
    Svd {
        eps: f64,
    },
    Qr,
    Eigh {
        eps: f64,
    },
    Eig {
        input_dtype: DType,
    },
    Solve {
        transpose_a: bool,
    },
    TriangularSolve {
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
    },
}

impl LinalgOp {
    pub fn output_count(self) -> usize {
        match self {
            Self::Cholesky
            | Self::FullPivLuSolve { .. }
            | Self::Solve { .. }
            | Self::TriangularSolve { .. } => 1,
            Self::Svd { .. } => 3,
            Self::Qr | Self::Eigh { .. } | Self::Eig { .. } => 2,
            Self::Lu => 4,
            Self::FullPivLu => 5,
        }
    }

    fn input_count(self) -> usize {
        match self {
            Self::FullPivLuSolve { .. } | Self::Solve { .. } | Self::TriangularSolve { .. } => 2,
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
            Self::Solve { .. } => 8,
            Self::TriangularSolve { .. } => 9,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
#[doc(hidden)]
pub struct LinalgExtensionOp {
    op: LinalgOp,
}

impl LinalgExtensionOp {
    pub fn new(op: LinalgOp) -> Self {
        Self { op }
    }

    pub fn op(&self) -> LinalgOp {
        self.op
    }
}

impl ExtensionOpTrait for LinalgExtensionOp {
    fn family_id(&self) -> &'static str {
        LINALG_EXTENSION_FAMILY_ID
    }

    fn payload_hash(&self, hasher: &mut dyn Hasher) {
        hasher.write_u8(self.op.tag());
        match self.op {
            LinalgOp::Svd { eps } | LinalgOp::Eigh { eps } => hasher.write_u64(eps.to_bits()),
            LinalgOp::Eig { input_dtype } => hash_dtype(hasher, input_dtype),
            LinalgOp::Solve { transpose_a } | LinalgOp::FullPivLuSolve { transpose_a } => {
                hasher.write_u8(u8::from(transpose_a));
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
            LinalgOp::Cholesky | LinalgOp::Lu | LinalgOp::FullPivLu | LinalgOp::Qr => {}
        }
    }

    fn payload_eq(&self, other: &dyn ExtensionOpTrait) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|that| self == that)
    }

    fn clone_arc(&self) -> Arc<dyn ExtensionOpTrait> {
        Arc::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn n_inputs(&self) -> usize {
        self.op.input_count()
    }

    fn n_outputs(&self) -> usize {
        self.op.output_count()
    }

    fn infer_output_meta(
        &self,
        input_dtypes: &[DType],
        input_shapes: &[&[SymDim]],
    ) -> Vec<(DType, Vec<SymDim>)> {
        assert_eq!(input_dtypes.len(), self.n_inputs());
        assert_eq!(input_shapes.len(), self.n_inputs());
        match self.op {
            LinalgOp::Cholesky
            | LinalgOp::FullPivLuSolve { .. }
            | LinalgOp::Solve { .. }
            | LinalgOp::TriangularSolve { .. } => {
                let output_shape = if self.n_inputs() == 1 {
                    input_shapes[0].to_vec()
                } else {
                    input_shapes[1].to_vec()
                };
                vec![(promote_dtypes(input_dtypes), output_shape)]
            }
            LinalgOp::Lu => lu_meta(input_dtypes[0], input_shapes[0]),
            LinalgOp::FullPivLu => full_piv_lu_meta(input_dtypes[0], input_shapes[0]),
            LinalgOp::Svd { .. } => svd_meta(input_dtypes[0], input_shapes[0]),
            LinalgOp::Qr => qr_meta(input_dtypes[0], input_shapes[0]),
            LinalgOp::Eigh { .. } => eigh_meta(input_dtypes[0], input_shapes[0]),
            LinalgOp::Eig { input_dtype } => eig_meta(input_dtype, input_shapes[0]),
        }
    }

    fn eager_execute(&self, inputs: &[&Tensor]) -> tenferro_tensor::Result<Vec<Tensor>> {
        let mut backend = tenferro_tensor::cpu::CpuBackend::new();
        execute_linalg(self.op, inputs, &mut backend)
    }
}

#[derive(Debug, Default)]
pub(crate) struct LinalgRuntime;

impl<B: TensorBackend + 'static> ExtensionRuntime<B> for LinalgRuntime {
    fn family_id(&self) -> &'static str {
        LINALG_EXTENSION_FAMILY_ID
    }

    fn execute(
        &self,
        op: &dyn ExtensionOpTrait,
        inputs: &[&Tensor],
        ctx: &mut ExtensionExecutionContext<'_, B>,
    ) -> tenferro_tensor::Result<Vec<Tensor>> {
        let op = op
            .as_any()
            .downcast_ref::<LinalgExtensionOp>()
            .ok_or_else(|| tenferro_tensor::Error::InvalidConfig {
                op: "linalg_extension",
                message: "payload type mismatch for tenferro-linalg.linalg.v1".into(),
            })?;
        execute_linalg(op.op(), inputs, ctx.backend_mut())
    }
}

pub fn register_runtime<B: TensorBackend + 'static>(
    executor: &mut ExtensionExecutor<B>,
) -> Result<(), ExtensionRuntimeRegistryError> {
    executor.registry_mut().register(Arc::new(LinalgRuntime))
}

fn execute_linalg<B: TensorBackend>(
    op: LinalgOp,
    inputs: &[&Tensor],
    backend: &mut B,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    match op {
        LinalgOp::Cholesky => Ok(vec![backend.cholesky(inputs[0])?]),
        LinalgOp::Lu => backend.lu(inputs[0]),
        LinalgOp::FullPivLu => backend.full_piv_lu(inputs[0]),
        LinalgOp::FullPivLuSolve { transpose_a } => Ok(vec![backend.full_piv_lu_solve(
            inputs[0],
            inputs[1],
            transpose_a,
        )?]),
        LinalgOp::Svd { .. } => backend.svd(inputs[0]),
        LinalgOp::Qr => backend.qr(inputs[0]),
        LinalgOp::Eigh { .. } => backend.eigh(inputs[0]),
        LinalgOp::Eig { .. } => backend.eig(inputs[0]),
        LinalgOp::Solve { transpose_a } => {
            let a_transposed;
            let a = if transpose_a {
                let perm = matrix_transpose_perm(inputs[0])?;
                a_transposed = backend.transpose(inputs[0], &perm)?;
                &a_transposed
            } else {
                inputs[0]
            };
            Ok(vec![backend.solve(a, inputs[1])?])
        }
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

fn matrix_transpose_perm(input: &Tensor) -> tenferro_tensor::Result<Vec<usize>> {
    let rank = input.shape().len();
    if rank < 2 {
        return Err(tenferro_tensor::Error::InvalidConfig {
            op: "solve",
            message: "matrix operand rank must be at least 2".into(),
        });
    }
    let mut perm: Vec<usize> = (0..rank).collect();
    perm.swap(0, 1);
    Ok(perm)
}

fn lu_meta(dtype: DType, shape: &[SymDim]) -> Vec<(DType, Vec<SymDim>)> {
    let m = shape[0].clone();
    let n = shape[1].clone();
    let k = m.clone().min(n.clone());
    let batch = &shape[2..];
    vec![
        (dtype, matrix_shape(m.clone(), m, batch)),
        (dtype, matrix_shape(shape[0].clone(), k.clone(), batch)),
        (dtype, matrix_shape(k, n, batch)),
        (dtype, batch.to_vec()),
    ]
}

fn full_piv_lu_meta(dtype: DType, shape: &[SymDim]) -> Vec<(DType, Vec<SymDim>)> {
    let n = shape[0].clone();
    let batch = &shape[2..];
    vec![
        (dtype, matrix_shape(n.clone(), n.clone(), batch)),
        (dtype, matrix_shape(n.clone(), n.clone(), batch)),
        (dtype, matrix_shape(n.clone(), n.clone(), batch)),
        (dtype, matrix_shape(n.clone(), n, batch)),
        (dtype, batch.to_vec()),
    ]
}

fn svd_meta(dtype: DType, shape: &[SymDim]) -> Vec<(DType, Vec<SymDim>)> {
    let m = shape[0].clone();
    let n = shape[1].clone();
    let k = m.clone().min(n.clone());
    let batch = &shape[2..];
    vec![
        (dtype, matrix_shape(m, k.clone(), batch)),
        (dtype, vector_shape(k.clone(), batch)),
        (dtype, matrix_shape(k, n, batch)),
    ]
}

fn qr_meta(dtype: DType, shape: &[SymDim]) -> Vec<(DType, Vec<SymDim>)> {
    let m = shape[0].clone();
    let n = shape[1].clone();
    let k = m.clone().min(n.clone());
    let batch = &shape[2..];
    vec![
        (dtype, matrix_shape(m, k.clone(), batch)),
        (dtype, matrix_shape(k, n, batch)),
    ]
}

fn eigh_meta(dtype: DType, shape: &[SymDim]) -> Vec<(DType, Vec<SymDim>)> {
    let n = shape[0].clone();
    let batch = &shape[2..];
    vec![
        (dtype, vector_shape(n.clone(), batch)),
        (dtype, matrix_shape(n.clone(), n, batch)),
    ]
}

fn eig_meta(input_dtype: DType, shape: &[SymDim]) -> Vec<(DType, Vec<SymDim>)> {
    let dtype = eig_output_dtype(input_dtype);
    let n = shape[0].clone();
    let batch = &shape[2..];
    vec![
        (dtype, vector_shape(n.clone(), batch)),
        (dtype, matrix_shape(n.clone(), n, batch)),
    ]
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

fn promote_dtypes(dtypes: &[DType]) -> DType {
    dtypes
        .iter()
        .copied()
        .reduce(promote_dtype)
        .unwrap_or(DType::F64)
}

fn promote_dtype(lhs: DType, rhs: DType) -> DType {
    use DType::*;
    if lhs == rhs {
        return lhs;
    }
    let (a, b) = if promotion_rank(lhs) <= promotion_rank(rhs) {
        (lhs, rhs)
    } else {
        (rhs, lhs)
    };
    match (a, b) {
        (Bool, other) => other,
        (I32, I64) => I64,
        (I32 | I64, F32 | F64) => F64,
        (I32 | I64, C32 | C64) => C64,
        (F32, F64) => F64,
        (F32, C32) => C32,
        (F32, C64) => C64,
        (F64, C32) => C64,
        (F64, C64) => C64,
        (C32, C64) => C64,
        _ => unreachable!("promote_dtype: unhandled pair {:?} {:?}", lhs, rhs),
    }
}

fn promotion_rank(dtype: DType) -> u8 {
    match dtype {
        DType::Bool => 0,
        DType::I32 => 1,
        DType::I64 => 2,
        DType::F32 => 3,
        DType::F64 => 4,
        DType::C32 => 5,
        DType::C64 => 6,
    }
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
