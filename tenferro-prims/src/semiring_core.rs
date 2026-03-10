use tenferro_algebra::Semiring;
use tenferro_device::Result;
use tenferro_tensor::Tensor;

use crate::{PrimDescriptor, TensorPrims};

/// Descriptor for semiring-core execution operations.
///
/// This is the minimal protocol family that `tenferro-einsum` may depend on.
///
/// # Examples
///
/// ```
/// use tenferro_prims::SemiringCoreDescriptor;
///
/// let desc = SemiringCoreDescriptor::MakeContiguous;
/// assert!(matches!(desc, SemiringCoreDescriptor::MakeContiguous));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SemiringCoreDescriptor {
    /// Batched semiring GEMM.
    BatchedGemm {
        /// Batch dimension sizes.
        batch_dims: Vec<usize>,
        /// Rows in A / C.
        m: usize,
        /// Columns in B / C.
        n: usize,
        /// Contracted dimension.
        k: usize,
    },
    /// Reduction using semiring addition.
    ReduceAdd {
        /// Mode labels for input tensor A.
        modes_a: Vec<u32>,
        /// Mode labels for output tensor C.
        modes_c: Vec<u32>,
    },
    /// Diagonal contraction.
    Trace {
        /// Mode labels for input tensor A.
        modes_a: Vec<u32>,
        /// Mode labels for output tensor C.
        modes_c: Vec<u32>,
        /// Paired diagonal modes.
        paired: Vec<(u32, u32)>,
    },
    /// Diagonal scatter-add for trace adjoints.
    AntiTrace {
        /// Mode labels for input tensor A.
        modes_a: Vec<u32>,
        /// Mode labels for output tensor C.
        modes_c: Vec<u32>,
        /// Paired diagonal modes.
        paired: Vec<(u32, u32)>,
    },
    /// Diagonal scatter/write for diag adjoints.
    AntiDiag {
        /// Mode labels for input tensor A.
        modes_a: Vec<u32>,
        /// Mode labels for output tensor C.
        modes_c: Vec<u32>,
        /// Paired diagonal modes.
        paired: Vec<(u32, u32)>,
    },
    /// Materialize a contiguous output tensor.
    MakeContiguous,
}

impl SemiringCoreDescriptor {
    pub(crate) fn to_legacy(&self) -> PrimDescriptor {
        match self {
            Self::BatchedGemm {
                batch_dims,
                m,
                n,
                k,
            } => PrimDescriptor::BatchedGemm {
                batch_dims: batch_dims.clone(),
                m: *m,
                n: *n,
                k: *k,
            },
            Self::ReduceAdd { modes_a, modes_c } => PrimDescriptor::Reduce {
                modes_a: modes_a.clone(),
                modes_c: modes_c.clone(),
                op: crate::ReduceOp::Sum,
            },
            Self::Trace {
                modes_a,
                modes_c,
                paired,
            } => PrimDescriptor::Trace {
                modes_a: modes_a.clone(),
                modes_c: modes_c.clone(),
                paired: paired.clone(),
            },
            Self::AntiTrace {
                modes_a,
                modes_c,
                paired,
            } => PrimDescriptor::AntiTrace {
                modes_a: modes_a.clone(),
                modes_c: modes_c.clone(),
                paired: paired.clone(),
            },
            Self::AntiDiag {
                modes_a,
                modes_c,
                paired,
            } => PrimDescriptor::AntiDiag {
                modes_a: modes_a.clone(),
                modes_c: modes_c.clone(),
                paired: paired.clone(),
            },
            Self::MakeContiguous => PrimDescriptor::MakeContiguous,
        }
    }
}

/// Minimal semiring execution protocol.
///
/// # Examples
///
/// ```ignore
/// use tenferro_algebra::Standard;
/// use tenferro_prims::{CpuBackend, CpuContext, SemiringCoreDescriptor, TensorSemiringCore};
///
/// let mut ctx = CpuContext::new(1);
/// let desc = SemiringCoreDescriptor::MakeContiguous;
/// let _plan = <CpuBackend as TensorSemiringCore<Standard<f64>>>::plan(
///     &mut ctx,
///     &desc,
///     &[&[2, 2], &[2, 2]],
/// )
/// .unwrap();
/// ```
pub trait TensorSemiringCore<Alg: Semiring> {
    /// Backend-specific plan type.
    type Plan;

    /// Backend-specific execution context.
    type Context;

    /// Plan a semiring-core operation.
    fn plan(
        ctx: &mut Self::Context,
        desc: &SemiringCoreDescriptor,
        shapes: &[&[usize]],
    ) -> Result<Self::Plan>;

    /// Execute a semiring-core operation.
    fn execute(
        ctx: &mut Self::Context,
        plan: &Self::Plan,
        alpha: Alg::Scalar,
        inputs: &[&Tensor<Alg::Scalar>],
        beta: Alg::Scalar,
        output: &mut Tensor<Alg::Scalar>,
    ) -> Result<()>;
}

impl<Alg, B> TensorSemiringCore<Alg> for B
where
    Alg: Semiring,
    B: TensorPrims<Alg>,
{
    type Plan = <B as TensorPrims<Alg>>::Plan;
    type Context = <B as TensorPrims<Alg>>::Context;

    fn plan(
        ctx: &mut Self::Context,
        desc: &SemiringCoreDescriptor,
        shapes: &[&[usize]],
    ) -> Result<Self::Plan> {
        <B as TensorPrims<Alg>>::plan(ctx, &desc.to_legacy(), shapes)
    }

    fn execute(
        ctx: &mut Self::Context,
        plan: &Self::Plan,
        alpha: Alg::Scalar,
        inputs: &[&Tensor<Alg::Scalar>],
        beta: Alg::Scalar,
        output: &mut Tensor<Alg::Scalar>,
    ) -> Result<()> {
        <B as TensorPrims<Alg>>::execute(ctx, plan, alpha, inputs, beta, output)
    }
}
