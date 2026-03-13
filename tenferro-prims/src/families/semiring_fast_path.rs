use tenferro_algebra::Semiring;
use tenferro_device::Result;
use tenferro_tensor::Tensor;

/// Semiring-valid optional binary fast-path operations.
///
/// # Examples
///
/// ```
/// use tenferro_prims::SemiringBinaryOp;
///
/// let op = SemiringBinaryOp::Mul;
/// assert_eq!(op, SemiringBinaryOp::Mul);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SemiringBinaryOp {
    /// Elementwise semiring addition.
    Add,
    /// Elementwise semiring multiplication.
    Mul,
}

/// Descriptor for optional semiring fast paths.
///
/// # Examples
///
/// ```
/// use tenferro_prims::{SemiringBinaryOp, SemiringFastPathDescriptor};
///
/// let desc = SemiringFastPathDescriptor::ElementwiseBinary {
///     op: SemiringBinaryOp::Mul,
/// };
/// assert!(matches!(desc, SemiringFastPathDescriptor::ElementwiseBinary { .. }));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SemiringFastPathDescriptor {
    /// Optional contraction fast path.
    Contract {
        /// Mode labels for input A.
        modes_a: Vec<u32>,
        /// Mode labels for input B.
        modes_b: Vec<u32>,
        /// Mode labels for output C.
        modes_c: Vec<u32>,
    },
    /// Optional elementwise semiring binary fast path.
    ElementwiseBinary {
        /// The semiring binary operation.
        op: SemiringBinaryOp,
    },
}

/// Optional semiring performance paths.
///
/// # Examples
///
/// ```ignore
/// use tenferro_algebra::Standard;
/// use tenferro_prims::{
///     CpuBackend, SemiringBinaryOp, SemiringFastPathDescriptor, TensorSemiringFastPath,
/// };
///
/// let supported =
///     <CpuBackend as TensorSemiringFastPath<Standard<f64>>>::has_fast_path(SemiringFastPathDescriptor::ElementwiseBinary {
///         op: SemiringBinaryOp::Mul,
///     });
/// assert!(supported);
/// ```
pub trait TensorSemiringFastPath<Alg: Semiring> {
    /// Backend-specific plan type.
    type Plan;

    /// Backend-specific execution context.
    type Context;

    /// Plan an optional semiring fast path.
    fn plan(
        ctx: &mut Self::Context,
        desc: &SemiringFastPathDescriptor,
        shapes: &[&[usize]],
    ) -> Result<Self::Plan>;

    /// Execute an optional semiring fast path.
    fn execute(
        ctx: &mut Self::Context,
        plan: &Self::Plan,
        alpha: Alg::Scalar,
        inputs: &[&Tensor<Alg::Scalar>],
        beta: Alg::Scalar,
        output: &mut Tensor<Alg::Scalar>,
    ) -> Result<()>;

    /// Query whether the optional path is available.
    fn has_fast_path(desc: SemiringFastPathDescriptor) -> bool;
}
