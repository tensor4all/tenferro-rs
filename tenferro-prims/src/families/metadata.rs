use tenferro_algebra::Algebra;
use tenferro_device::Result;
use tenferro_tensor::Tensor;

/// Integer/bool metadata unary operations.
///
/// This family is intentionally narrow. It exists to support metadata-heavy
/// linalg paths such as LU pivots, determinant parity, and future mask-style
/// workflows without forcing those values through the scalar or analytic
/// families.
///
/// # Examples
///
/// ```rust
/// use tenferro_prims::MetadataUnaryOp;
///
/// let op = MetadataUnaryOp::IotaStartZero;
/// assert_eq!(op, MetadataUnaryOp::IotaStartZero);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetadataUnaryOp {
    /// Generate a zero-based iota/arange tensor.
    IotaStartZero,
}

/// Integer/bool metadata binary operations.
///
/// # Examples
///
/// ```rust
/// use tenferro_prims::MetadataBinaryOp;
///
/// let op = MetadataBinaryOp::NotEqual;
/// assert_eq!(op, MetadataBinaryOp::NotEqual);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetadataBinaryOp {
    Equal,
    NotEqual,
    Add,
    Sub,
    Mul,
    BitAnd,
}

/// Integer/bool metadata ternary operations.
///
/// # Examples
///
/// ```rust
/// use tenferro_prims::MetadataTernaryOp;
///
/// let op = MetadataTernaryOp::Where;
/// assert_eq!(op, MetadataTernaryOp::Where);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetadataTernaryOp {
    Where,
}

/// Integer/bool metadata reduction operations.
///
/// # Examples
///
/// ```rust
/// use tenferro_prims::MetadataReductionOp;
///
/// let op = MetadataReductionOp::Sum;
/// assert_eq!(op, MetadataReductionOp::Sum);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetadataReductionOp {
    Sum,
    All,
    Any,
}

/// Descriptor for metadata tensor planning.
///
/// # Examples
///
/// ```rust
/// use tenferro_prims::{MetadataPrimsDescriptor, MetadataUnaryOp};
///
/// let desc = MetadataPrimsDescriptor::PointwiseUnary {
///     op: MetadataUnaryOp::IotaStartZero,
/// };
/// assert!(matches!(desc, MetadataPrimsDescriptor::PointwiseUnary { .. }));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MetadataPrimsDescriptor {
    /// Apply a metadata unary operation to one input tensor.
    PointwiseUnary {
        /// Unary operation to apply.
        op: MetadataUnaryOp,
    },
    /// Apply a metadata binary operation to two input tensors.
    PointwiseBinary {
        /// Binary operation to apply.
        op: MetadataBinaryOp,
    },
    /// Apply a metadata ternary operation to three input tensors.
    PointwiseTernary {
        /// Ternary operation to apply.
        op: MetadataTernaryOp,
    },
    /// Reduce one tensor into an output tensor over the dropped modes.
    Reduction {
        /// Input modes associated with the source tensor.
        modes_a: Vec<u32>,
        /// Output modes that remain after reduction.
        modes_c: Vec<u32>,
        /// Reduction operator to use.
        op: MetadataReductionOp,
    },
}

/// Bridge trait that binds a metadata execution context to its backend.
///
/// This mirrors the other family-context bridge traits but is reserved for
/// integer/bool metadata tensor workflows.
///
/// # Examples
///
/// ```ignore
/// use tenferro_prims::{CpuContext, TensorMetadataContextFor};
///
/// fn accepts_context<C>(_: &mut C)
/// where
///     C: TensorMetadataContextFor<tenferro_algebra::Standard<i32>>,
/// {
/// }
///
/// let mut ctx = CpuContext::new(1);
/// accepts_context(&mut ctx);
/// ```
pub trait TensorMetadataContextFor<Alg: Algebra> {
    /// Backend associated with this context for the metadata family.
    type MetadataBackend: TensorMetadataPrims<Alg, Context = Self>;
}

/// Metadata tensor planning and execution protocol.
///
/// The first tranche is intentionally small and aimed at LU/pivot-style tensor
/// metadata. It should not be used as a generic integer algebra.
///
/// # Examples
///
/// ```ignore
/// use tenferro_algebra::Standard;
/// use tenferro_prims::{CpuBackend, CpuContext, MetadataPrimsDescriptor, TensorMetadataPrims};
///
/// let mut ctx = CpuContext::new(1);
/// let desc = MetadataPrimsDescriptor::Reduction {
///     modes_a: vec![0, 1],
///     modes_c: vec![1],
///     op: tenferro_prims::MetadataReductionOp::Sum,
/// };
/// let _plan = <CpuBackend as TensorMetadataPrims<Standard<i32>>>::plan(
///     &mut ctx,
///     &desc,
///     &[&[2, 2], &[2]],
/// )
/// .unwrap();
/// ```
pub trait TensorMetadataPrims<Alg: Algebra> {
    /// Backend plan type.
    type Plan;
    /// Backend execution context.
    type Context;

    /// Plan a metadata-family operation for the given input/output shapes.
    fn plan(
        ctx: &mut Self::Context,
        desc: &MetadataPrimsDescriptor,
        shapes: &[&[usize]],
    ) -> Result<Self::Plan>;

    /// Execute a previously planned metadata-family operation.
    ///
    /// The execution contract matches the rest of tenferro prims:
    /// `output <- alpha * op(inputs) + beta * output`.
    fn execute(
        ctx: &mut Self::Context,
        plan: &Self::Plan,
        alpha: Alg::Scalar,
        inputs: &[&Tensor<Alg::Scalar>],
        beta: Alg::Scalar,
        output: &mut Tensor<Alg::Scalar>,
    ) -> Result<()>;

    /// Report whether the backend advertises support for the given descriptor.
    fn has_metadata_support(desc: MetadataPrimsDescriptor) -> bool;
}
