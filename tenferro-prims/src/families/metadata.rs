use tenferro_device::Result;
use tenferro_tensor::Tensor;

/// Metadata tensor dtypes.
///
/// Metadata tensors currently use logical `I32` and `Bool` dtypes. `Bool` is
/// backed by `u8` storage for now; that storage detail is provisional and may
/// change once native bool tensor support exists.
///
/// # Examples
///
/// ```rust
/// use tenferro_prims::MetadataDType;
///
/// assert_eq!(MetadataDType::I32, MetadataDType::I32);
/// assert_eq!(MetadataDType::Bool, MetadataDType::Bool);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetadataDType {
    /// Signed 32-bit integer metadata tensors.
    I32,
    /// Logical bool/mask metadata tensors.
    Bool,
}

/// Constant payload for metadata tensor generation.
///
/// # Examples
///
/// ```rust
/// use tenferro_prims::{MetadataConstantValue, MetadataDType};
///
/// let int_value = MetadataConstantValue::I32(7);
/// let bool_value = MetadataConstantValue::Bool(true);
/// assert_eq!(int_value.dtype(), MetadataDType::I32);
/// assert_eq!(bool_value.dtype(), MetadataDType::Bool);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetadataConstantValue {
    /// Integer metadata payload.
    I32(i32),
    /// Logical bool payload.
    Bool(bool),
}

impl MetadataConstantValue {
    /// Return the logical dtype carried by this constant payload.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_prims::{MetadataConstantValue, MetadataDType};
    ///
    /// assert_eq!(MetadataConstantValue::I32(-3).dtype(), MetadataDType::I32);
    /// assert_eq!(MetadataConstantValue::Bool(false).dtype(), MetadataDType::Bool);
    /// ```
    pub const fn dtype(self) -> MetadataDType {
        match self {
            Self::I32(_) => MetadataDType::I32,
            Self::Bool(_) => MetadataDType::Bool,
        }
    }
}

/// Metadata tensor generation operations.
///
/// Generation is intentionally separate from pointwise metadata ops so the
/// contract can describe `iota`-style and constant metadata tensors without
/// requiring a dummy input tensor.
///
/// # Examples
///
/// ```rust
/// use tenferro_prims::{MetadataConstantValue, MetadataGenerateOp};
///
/// let op = MetadataGenerateOp::IotaStartZero;
/// assert!(matches!(op, MetadataGenerateOp::IotaStartZero));
/// let constant = MetadataGenerateOp::Constant(MetadataConstantValue::Bool(true));
/// assert!(matches!(
///     constant,
///     MetadataGenerateOp::Constant(MetadataConstantValue::Bool(true))
/// ));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetadataGenerateOp {
    /// Generate a zero-based iota/arange tensor.
    IotaStartZero,
    /// Generate a tensor filled with a constant payload.
    Constant(MetadataConstantValue),
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
    /// Elementwise equality.
    Equal,
    /// Elementwise inequality.
    NotEqual,
    /// Metadata addition.
    Add,
    /// Metadata subtraction.
    Sub,
    /// Metadata multiplication.
    Mul,
    /// Elementwise bitwise-and.
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
    /// Select from the second or third input using the first input as the mask.
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
    /// Sum metadata values into the output tensor.
    Sum,
    /// Logical all-reduction over bool-like metadata tensors.
    All,
    /// Logical any-reduction over bool-like metadata tensors.
    Any,
}

/// Erased immutable metadata tensor reference.
///
/// The contract is intentionally narrow and only admits integer metadata and
/// logical bool metadata. The bool handle is currently backed by `u8`
/// storage, but that is a provisional implementation detail.
///
/// # Examples
///
/// ```ignore
/// use tenferro_device::LogicalMemorySpace;
/// use tenferro_prims::{MetadataDType, MetadataTensorRef};
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let tensor = Tensor::<i32>::zeros(
///     &[2, 2],
///     LogicalMemorySpace::MainMemory,
///     MemoryOrder::ColumnMajor,
/// );
/// let metadata = MetadataTensorRef::I32(&tensor);
/// assert_eq!(metadata.dtype(), MetadataDType::I32);
/// ```
#[derive(Debug, Clone, Copy)]
pub enum MetadataTensorRef<'a> {
    /// A metadata tensor stored as `i32`.
    I32(&'a Tensor<i32>),
    /// A logical bool/mask metadata tensor backed by `u8` for now.
    Bool(&'a Tensor<u8>),
}

impl<'a> MetadataTensorRef<'a> {
    /// Return the logical dtype carried by this metadata tensor reference.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::LogicalMemorySpace;
    /// use tenferro_prims::{MetadataDType, MetadataTensorRef};
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let tensor = Tensor::<u8>::zeros(
    ///     &[1],
    ///     LogicalMemorySpace::MainMemory,
    ///     MemoryOrder::ColumnMajor,
    /// );
    /// let metadata = MetadataTensorRef::Bool(&tensor);
    /// assert_eq!(metadata.dtype(), MetadataDType::Bool);
    /// ```
    pub const fn dtype(&self) -> MetadataDType {
        match self {
            Self::I32(_) => MetadataDType::I32,
            Self::Bool(_) => MetadataDType::Bool,
        }
    }
}

/// Erased mutable metadata tensor reference.
///
/// # Examples
///
/// ```ignore
/// use tenferro_device::LogicalMemorySpace;
/// use tenferro_prims::{MetadataDType, MetadataTensorMut};
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let mut tensor = Tensor::<u8>::zeros(
///     &[2, 2],
///     LogicalMemorySpace::MainMemory,
///     MemoryOrder::ColumnMajor,
/// );
/// let metadata = MetadataTensorMut::Bool(&mut tensor);
/// assert_eq!(metadata.dtype(), MetadataDType::Bool);
/// ```
#[derive(Debug)]
pub enum MetadataTensorMut<'a> {
    /// A metadata tensor stored as `i32`.
    I32(&'a mut Tensor<i32>),
    /// A logical bool/mask metadata tensor backed by `u8` for now.
    Bool(&'a mut Tensor<u8>),
}

impl<'a> MetadataTensorMut<'a> {
    /// Return the logical dtype carried by this metadata tensor handle.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::LogicalMemorySpace;
    /// use tenferro_prims::{MetadataDType, MetadataTensorMut};
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let mut tensor = Tensor::<i32>::zeros(
    ///     &[1],
    ///     LogicalMemorySpace::MainMemory,
    ///     MemoryOrder::ColumnMajor,
    /// );
    /// let metadata = MetadataTensorMut::I32(&mut tensor);
    /// assert_eq!(metadata.dtype(), MetadataDType::I32);
    /// ```
    pub const fn dtype(&self) -> MetadataDType {
        match self {
            Self::I32(_) => MetadataDType::I32,
            Self::Bool(_) => MetadataDType::Bool,
        }
    }
}

/// Descriptor for metadata tensor planning.
///
/// # Examples
///
/// ```rust
/// use tenferro_prims::{
///     MetadataConstantValue, MetadataDType, MetadataGenerateOp, MetadataPrimsDescriptor,
/// };
///
/// let desc = MetadataPrimsDescriptor::Generate {
///     op: MetadataGenerateOp::Constant(MetadataConstantValue::I32(3)),
///     output_dtype: MetadataDType::I32,
/// };
/// assert!(matches!(desc, MetadataPrimsDescriptor::Generate { .. }));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MetadataPrimsDescriptor {
    /// Generate a metadata tensor without consuming an input tensor.
    Generate {
        /// Generation operation to apply.
        op: MetadataGenerateOp,
        /// Logical dtype of the generated tensor.
        output_dtype: MetadataDType,
    },
    /// Apply a metadata binary operation to two input tensors.
    Binary {
        /// Binary operation to apply.
        op: MetadataBinaryOp,
        /// Logical dtype of the left-hand side operand.
        lhs_dtype: MetadataDType,
        /// Logical dtype of the right-hand side operand.
        rhs_dtype: MetadataDType,
        /// Logical dtype of the output tensor.
        output_dtype: MetadataDType,
    },
    /// Apply a metadata ternary operation to three input tensors.
    Ternary {
        /// Ternary operation to apply.
        op: MetadataTernaryOp,
        /// Logical dtype of the condition/mask input.
        cond_dtype: MetadataDType,
        /// Logical dtype of the first data input.
        lhs_dtype: MetadataDType,
        /// Logical dtype of the second data input.
        rhs_dtype: MetadataDType,
        /// Logical dtype of the output tensor.
        output_dtype: MetadataDType,
    },
    /// Reduce one tensor into an output tensor over the dropped modes.
    Reduction {
        /// Input modes associated with the source tensor.
        modes_a: Vec<u32>,
        /// Output modes that remain after reduction.
        modes_c: Vec<u32>,
        /// Logical dtype of the input tensor.
        input_dtype: MetadataDType,
        /// Logical dtype of the output tensor.
        output_dtype: MetadataDType,
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
/// use tenferro_prims::TensorMetadataContextFor;
///
/// fn accepts_context<C>(_: &mut C)
/// where
///     C: TensorMetadataContextFor,
/// {
/// }
///
/// // A backend context can satisfy this trait once the metadata family is wired.
/// ```
pub trait TensorMetadataContextFor {
    /// Backend associated with this context for the metadata family.
    type MetadataBackend: TensorMetadataPrims<Context = Self>;
}

/// Metadata tensor planning and execution protocol.
///
/// Metadata execution is overwrite-based and uses erased integer/bool metadata
/// tensor handles instead of scalar-family `alpha` / `beta` scaling.
///
/// # Examples
///
/// ```ignore
/// use tenferro_prims::{
///     MetadataGenerateOp, MetadataPrimsDescriptor, MetadataTensorMut,
///     MetadataTensorRef, TensorMetadataPrims,
/// };
///
/// fn accepts_family<F: TensorMetadataPrims>(_: &F) {}
///
/// let _ = MetadataPrimsDescriptor::Generate {
///     op: MetadataGenerateOp::IotaStartZero,
///     output_dtype: tenferro_prims::MetadataDType::I32,
/// };
/// ```
pub trait TensorMetadataPrims {
    /// Backend plan type.
    type Plan;
    /// Backend execution context.
    type Context;

    /// Plan a metadata-family operation for the given input and output tensor
    /// handles.
    fn plan(
        ctx: &mut Self::Context,
        desc: &MetadataPrimsDescriptor,
        inputs: &[MetadataTensorRef<'_>],
        output: MetadataTensorMut<'_>,
    ) -> Result<Self::Plan>;

    /// Execute a previously planned metadata-family operation in overwrite
    /// mode.
    fn execute(
        ctx: &mut Self::Context,
        plan: &Self::Plan,
        inputs: &[MetadataTensorRef<'_>],
        output: MetadataTensorMut<'_>,
    ) -> Result<()>;

    /// Report whether the backend advertises support for the given descriptor.
    fn has_metadata_support(desc: MetadataPrimsDescriptor) -> bool;
}
