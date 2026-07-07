//! Backend operation capability descriptors.
//!
//! # Examples
//!
//! ```rust
//! use tenferro_core_ops::PrimitiveOpKind;
//! use tenferro_tensor::{capability_output_dtype, DType};
//!
//! assert_eq!(
//!     capability_output_dtype(PrimitiveOpKind::Compare, DType::F64),
//!     Some(DType::Bool)
//! );
//! ```

use std::fmt;

use tenferro_core_ops::{descriptor, DTypePolicy, PrimitiveOpKind};

use crate::DType;

/// Stable backend identifier used by capability descriptors.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::BackendId;
///
/// assert_eq!(BackendId::Cuda.as_str(), "cuda");
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BackendId {
    Cpu,
    Cuda,
    WebGpu,
    Other(&'static str),
}

impl BackendId {
    /// Return the stable backend name used in diagnostics and generated docs.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::BackendId;
    ///
    /// assert_eq!(BackendId::Cpu.as_str(), "cpu");
    /// ```
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::WebGpu => "webgpu",
            Self::Other(name) => name,
        }
    }
}

impl fmt::Display for BackendId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Three-valued support level for a backend capability axis.
///
/// `FallbackCopy` is intentionally distinct from `Native`: the operation is
/// accepted, but only by materializing through a default/copy path.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::SupportLevel;
///
/// assert!(SupportLevel::Native > SupportLevel::FallbackCopy);
/// assert!(SupportLevel::FallbackCopy.is_supported());
/// assert!(!SupportLevel::Unsupported.is_supported());
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SupportLevel {
    Unsupported,
    FallbackCopy,
    Native,
}

impl SupportLevel {
    /// Return whether this level represents any usable implementation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::SupportLevel;
    ///
    /// assert!(SupportLevel::Native.is_supported());
    /// assert!(!SupportLevel::Unsupported.is_supported());
    /// ```
    #[must_use]
    pub const fn is_supported(self) -> bool {
        !matches!(self, Self::Unsupported)
    }
}

/// Axis within an operation capability entry.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::CapabilityAxis;
///
/// let axis = CapabilityAxis::ReadInputs;
/// assert_eq!(format!("{axis:?}"), "ReadInputs");
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CapabilityAxis {
    OwnedResult,
    ReadInputs,
    WriteOutput,
    StridedOutput,
    Accumulation,
}

/// Query key for a backend capability lookup.
///
/// # Examples
///
/// ```rust
/// use tenferro_core_ops::PrimitiveOpKind;
/// use tenferro_tensor::{CapabilityQuery, DType};
///
/// let query = CapabilityQuery::new(PrimitiveOpKind::Add, DType::F32);
/// assert_eq!(query.dtype, DType::F32);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CapabilityQuery {
    pub op: PrimitiveOpKind,
    pub dtype: DType,
}

impl CapabilityQuery {
    /// Build a capability lookup key.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_core_ops::PrimitiveOpKind;
    /// use tenferro_tensor::{CapabilityQuery, DType};
    ///
    /// assert_eq!(
    ///     CapabilityQuery::new(PrimitiveOpKind::Mul, DType::I64).op,
    ///     PrimitiveOpKind::Mul
    /// );
    /// ```
    #[must_use]
    pub const fn new(op: PrimitiveOpKind, dtype: DType) -> Self {
        Self { op, dtype }
    }
}

/// One backend capability entry for a primitive op and input dtype.
///
/// # Examples
///
/// ```rust
/// use tenferro_core_ops::PrimitiveOpKind;
/// use tenferro_tensor::{
///     BackendId, CapabilityAxis, DType, OperationCapability, SupportLevel,
/// };
///
/// let entry = OperationCapability {
///     backend: BackendId::Cpu,
///     op: PrimitiveOpKind::Add,
///     dtype: DType::F64,
///     output_dtype: DType::F64,
///     result: SupportLevel::Native,
///     read_inputs: SupportLevel::Native,
///     write_output: SupportLevel::Native,
///     strided_output: SupportLevel::Native,
///     accumulation: SupportLevel::Unsupported,
/// };
/// assert_eq!(entry.axis(CapabilityAxis::OwnedResult), SupportLevel::Native);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OperationCapability {
    pub backend: BackendId,
    pub op: PrimitiveOpKind,
    pub dtype: DType,
    pub output_dtype: DType,
    pub result: SupportLevel,
    pub read_inputs: SupportLevel,
    pub write_output: SupportLevel,
    pub strided_output: SupportLevel,
    pub accumulation: SupportLevel,
}

impl OperationCapability {
    /// Return the support level for one capability axis.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_core_ops::PrimitiveOpKind;
    /// use tenferro_tensor::{
    ///     BackendId, CapabilityAxis, DType, OperationCapability, SupportLevel,
    /// };
    ///
    /// let entry = OperationCapability {
    ///     backend: BackendId::Cuda,
    ///     op: PrimitiveOpKind::ReduceSum,
    ///     dtype: DType::I32,
    ///     output_dtype: DType::I32,
    ///     result: SupportLevel::Native,
    ///     read_inputs: SupportLevel::FallbackCopy,
    ///     write_output: SupportLevel::Unsupported,
    ///     strided_output: SupportLevel::Unsupported,
    ///     accumulation: SupportLevel::Unsupported,
    /// };
    /// assert_eq!(entry.axis(CapabilityAxis::ReadInputs), SupportLevel::FallbackCopy);
    /// ```
    #[must_use]
    pub const fn axis(&self, axis: CapabilityAxis) -> SupportLevel {
        match axis {
            CapabilityAxis::OwnedResult => self.result,
            CapabilityAxis::ReadInputs => self.read_inputs,
            CapabilityAxis::WriteOutput => self.write_output,
            CapabilityAxis::StridedOutput => self.strided_output,
            CapabilityAxis::Accumulation => self.accumulation,
        }
    }
}

/// Backend capability query surface.
///
/// # Examples
///
/// ```rust
/// use tenferro_core_ops::PrimitiveOpKind;
/// use tenferro_tensor::{
///     BackendId, CapabilityQuery, DType, OperationCapability, SupportLevel,
///     TensorBackendCapability,
/// };
///
/// struct Backend;
///
/// const ENTRIES: &[OperationCapability] = &[OperationCapability {
///     backend: BackendId::Cpu,
///     op: PrimitiveOpKind::Add,
///     dtype: DType::F32,
///     output_dtype: DType::F32,
///     result: SupportLevel::Native,
///     read_inputs: SupportLevel::Native,
///     write_output: SupportLevel::Native,
///     strided_output: SupportLevel::Native,
///     accumulation: SupportLevel::Unsupported,
/// }];
///
/// impl TensorBackendCapability for Backend {
///     fn backend_id(&self) -> BackendId { BackendId::Cpu }
///     fn capabilities(&self) -> &'static [OperationCapability] { ENTRIES }
/// }
///
/// assert!(Backend
///     .capability(CapabilityQuery::new(PrimitiveOpKind::Add, DType::F32))
///     .is_some());
/// ```
pub trait TensorBackendCapability {
    fn backend_id(&self) -> BackendId;
    fn capabilities(&self) -> &'static [OperationCapability];

    /// Look up one operation/dtype capability for this backend.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_core_ops::PrimitiveOpKind;
    /// use tenferro_tensor::{
    ///     BackendId, CapabilityQuery, DType, OperationCapability, SupportLevel,
    ///     TensorBackendCapability,
    /// };
    ///
    /// struct Backend;
    /// const ENTRIES: &[OperationCapability] = &[OperationCapability {
    ///     backend: BackendId::Cpu,
    ///     op: PrimitiveOpKind::Mul,
    ///     dtype: DType::I64,
    ///     output_dtype: DType::I64,
    ///     result: SupportLevel::Native,
    ///     read_inputs: SupportLevel::Native,
    ///     write_output: SupportLevel::Unsupported,
    ///     strided_output: SupportLevel::Unsupported,
    ///     accumulation: SupportLevel::Unsupported,
    /// }];
    /// impl TensorBackendCapability for Backend {
    ///     fn backend_id(&self) -> BackendId { BackendId::Cpu }
    ///     fn capabilities(&self) -> &'static [OperationCapability] { ENTRIES }
    /// }
    ///
    /// let entry = Backend
    ///     .capability(CapabilityQuery::new(PrimitiveOpKind::Mul, DType::I64))
    ///     .unwrap();
    /// assert_eq!(entry.result, SupportLevel::Native);
    /// ```
    #[must_use]
    fn capability(&self, query: CapabilityQuery) -> Option<OperationCapability> {
        self.capabilities().iter().copied().find(|entry| {
            entry.backend == self.backend_id() && entry.op == query.op && entry.dtype == query.dtype
        })
    }

    /// Require support for one operation/dtype/axis, returning a structured
    /// unsupported error otherwise.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_core_ops::PrimitiveOpKind;
    /// use tenferro_tensor::{
    ///     BackendId, CapabilityAxis, CapabilityQuery, DType, Error, OperationCapability,
    ///     SupportLevel, TensorBackendCapability,
    /// };
    ///
    /// struct Backend;
    /// const ENTRIES: &[OperationCapability] = &[OperationCapability {
    ///     backend: BackendId::Cuda,
    ///     op: PrimitiveOpKind::Neg,
    ///     dtype: DType::I32,
    ///     output_dtype: DType::I32,
    ///     result: SupportLevel::Unsupported,
    ///     read_inputs: SupportLevel::Unsupported,
    ///     write_output: SupportLevel::Unsupported,
    ///     strided_output: SupportLevel::Unsupported,
    ///     accumulation: SupportLevel::Unsupported,
    /// }];
    /// impl TensorBackendCapability for Backend {
    ///     fn backend_id(&self) -> BackendId { BackendId::Cuda }
    ///     fn capabilities(&self) -> &'static [OperationCapability] { ENTRIES }
    /// }
    ///
    /// let err = Backend
    ///     .require_capability(
    ///         CapabilityQuery::new(PrimitiveOpKind::Neg, DType::I32),
    ///         CapabilityAxis::OwnedResult,
    ///     )
    ///     .unwrap_err();
    /// assert!(matches!(err, Error::UnsupportedOpDType { backend: BackendId::Cuda, .. }));
    /// ```
    fn require_capability(
        &self,
        query: CapabilityQuery,
        axis: CapabilityAxis,
    ) -> crate::Result<OperationCapability> {
        let entry = self.capability(query).ok_or_else(|| {
            crate::Error::unsupported_op_dtype(
                descriptor(query.op).name,
                query.dtype,
                self.backend_id(),
            )
        })?;
        if entry.axis(axis).is_supported() {
            Ok(entry)
        } else {
            Err(crate::Error::unsupported_op_dtype(
                descriptor(query.op).name,
                query.dtype,
                self.backend_id(),
            ))
        }
    }
}

/// Return the output dtype allowed by the core op catalog policy for a unary
/// dtype representative.
///
/// `None` means the catalog policy does not admit the queried dtype. Backend
/// descriptors add implementation support on top of this semantic policy.
///
/// # Examples
///
/// ```rust
/// use tenferro_core_ops::PrimitiveOpKind;
/// use tenferro_tensor::{capability_output_dtype, DType};
///
/// assert_eq!(
///     capability_output_dtype(PrimitiveOpKind::Abs, DType::C64),
///     Some(DType::F64)
/// );
/// assert_eq!(capability_output_dtype(PrimitiveOpKind::Pow, DType::I32), None);
/// ```
#[must_use]
pub fn capability_output_dtype(op: PrimitiveOpKind, dtype: DType) -> Option<DType> {
    let policy = descriptor(op).dtype_policy;
    match policy {
        DTypePolicy::SameAny => Some(dtype),
        DTypePolicy::SameNumeric => numeric_dtype(dtype).then_some(dtype),
        DTypePolicy::SameFloat => float_dtype(dtype).then_some(dtype),
        DTypePolicy::AbsToReal => match dtype {
            DType::F32 => Some(DType::F32),
            DType::F64 => Some(DType::F64),
            DType::C32 => Some(DType::F32),
            DType::C64 => Some(DType::F64),
            DType::I32 | DType::I64 | DType::Bool => None,
        },
        DTypePolicy::SameFloatOrComplex => float_or_complex_dtype(dtype).then_some(dtype),
        DTypePolicy::CompareToBool => comparable_dtype(dtype).then_some(DType::Bool),
        DTypePolicy::BoolSelect => Some(dtype),
        DTypePolicy::Convert | DTypePolicy::Shape | DTypePolicy::Constant => Some(dtype),
    }
}

const fn numeric_dtype(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::F32 | DType::F64 | DType::I32 | DType::I64 | DType::C32 | DType::C64
    )
}

const fn float_dtype(dtype: DType) -> bool {
    matches!(dtype, DType::F32 | DType::F64)
}

const fn float_or_complex_dtype(dtype: DType) -> bool {
    matches!(dtype, DType::F32 | DType::F64 | DType::C32 | DType::C64)
}

const fn comparable_dtype(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::F32 | DType::F64 | DType::I32 | DType::I64 | DType::Bool
    )
}
