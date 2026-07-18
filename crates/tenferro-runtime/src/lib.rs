//! Traced graph runtime and extension dispatch infrastructure for tenferro.
//!
//! This crate owns graph construction, lowering to execution IR, graph
//! execution, and backend-parametric extension runtime dispatch. Standard
//! operations are lowered through the runtime's internal operation vocabulary;
//! tensor storage and backend kernels live in `tenferro-tensor`.
//!
//! Use this crate directly when you want concrete tensor helpers or reusable
//! traced graph execution without opting into autodiff. Start with
//! [`TypedTensor`] when the scalar type is fixed in Rust, [`Tensor`] when dtype
//! is selected at runtime, and [`TracedTensor`] plus [`GraphCompiler`] and
//! [`GraphExecutor`] when the same expression should be compiled once and run
//! repeatedly. Operation-family crates such as `tenferro-einsum`,
//! `tenferro-linalg`, and `tenferro-fft` register extension runtimes with
//! [`GraphExecutor`] when compiled execution reaches those operations.
//!
//! User-facing guides live at
//! <https://tensor4all.org/tenferro-rs/guides/choosing-an-api.html> and
//! <https://tensor4all.org/tenferro-rs/guides/execution-models.html>.
//!
//! # Examples
//!
//! ```rust
//! use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};
//! use tenferro_cpu::CpuBackend;
//!
//! let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
//! let y = (&x + &x).unwrap();
//! let mut compiler = GraphCompiler::new();
//! let program = compiler.compile(&y).unwrap();
//! let out = GraphExecutor::new(CpuBackend::default()).run(&program).unwrap();
//! assert_eq!(out.as_slice::<f64>().unwrap(), &[2.0, 4.0]);
//! ```

#[doc(hidden)]
pub mod ad_support;
mod checkpoint;
mod compiler;
pub mod error;
mod exec;
pub mod extension;
pub mod extension_cache;
pub mod extension_runtime;
pub mod graph;
mod metadata;
#[doc(hidden)]
pub mod scalar_semantics;
mod segment;
mod shape_constraint;
mod shape_infer;
mod shape_packing;
pub mod sym_dim;
mod tensor;
pub mod traced;
mod typed_tensor;

pub use compiler::{CompilerOptions, OptimizerConfig};
pub use error::{ContextId, Error, ErrorPhase, Result, ShapeConstraintEvalError};
pub use extension_cache::{
    ExtensionCacheKey, ExtensionCacheLimits, ExtensionCacheSelector, ExtensionCacheStore,
};
pub use extension_runtime::{
    ExtensionExecutionContext, ExtensionExecutor, ExtensionRegistry, ExtensionRuntime,
    ExtensionRuntimeRegistryError, HostReferenceRuntime,
};
pub use graph::{
    GraphCompiler, GraphCompilerCacheStats, GraphExecutor, GraphExecutorCacheStats,
    GraphInstructionView, GraphOpView, GraphProgram, GraphProgramInput,
    GraphProgramLoweringShapeError, GraphProgramLoweringView,
};
#[doc(hidden)]
pub use shape_constraint::ShapeGuard;
pub use shape_packing::TracedSliceBuilder;
pub use sym_dim::SymDim;
pub use tenferro_ops::ShapeRelation;
pub use tenferro_tensor::{
    CacheStats, CompareDir, DType, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig,
    SliceConfig, Tensor, TensorBackend, TensorRead, TensorScalar, TensorValue, TensorView,
    TypedTensor, TypedTensorView,
};

/// Backend-explicit concrete tensor operations.
///
/// `Tensor` is owned by `tenferro-tensor`, so `tenferro-runtime` exposes these
/// operations as a crate-root extension trait rather than as inherent methods.
///
/// # Public API rationale
///
/// This trait is intentionally public: it is the supported non-AD concrete
/// tensor operation surface for downstream users who want to run operations on
/// an explicit backend. The old public module/free-function surface was
/// removed; the private `tensor` module now contains implementation helpers
/// only and must not be treated as a compatibility API.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::CpuBackend;
/// use tenferro_runtime::{Tensor, TensorOpsExt};
///
/// let mut backend = CpuBackend::new();
/// let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64; 4]).unwrap();
/// let b = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64; 4]).unwrap();
/// let c = a.matmul(&b, &mut backend).unwrap();
/// assert_eq!(c.shape(), &[2, 2]);
/// ```
pub trait TensorOpsExt {
    /// Convert to a different dtype using the checked conversion lattice.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::UnsupportedDTypeConversion`] when the
    /// conversion is outside the checked lattice,
    /// [`tenferro_tensor::Error::Validation`] with `DTypeMismatch` or
    /// `InvalidArgument` for invalid tensor metadata, or
    /// [`tenferro_tensor::Error::BackendSource`] when the backend reports a
    /// typed failure.
    fn convert<B: TensorBackend>(
        &self,
        to: DType,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Cast to a different dtype using explicit lossy projection.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::UnsupportedDTypeConversion`] when the
    /// requested cast is unsupported, [`tenferro_tensor::Error::Validation`]
    /// with `DTypeMismatch` or `InvalidArgument` for invalid tensor metadata,
    /// or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn cast<B: TensorBackend>(&self, to: DType, backend: &mut B)
        -> tenferro_tensor::Result<Tensor>;
    /// Elementwise addition with NumPy-style broadcasting.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with a
    /// [`ShapeMismatch`](tenferro_tensor::ValidationError::ShapeMismatch) or
    /// `DTypeMismatch` payload when operands are incompatible, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn add<B: TensorBackend>(
        &self,
        rhs: &Tensor,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise subtraction with NumPy-style broadcasting.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with
    /// `ShapeMismatch` or `DTypeMismatch` for incompatible operands, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn sub<B: TensorBackend>(
        &self,
        rhs: &Tensor,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise multiplication with NumPy-style broadcasting.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with
    /// `ShapeMismatch` or `DTypeMismatch` for incompatible operands, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn mul<B: TensorBackend>(
        &self,
        rhs: &Tensor,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise division with NumPy-style broadcasting.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` or
    /// `DTypeMismatch` for shape/dtype incompatibility,
    /// [`tenferro_tensor::Error::Extension`] with a numerical classification
    /// for a detected zero divisor, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn div<B: TensorBackend>(
        &self,
        rhs: &Tensor,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise remainder with NumPy-style broadcasting.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` or
    /// `DTypeMismatch` for shape/dtype incompatibility, a numerical
    /// [`tenferro_tensor::Error::Extension`] for a detected zero divisor, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn rem<B: TensorBackend>(
        &self,
        rhs: &Tensor,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise power with NumPy-style broadcasting.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` or
    /// `DTypeMismatch` for incompatible metadata, a numerical
    /// [`tenferro_tensor::Error::Extension`] for a detected negative integer
    /// exponent, or [`tenferro_tensor::Error::BackendSource`] for a typed
    /// backend failure.
    fn pow<B: TensorBackend>(
        &self,
        rhs: &Tensor,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise maximum with NumPy-style broadcasting.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with
    /// `ShapeMismatch` or `DTypeMismatch` for incompatible operands, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn maximum<B: TensorBackend>(
        &self,
        rhs: &Tensor,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise minimum with NumPy-style broadcasting.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with
    /// `ShapeMismatch` or `DTypeMismatch` for incompatible operands, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn minimum<B: TensorBackend>(
        &self,
        rhs: &Tensor,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise negation.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] when the dtype is not
    /// supported by the operation, or [`tenferro_tensor::Error::BackendSource`]
    /// for a typed backend failure.
    fn neg<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise absolute value.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn abs<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise sign.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn sign<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise complex conjugate.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn conj<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise exponential.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn exp<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise natural logarithm.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn log<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise sine.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn sin<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise cosine.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn cos<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise hyperbolic tangent.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn tanh<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise square root.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn sqrt<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise reciprocal square root.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn rsqrt<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise `exp(x) - 1`.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn expm1<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise `log(1 + x)`.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn log1p<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise comparison with NumPy-style broadcasting.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` or
    /// `DTypeMismatch` for incompatible shape/dtype metadata, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn compare<B: TensorBackend>(
        &self,
        rhs: &Tensor,
        dir: CompareDir,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Select values from `on_true` or `on_false` using this tensor as condition.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` or
    /// `DTypeMismatch` when the condition and branches are incompatible, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn where_select<B: TensorBackend>(
        &self,
        on_true: &Tensor,
        on_false: &Tensor,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Clamp values elementwise between lower and upper bounds.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` or
    /// `DTypeMismatch` when bounds are incompatible with the input, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn clamp<B: TensorBackend>(
        &self,
        lower: &Tensor,
        upper: &Tensor,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Rank-2 matrix multiplication.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `RankMismatch`,
    /// `ShapeMismatch`, or `DTypeMismatch` for incompatible matrices, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn matmul<B: TensorBackend>(
        &self,
        rhs: &Tensor,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Reshape without changing element order.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with
    /// `ShapeMismatch`, `RankMismatch`, or `InvalidArgument` when element
    /// counts or ranks are invalid, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn reshape<B: TensorBackend>(
        &self,
        shape: &[usize],
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Permute axes.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with
    /// `InvalidPermutationLength`, `AxisOutOfBounds`, or `DuplicateAxis` for
    /// an invalid permutation, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn transpose<B: TensorBackend>(
        &self,
        perm: &[usize],
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Sum over one or more axes.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `AxisOutOfBounds`
    /// or `DuplicateAxis` for invalid reductions, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn reduce_sum<B: TensorBackend>(
        &self,
        axes: &[usize],
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;
}

/// Backend-explicit operations for dynamic-rank typed tensors.
///
/// `TypedTensor` is owned by `tenferro-tensor`, so `tenferro-runtime` exposes
/// these operations as a crate-root extension trait rather than as inherent
/// methods.
///
/// # Public API rationale
///
/// This trait is intentionally public for the same reason as [`TensorOpsExt`]:
/// downstream users need a supported backend-explicit typed tensor surface, and
/// `tenferro-runtime` cannot add inherent methods to a type owned by
/// `tenferro-tensor`. The private `typed_tensor` module is implementation
/// detail, not a retained module/free-function API.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::CpuBackend;
/// use tenferro_runtime::{TypedTensor, TypedTensorOpsExt};
///
/// let mut backend = CpuBackend::new();
/// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap();
/// let y = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![3.0, 4.0]).unwrap();
/// let sum = x.add(&y, &mut backend).unwrap();
/// assert_eq!(sum.host_data().unwrap(), &[4.0, 6.0]);
/// ```
pub trait TypedTensorOpsExt<T: TensorScalar> {
    /// Elementwise addition with NumPy-style broadcasting.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` for
    /// incompatible operands, or [`tenferro_tensor::Error::BackendSource`] for
    /// a typed backend failure.
    fn add<B: TensorBackend>(
        &self,
        rhs: &TypedTensor<T>,
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise subtraction with NumPy-style broadcasting.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` for
    /// incompatible operands, or [`tenferro_tensor::Error::BackendSource`] for
    /// a typed backend failure.
    fn sub<B: TensorBackend>(
        &self,
        rhs: &TypedTensor<T>,
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise multiplication with NumPy-style broadcasting.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` for
    /// incompatible operands, or [`tenferro_tensor::Error::BackendSource`] for
    /// a typed backend failure.
    fn mul<B: TensorBackend>(
        &self,
        rhs: &TypedTensor<T>,
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise division with NumPy-style broadcasting.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` for
    /// incompatible shapes, a numerical [`tenferro_tensor::Error::Extension`]
    /// for a detected zero divisor, or [`tenferro_tensor::Error::BackendSource`]
    /// for a typed backend failure.
    fn div<B: TensorBackend>(
        &self,
        rhs: &TypedTensor<T>,
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise remainder with NumPy-style broadcasting.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` for
    /// incompatible shapes, a numerical [`tenferro_tensor::Error::Extension`]
    /// for a detected zero divisor, or [`tenferro_tensor::Error::BackendSource`]
    /// for a typed backend failure.
    fn rem<B: TensorBackend>(
        &self,
        rhs: &TypedTensor<T>,
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise power with NumPy-style broadcasting.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` for
    /// incompatible shapes, a numerical [`tenferro_tensor::Error::Extension`]
    /// for a detected negative integer exponent, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn pow<B: TensorBackend>(
        &self,
        rhs: &TypedTensor<T>,
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise maximum with NumPy-style broadcasting.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` for
    /// incompatible operands, or [`tenferro_tensor::Error::BackendSource`] for
    /// a typed backend failure.
    fn maximum<B: TensorBackend>(
        &self,
        rhs: &TypedTensor<T>,
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise minimum with NumPy-style broadcasting.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` for
    /// incompatible operands, or [`tenferro_tensor::Error::BackendSource`] for
    /// a typed backend failure.
    fn minimum<B: TensorBackend>(
        &self,
        rhs: &TypedTensor<T>,
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise negation.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn neg<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise absolute value.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn abs<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise sign.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn sign<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise complex conjugate.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn conj<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise exponential.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn exp<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise natural logarithm.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn log<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise sine.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn sin<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise cosine.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn cos<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise hyperbolic tangent.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn tanh<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise square root.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn sqrt<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise reciprocal square root.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn rsqrt<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise `exp(x) - 1`.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn expm1<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise `log(1 + x)`.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn log1p<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise comparison with NumPy-style broadcasting.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with
    /// `ShapeMismatch::IncompatibleShapes` when broadcasting the operands is
    /// impossible, or [`tenferro_tensor::Error::BackendSource`] for a typed
    /// backend failure.
    fn compare<B: TensorBackend>(
        &self,
        rhs: &TypedTensor<T>,
        dir: CompareDir,
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedTensor<bool>>;
    /// Clamp values elementwise between lower and upper bounds.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with
    /// `ShapeMismatch::IncompatibleShapes` when a bound cannot broadcast to
    /// the input, or [`tenferro_tensor::Error::BackendSource`] for a typed
    /// backend failure.
    fn clamp<B: TensorBackend>(
        &self,
        lower: &TypedTensor<T>,
        upper: &TypedTensor<T>,
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Rank-2 matrix multiplication.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `RankMismatch` when
    /// either operand is not rank two or `ShapeMismatch::ContractedDimensions`
    /// when the inner dimensions differ, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn matmul<B: TensorBackend>(
        &self,
        rhs: &TypedTensor<T>,
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Sum over one or more axes.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `AxisOutOfBounds`
    /// for an axis outside the input rank or `DuplicateAxis` when `axes`
    /// repeats an axis, or [`tenferro_tensor::Error::BackendSource`] for a
    /// typed backend failure.
    fn reduce_sum<B: TensorBackend>(
        &self,
        axes: &[usize],
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Reshape through the backend structural operation.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with
    /// `ShapeMismatch::ReshapeElementCount` when the element counts differ,
    /// `IntegerOverflow` when shape arithmetic overflows, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn reshape<B: TensorBackend>(
        &self,
        shape: &[usize],
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Permute axes through the backend structural operation.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with
    /// `InvalidPermutationLength` when `perm` has the wrong length,
    /// `AxisOutOfBounds` for an invalid axis, or `DuplicateAxis` for a
    /// repeated axis, or [`tenferro_tensor::Error::BackendSource`] for a typed
    /// backend failure.
    fn transpose<B: TensorBackend>(
        &self,
        perm: &[usize],
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Broadcast into a larger shape.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `RankMismatch` when
    /// `dims` does not match the input rank, `AxisOutOfBounds` or
    /// `DuplicateAxis` for an invalid mapping, or
    /// `ShapeMismatch::IncompatibleShapes` when known dimensions cannot
    /// broadcast. [`tenferro_tensor::Error::BackendSource`] reports a typed
    /// backend failure.
    fn broadcast_in_dim<B: TensorBackend>(
        &self,
        shape: &[usize],
        dims: &[usize],
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
}

/// Backend-explicit bool-mask operations for typed tensors.
///
/// # Public API rationale
///
/// This trait keeps `where_select` available as a method on bool
/// `TypedTensor`s while preserving the crate-root extension-trait surface. It
/// is public because downstream users call it directly; the implementation
/// helper in the private `typed_tensor` module is not a compatibility API.
pub trait TypedTensorMaskOpsExt {
    /// Select typed values using this bool tensor as condition.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with
    /// `ShapeMismatch::IncompatibleShapes` when the condition or either branch
    /// cannot broadcast to the other operands, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn where_select<T: TensorScalar, B: TensorBackend>(
        &self,
        on_true: &TypedTensor<T>,
        on_false: &TypedTensor<T>,
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
}

pub use traced::TracedTensor;
