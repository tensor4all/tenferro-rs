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
mod shape_infer;
mod shape_packing;
pub mod sym_dim;
mod tensor;
pub mod traced;
mod typed_tensor;

pub use compiler::{CompilerOptions, OptimizerConfig};
pub use error::{ContextId, Error, Result};
pub use extension_cache::{
    ExtensionCacheKey, ExtensionCacheLimits, ExtensionCacheSelector, ExtensionCacheStore,
};
pub use extension_runtime::{
    ExtensionExecutionContext, ExtensionExecutor, ExtensionRegistry, ExtensionRuntime,
    ExtensionRuntimeRegistryError,
};
pub use graph::{
    GraphCompiler, GraphCompilerCacheStats, GraphExecutor, GraphExecutorCacheStats,
    GraphInstructionView, GraphOpView, GraphProgram, GraphProgramInput,
    GraphProgramLoweringShapeError, GraphProgramLoweringView,
};
pub use sym_dim::SymDim;
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
    fn convert<B: TensorBackend>(
        &self,
        to: DType,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Cast to a different dtype using explicit lossy projection.
    fn cast<B: TensorBackend>(&self, to: DType, backend: &mut B)
        -> tenferro_tensor::Result<Tensor>;
    /// Elementwise addition with NumPy-style broadcasting.
    fn add<B: TensorBackend>(
        &self,
        rhs: &Tensor,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise subtraction with NumPy-style broadcasting.
    fn sub<B: TensorBackend>(
        &self,
        rhs: &Tensor,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise multiplication with NumPy-style broadcasting.
    fn mul<B: TensorBackend>(
        &self,
        rhs: &Tensor,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise division with NumPy-style broadcasting.
    fn div<B: TensorBackend>(
        &self,
        rhs: &Tensor,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise power with NumPy-style broadcasting.
    fn pow<B: TensorBackend>(
        &self,
        rhs: &Tensor,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise maximum with NumPy-style broadcasting.
    fn maximum<B: TensorBackend>(
        &self,
        rhs: &Tensor,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise minimum with NumPy-style broadcasting.
    fn minimum<B: TensorBackend>(
        &self,
        rhs: &Tensor,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise negation.
    fn neg<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise absolute value.
    fn abs<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise sign.
    fn sign<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise complex conjugate.
    fn conj<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise exponential.
    fn exp<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise natural logarithm.
    fn log<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise sine.
    fn sin<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise cosine.
    fn cos<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise hyperbolic tangent.
    fn tanh<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise square root.
    fn sqrt<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise reciprocal square root.
    fn rsqrt<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise `exp(x) - 1`.
    fn expm1<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise `log(1 + x)`.
    fn log1p<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise comparison with NumPy-style broadcasting.
    fn compare<B: TensorBackend>(
        &self,
        rhs: &Tensor,
        dir: CompareDir,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Select values from `on_true` or `on_false` using this tensor as condition.
    fn where_select<B: TensorBackend>(
        &self,
        on_true: &Tensor,
        on_false: &Tensor,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Clamp values elementwise between lower and upper bounds.
    fn clamp<B: TensorBackend>(
        &self,
        lower: &Tensor,
        upper: &Tensor,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Rank-2 matrix multiplication.
    fn matmul<B: TensorBackend>(
        &self,
        rhs: &Tensor,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Reshape without changing element order.
    fn reshape<B: TensorBackend>(
        &self,
        shape: &[usize],
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Permute axes.
    fn transpose<B: TensorBackend>(
        &self,
        perm: &[usize],
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Sum over one or more axes.
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
    fn add<B: TensorBackend>(
        &self,
        rhs: &TypedTensor<T>,
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise subtraction with NumPy-style broadcasting.
    fn sub<B: TensorBackend>(
        &self,
        rhs: &TypedTensor<T>,
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise multiplication with NumPy-style broadcasting.
    fn mul<B: TensorBackend>(
        &self,
        rhs: &TypedTensor<T>,
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise division with NumPy-style broadcasting.
    fn div<B: TensorBackend>(
        &self,
        rhs: &TypedTensor<T>,
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise power with NumPy-style broadcasting.
    fn pow<B: TensorBackend>(
        &self,
        rhs: &TypedTensor<T>,
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise maximum with NumPy-style broadcasting.
    fn maximum<B: TensorBackend>(
        &self,
        rhs: &TypedTensor<T>,
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise minimum with NumPy-style broadcasting.
    fn minimum<B: TensorBackend>(
        &self,
        rhs: &TypedTensor<T>,
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise negation.
    fn neg<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise absolute value.
    fn abs<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise sign.
    fn sign<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise complex conjugate.
    fn conj<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise exponential.
    fn exp<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise natural logarithm.
    fn log<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise sine.
    fn sin<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise cosine.
    fn cos<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise hyperbolic tangent.
    fn tanh<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise square root.
    fn sqrt<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise reciprocal square root.
    fn rsqrt<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise `exp(x) - 1`.
    fn expm1<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise `log(1 + x)`.
    fn log1p<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise comparison with NumPy-style broadcasting.
    fn compare<B: TensorBackend>(
        &self,
        rhs: &TypedTensor<T>,
        dir: CompareDir,
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedTensor<bool>>;
    /// Clamp values elementwise between lower and upper bounds.
    fn clamp<B: TensorBackend>(
        &self,
        lower: &TypedTensor<T>,
        upper: &TypedTensor<T>,
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Rank-2 matrix multiplication.
    fn matmul<B: TensorBackend>(
        &self,
        rhs: &TypedTensor<T>,
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Sum over one or more axes.
    fn reduce_sum<B: TensorBackend>(
        &self,
        axes: &[usize],
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Reshape through the backend structural operation.
    fn reshape<B: TensorBackend>(
        &self,
        shape: &[usize],
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Permute axes through the backend structural operation.
    fn transpose<B: TensorBackend>(
        &self,
        perm: &[usize],
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Broadcast into a larger shape.
    fn broadcast_in_dim<B: TensorBackend>(
        &self,
        shape: &[usize],
        dims: &[usize],
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
}

/// Backend-explicit bool-mask operations for typed tensors.
pub trait TypedTensorMaskOpsExt {
    /// Select typed values using this bool tensor as condition.
    fn where_select<T: TensorScalar, B: TensorBackend>(
        &self,
        on_true: &TypedTensor<T>,
        on_false: &TypedTensor<T>,
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
}

pub use traced::TracedTensor;
