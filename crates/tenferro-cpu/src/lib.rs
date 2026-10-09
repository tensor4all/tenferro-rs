//! CPU backend, kernels, provider selection, and CPU resource pools.
//!
//! # Examples
//!
//! ```rust
//! use tenferro_cpu::CpuBackend;
//! use tenferro_tensor::{BackendSessionHost, Tensor, TensorRead};
//!
//! let mut backend = CpuBackend::new();
//! let a = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
//! let b = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0])?;
//! let c = backend
//!     .with_backend_session(|session| {
//!         session.add_read(TensorRead::from_tensor(&a), TensorRead::from_tensor(&b))
//!     })??;
//! assert_eq!(c.as_slice::<f64>().unwrap(), &[4.0, 6.0]);
//! # Ok::<(), tenferro_tensor::Error>(())
//! ```
//!
//! The deleted one-shot spellings do not compile on the owner or on a session.
//! Each fixture below fails for that reason and nothing else.
//!
//! ```compile_fail
//! use tenferro_cpu::CpuBackend;
//! use tenferro_tensor::Tensor;
//!
//! let mut backend = CpuBackend::new();
//! let a = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
//! let b = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap();
//! let _ = backend.add(&a, &b);
//! ```
//!
//! ```compile_fail
//! use tenferro_cpu::CpuBackend;
//! use tenferro_tensor::Tensor;
//!
//! let mut backend = CpuBackend::new();
//! let a = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
//! let b = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap();
//! let _ = backend.mul(&a, &b);
//! ```
//!
//! ```compile_fail
//! use tenferro_cpu::CpuBackend;
//! use tenferro_tensor::Tensor;
//!
//! let mut backend = CpuBackend::new();
//! let a = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
//! let _ = backend.exp(&a);
//! ```
//!
//! ```compile_fail
//! use tenferro_cpu::CpuBackend;
//! use tenferro_tensor::Tensor;
//!
//! let mut backend = CpuBackend::new();
//! let a = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
//! let _ = backend.reduce_sum(&a, &[0]);
//! ```
//!
//! ```compile_fail
//! use tenferro_cpu::CpuBackend;
//! use tenferro_tensor::Tensor;
//!
//! let mut backend = CpuBackend::new();
//! let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
//! let _ = backend.transpose(&a, &[1, 0]);
//! ```
//!
//! ```compile_fail
//! use tenferro_cpu::CpuBackend;
//! use tenferro_tensor::{DotGeneralConfig, Tensor};
//!
//! let mut backend = CpuBackend::new();
//! let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
//! let config = DotGeneralConfig {
//!     lhs_contracting_dims: [1].as_slice().into(),
//!     rhs_contracting_dims: [0].as_slice().into(),
//!     lhs_batch_dims: [].as_slice().into(),
//!     rhs_batch_dims: [].as_slice().into(),
//! };
//! let _ = backend.dot_general(&a, &a, &config);
//! ```
//!
//! The owner no longer implements the cache-aware contraction entry, so an
//! owner-level `BackendCachedDot` bound does not hold either:
//!
//! ```compile_fail
//! use tenferro_cpu::CpuBackend;
//! use tenferro_tensor::BackendCachedDot;
//!
//! fn requires_cached_dot<B: BackendCachedDot>(_backend: &mut B) {}
//!
//! let mut backend = CpuBackend::new();
//! requires_cached_dot(&mut backend);
//! ```
#![cfg_attr(docsrs, feature(doc_cfg))]
// With neither backend the numerical routes are unreachable by design, so the
// implementation they would call is dead code in that configuration only.
#![cfg_attr(
    all(not(feature = "native"), not(feature = "blas")),
    allow(dead_code, unused_imports)
)]
// `provider-inject` unit tests deliberately omit the broad default-backend
// suite below because no fixture has registered its FFI symbols. That makes
// private helpers referenced only by the broad suite appear unused in this one
// test build; call-through coverage lives in the registered integration test.
#![cfg_attr(
    all(test, feature = "provider-inject"),
    allow(dead_code, unused_imports)
)]

/// The Rust scalar type behind a preset variant name a macro received.
macro_rules! preset_scalar {
    (F32) => {
        f32
    };
    (F64) => {
        f64
    };
    (I32) => {
        i32
    };
    (I64) => {
        i64
    };
    (Bool) => {
        bool
    };
    (C32) => {
        num_complex::Complex32
    };
    (C64) => {
        num_complex::Complex64
    };
}
// A build with neither backend compiles: the CPU numerical routes report a
// typed `Unsupported` instead. `native` is the default feature, so a normal
// dependency always gets a backend, and selecting `blas` alone is the supported
// alternative. This keeps any subset of the stack compilable, which is what
// consumers that select a backend crate by crate rely on.
#[cfg(all(feature = "native", feature = "blas"))]
compile_error!("native and blas are mutually exclusive; use --no-default-features for blas");

#[cfg(all(feature = "provider-inject", not(feature = "blas")))]
compile_error!("provider-inject requires blas");

#[cfg(any(
    all(feature = "blas-openblas", feature = "blas-accelerate"),
    all(feature = "blas-openblas", feature = "blas-mkl"),
    all(feature = "blas-accelerate", feature = "blas-mkl"),
))]
compile_error!(
    "enable at most one explicit BLAS provider feature: blas-openblas, blas-accelerate, or blas-mkl"
);

#[cfg(all(
    feature = "provider-inject",
    any(
        feature = "blas-openblas",
        feature = "blas-accelerate",
        feature = "blas-mkl"
    )
))]
compile_error!("provider-inject cannot be combined with explicit BLAS provider features");

pub mod affinity;
mod analytic;
mod arbiter;
pub mod backend;
mod blas1;
pub(crate) mod buffer_pool {
    pub use tenferro_cpu_basic::buffer_pool::*;
}
mod capability;
mod context;
mod dot_runtime;
pub(crate) use tenferro_cpu_basic::PooledUninitOutput;
pub(crate) use tenferro_cpu_basic::{erased_raw_strided_ref, erased_raw_strided_uninit_mut};
pub(crate) use tenferro_internal_cpu_kernels::elementwise;
mod contraction;
mod engine;
mod exec_session;
#[doc(hidden)]
pub use contraction::ContractionWorkspaces;
mod gemm;
mod indexed_plan_cache;
mod indexing;
#[cfg(feature = "provider-inject")]
pub mod inject;
mod placement;
pub mod provider;
mod reduction;
mod resource_domain;
mod runtime_adapter;
mod structural;
mod topology;

use num_complex::{Complex32, Complex64};
#[cfg(test)]
use strided_kernel::col_major_strides as kernel_col_major_strides;
#[cfg(test)]
use strided_kernel::StridedArray;

use crate::buffer_pool::BufferPool;
pub(crate) use tenferro_tensor::*;

/// Stable provider identity of the compile-time-selected CPU backend.
#[doc(hidden)]
pub fn cpu_provider_id() -> &'static str {
    #[cfg(feature = "native")]
    {
        "tenferro.cpu.faer"
    }
    #[cfg(feature = "blas")]
    {
        "tenferro.cpu.blas"
    }
    #[cfg(all(not(feature = "native"), not(feature = "blas")))]
    {
        "tenferro.cpu.none"
    }
}

pub(crate) fn cpu_contraction_unsupported_dtype_message(dtype: DType) -> String {
    let remedy = matches!(dtype, DType::I32 | DType::I64)
        .then_some(format!("; convert {dtype:?} to F64 before contraction"));
    format!(
        "CPU contraction providers support F32/F64/C32/C64{}",
        remedy.unwrap_or_default()
    )
}

#[cfg(feature = "provider-src")]
extern crate blas_src as _;
#[cfg(feature = "provider-inject")]
extern crate cblas_inject as _;
#[cfg(feature = "provider-inject")]
extern crate lapack_inject as _;
#[cfg(feature = "provider-src")]
extern crate lapack_src as _;

pub use affinity::{
    available_parallelism, process_cpu_affinity, process_cpu_affinity_count, CpuAffinityError,
};
pub use backend::execution_scope::{current_cpu_execution, CpuThreadExecution};
pub use backend::{CpuBackend, CpuBackendError, CpuRuntimeIdentity};

pub use buffer_pool::BufferPoolStats;
pub use capability::cpu_capabilities;
pub use context::DEFAULT_WORKER_STACK_BYTES;
// `CpuContext` is the engine's own resource holder. It stays reachable for the
// crate's benchmarks, but it is not part of the supported CPU API: placement
// goes through `CpuBackend::builder()`, and lower libraries see
// `CpuExecutionContext`.
#[doc(hidden)]
pub use context::{CpuContext, CpuContextError};
#[doc(hidden)]
pub use dot_runtime::{validate_cpu_host_read, validate_cpu_host_write};
#[doc(hidden)]
pub use exec_session::{CpuChildExecution, CpuExecSession};
pub use indexed_plan_cache::IndexedPlanCacheLimits;
pub use placement::{
    CpuEngineConstructionError, CpuPlacement, CpuPlacementError, ResolvedCpuPlacement,
};
pub use provider::{CpuExecutionContext, ParallelMode};
pub use runtime_adapter::{
    runtime_engine_id, runtime_engine_registration, runtime_engine_registration_with_id,
    runtime_hardware_class,
};
/// Ordinary CPU entry points that take a caller-provided destination and the
/// caller's own arithmetic instead of the typed pool.
pub use tenferro_internal_cpu_kernels::scalar_ops::{
    scalar_binary_into, scalar_fold, AddOp, BinaryScalarOp, MulOp, SubOp,
};
pub use tenferro_internal_cpu_kernels::{same_variant_pair, same_variant_unary};
pub use topology::{
    discover_cpu_topology, CpuId, CpuNode, CpuSet, CpuSetError, CpuTopology, CpuTopologyError,
    NumaNodeId,
};

/// Visit a CPU execution session carried by a type-erased backend session.
///
/// This is a backend-leaf capability bridge. The exact session marker is checked
/// before the erased pointer is reconstructed, and the callback cannot return a
/// borrow of the session, so the borrowed resource lease remains scoped to the
/// caller's session closure.
#[doc(hidden)]
pub fn with_cpu_exec_session<B, R>(
    session: &mut B,
    f: impl for<'a> FnOnce(&'a mut CpuExecSession<'a>) -> R,
) -> Option<R>
where
    B: tenferro_tensor::BackendSession + ?Sized,
{
    let data = session
        .native_session()?
        .into_marked_ptr::<exec_session::CpuExecSessionMarker>()?;
    // SAFETY: only `CpuExecSession::native_session` creates a token with the
    // crate-private `CpuExecSessionMarker`, and it points that token at a live
    // `CpuExecSession`. The token borrowed `*session` exclusively, and this
    // function keeps holding `session: &mut B` for the whole visit. The
    // callback is higher-ranked and returns no session borrow, so the
    // reconstructed reference cannot escape the original session borrow.
    Some(unsafe { f(data.cast::<CpuExecSession<'static>>().as_mut()) })
}

/// Invoke a direct faer operation with the parallelism selected by a CPU session.
///
/// The `faer::Par` value is scoped to the callback and is derived from the
/// session's managed thread budget and nesting policy. A non-CPU session, or a
/// CPU session built without `native`, returns a typed unsupported error.
/// `Par::Seq` remains the portable choice for direct calls outside a session.
///
/// # Examples
///
/// ```rust
/// # #[cfg(feature = "native")]
/// # fn example() -> tenferro_tensor::Result<()> {
/// use tenferro_cpu::{CpuBackend, FaerParallelismExt};
/// use tenferro_tensor::BackendSessionHost;
///
/// let mut backend = CpuBackend::with_threads(2)?;
/// backend.with_backend_session(|session| {
///     session.with_faer_parallelism(|parallel| {
///         let _ = parallel;
///         Ok(())
///     })
/// })??;
/// # Ok(())
/// # }
/// # fn main() {}
/// ```
#[cfg(feature = "native")]
#[cfg_attr(docsrs, doc(cfg(feature = "native")))]
pub trait FaerParallelismExt {
    /// Run a scoped callback with this session's faer parallelism policy.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] when the session is not
    /// a CPU/faer execution session, or the callback's own typed error.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "native")]
    /// # fn example(session: &mut dyn tenferro_tensor::BackendSession) -> tenferro_tensor::Result<()> {
    /// use tenferro_cpu::FaerParallelismExt;
    /// session.with_faer_parallelism(|parallel| {
    ///     let _ = parallel;
    ///     Ok::<_, tenferro_tensor::Error>(())
    /// })?;
    /// # Ok(())
    /// # }
    /// ```
    fn with_faer_parallelism(
        &mut self,
        callback: impl FnOnce(faer::Par) -> tenferro_tensor::Result<()> + Send,
    ) -> tenferro_tensor::Result<()>;
}

#[cfg(feature = "native")]
impl<S> FaerParallelismExt for S
where
    S: tenferro_tensor::BackendSession + ?Sized,
{
    fn with_faer_parallelism(
        &mut self,
        callback: impl FnOnce(faer::Par) -> tenferro_tensor::Result<()> + Send,
    ) -> tenferro_tensor::Result<()> {
        with_cpu_exec_session(self, |session| session.with_faer_parallelism(callback))
            .unwrap_or_else(|| {
                Err(tenferro_tensor::Error::unsupported(
                    "with_faer_parallelism",
                    "selected session is not a CPU/faer execution session",
                ))
            })
    }
}

// Unit tests exercise the pool-aware kernels through the former convenience
// names without restoring those names to the production crate surface.
#[cfg(test)]
pub(crate) use analytic::pow;
#[cfg(test)]
macro_rules! test_elementwise_wrapper {
    ($name:ident($($arg:ident: $ty:ty),*) => $with_pool:ident) => {
        pub(crate) fn $name($($arg: $ty),*) -> crate::Result<Tensor> {
            let mut buffers = BufferPool::new();
            elementwise::$with_pool(&mut buffers, &strided_kernel::ExecContext::serial(), $($arg),*)
        }
    };
}
#[cfg(test)]
test_elementwise_wrapper!(abs(input: &Tensor) => abs_with_pool);
#[cfg(test)]
test_elementwise_wrapper!(add(lhs: &Tensor, rhs: &Tensor) => add_with_pool);
#[cfg(test)]
test_elementwise_wrapper!(clamp(input: &Tensor, lower: &Tensor, upper: &Tensor) => clamp_with_pool);
#[cfg(test)]
test_elementwise_wrapper!(compare(lhs: &Tensor, rhs: &Tensor, dir: &CompareDir) => compare_with_pool);
#[cfg(test)]
test_elementwise_wrapper!(conj(input: &Tensor) => conj_with_pool);
#[cfg(test)]
test_elementwise_wrapper!(div(lhs: &Tensor, rhs: &Tensor) => div_with_pool);
#[cfg(test)]
test_elementwise_wrapper!(maximum(lhs: &Tensor, rhs: &Tensor) => maximum_with_pool);
#[cfg(test)]
test_elementwise_wrapper!(minimum(lhs: &Tensor, rhs: &Tensor) => minimum_with_pool);
#[cfg(test)]
test_elementwise_wrapper!(mul(lhs: &Tensor, rhs: &Tensor) => mul_with_pool);
#[cfg(test)]
test_elementwise_wrapper!(neg(input: &Tensor) => neg_with_pool);
#[cfg(test)]
test_elementwise_wrapper!(rem(lhs: &Tensor, rhs: &Tensor) => rem_with_pool);
#[cfg(test)]
test_elementwise_wrapper!(select(pred: &Tensor, on_true: &Tensor, on_false: &Tensor) => select_with_pool);
#[cfg(test)]
test_elementwise_wrapper!(sign(input: &Tensor) => sign_with_pool);
#[cfg(test)]
test_elementwise_wrapper!(sub(lhs: &Tensor, rhs: &Tensor) => sub_with_pool);
#[cfg(test)]
pub(crate) use indexing::{dynamic_slice, dynamic_update_slice, gather, pad, scatter};
#[cfg(test)]
pub(crate) use reduction::{reduce_max, reduce_min, reduce_prod, reduce_sum, reduce_sum_squares};
#[cfg(test)]
pub(crate) use structural::{
    broadcast_in_dim, embed_diagonal, extract_diagonal, reshape, transpose, tril, triu,
};

/// Owner-scoped CPU scratch-pool API for operation-family crates.
///
/// This module is not an application-facing tensor API. It exists so
/// operation crates that implement CPU kernels can share `CpuBackend`'s
/// allocation pool without exposing the pool as a general public contract.
#[doc(hidden)]
pub mod linalg_interop {
    pub use crate::buffer_pool::{BufferPool, PoolScalar};
    pub use tenferro_cpu_basic::PooledUninitOutput;
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum CpuNumericalError {
    #[error("{op} received a negative integer exponent for dtype {dtype:?}")]
    NegativeIntegerExponent { op: &'static str, dtype: DType },
}

pub(crate) fn cpu_negative_integer_exponent(op: &'static str, dtype: DType) -> crate::Error {
    crate::Error::extension(
        op,
        "cpu",
        ErrorKind::NumericalFailure,
        CpuNumericalError::NegativeIntegerExponent { op, dtype },
    )
}

pub(crate) use tenferro_cpu_basic::{
    cpu_backend_buffer_error, typed_host_data, typed_view, typed_view_from_view,
};
pub(crate) fn materialize_tensor_read_in_domain(
    buffers: &mut BufferPool,
    op: &'static str,
    input: TensorRead<'_>,
    domain: Option<&dyn SharedTensorAllocationDomain>,
) -> crate::Result<Tensor> {
    if let Some(domain) = domain {
        if input.backend_family().is_some() {
            return materialize_managed_read(op, input, domain);
        }
    }
    materialize_tensor_read(buffers, op, input)
}

pub(crate) fn materialize_tensor_read(
    buffers: &mut BufferPool,
    op: &'static str,
    input: TensorRead<'_>,
) -> crate::Result<Tensor> {
    match input {
        TensorRead::Tensor(tensor) => clone_host_tensor_read(op, tensor),
        TensorRead::View(view) => materialize_tensor_view(buffers, op, view),
    }
}

fn materialize_managed_read(
    op: &'static str,
    input: TensorRead<'_>,
    domain: &dyn SharedTensorAllocationDomain,
) -> crate::Result<Tensor> {
    if input.placement().memory_kind != MemoryKind::Managed {
        return Err(Error::host_access(
            op,
            HostAccessError::Unsupported { backend: "backend" },
        ));
    }
    match input.allocation_domain() {
        Some(actual) if actual == domain.id() => {}
        Some(actual) => {
            return Err(Error::host_access(
                op,
                HostAccessError::ForeignDomain {
                    expected: domain.id(),
                    actual,
                },
            ))
        }
        None => {
            return Err(Error::host_access(
                op,
                HostAccessError::Unsupported { backend: "backend" },
            ))
        }
    }
    fn copy<T: TensorScalar>(
        op: &'static str,
        input: TypedTensorView<'_, T>,
        domain: &dyn SharedTensorAllocationDomain,
    ) -> crate::Result<Tensor> {
        // INVARIANT: semantic snapshots require independent storage, not a new
        // writable alias. Both mappings remain scoped to this same-domain copy.
        input.with_host_read(|source| {
            let mut output = domain.allocate(T::dtype(), input.shape())?;
            if output.shape() != input.shape()
                || output.placement().memory_kind != MemoryKind::Managed
                || TensorRead::from_tensor(&output).allocation_domain() != Some(domain.id())
            {
                return Err(Error::runtime_state(
                    op,
                    "shared allocator returned incompatible output",
                ));
            }
            let typed = output.as_typed_mut::<T>().ok_or_else(|| {
                Error::runtime_state(op, "shared allocator returned the wrong dtype")
            })?;
            if let Some(buffer) = typed.backend_buffer_mut() {
                buffer
                    .map_write()
                    .map_err(|error| Error::host_access(op, error))?
                    .copy_from_slice(source)
                    .map_err(|error| Error::host_access(op, error))?;
            } else {
                typed.with_host_write(|target| target.copy_from_slice(source))?;
            }
            Ok(output)
        })?
    }
    // Compact managed descriptors cover retained eager/traced values. Strided
    // managed canonicalization remains an explicit unsupported boundary.
    match input.tensor_view() {
        TensorView::F32(view) => copy(op, view, domain),
        TensorView::F64(view) => copy(op, view, domain),
        TensorView::I32(view) => copy(op, view, domain),
        TensorView::I64(view) => copy(op, view, domain),
        TensorView::Bool(view) => copy(op, view, domain),
        TensorView::C32(view) => copy(op, view, domain),
        TensorView::C64(view) => copy(op, view, domain),
    }
}

pub(crate) fn copy_tensor_read_into(
    op: &'static str,
    src: TensorRead<'_>,
    dst: TensorWrite<'_>,
) -> crate::Result<()> {
    let src_dtype = src.dtype();
    let dst_dtype = dst.dtype();
    macro_rules! copy_source {
        ($variant:ident, $src:expr) => {{
            let src = $src;
            match dst {
                TensorWrite::Tensor(dst)
                    if dst.dtype()
                        == <preset_scalar!($variant) as tenferro_tensor::TensorScalar>::dtype() =>
                {
                    let dst = dst
                        .as_typed_mut::<preset_scalar!($variant)>()
                        .expect("the dtype guard selects this arm");
                    let mut dst = dst.as_view_mut();
                    structural::typed_copy_view_into(&src, &mut dst, op)
                }
                TensorWrite::View(TensorViewMut::$variant(mut dst)) => {
                    structural::typed_copy_view_into(&src, &mut dst, op)
                }
                _ => Err(crate::Error::dtype_mismatch(op, src_dtype, dst_dtype)),
            }
        }};
    }
    /// The typed tensor behind a read adapter's tensor, or the refusal this adapter reports.
    fn read_refusal(tensor: &Tensor) -> crate::Error {
        crate::Error::unsupported_dtype(
            "copy_tensor_read_into",
            tensor.dtype(),
            "an externally defined payload is not a runtime read",
        )
    }

    match src {
        TensorRead::Tensor(tensor) => match tensor.dtype() {
            DType::F32 => copy_source!(
                F32,
                tensor
                    .as_typed::<f32>()
                    .ok_or_else(|| read_refusal(tensor))?
                    .as_view()
            ),
            DType::F64 => copy_source!(
                F64,
                tensor
                    .as_typed::<f64>()
                    .ok_or_else(|| read_refusal(tensor))?
                    .as_view()
            ),
            DType::I32 => copy_source!(
                I32,
                tensor
                    .as_typed::<i32>()
                    .ok_or_else(|| read_refusal(tensor))?
                    .as_view()
            ),
            DType::I64 => copy_source!(
                I64,
                tensor
                    .as_typed::<i64>()
                    .ok_or_else(|| read_refusal(tensor))?
                    .as_view()
            ),
            DType::Bool => copy_source!(
                Bool,
                tensor
                    .as_typed::<bool>()
                    .ok_or_else(|| read_refusal(tensor))?
                    .as_view()
            ),
            DType::C32 => copy_source!(
                C32,
                tensor
                    .as_typed::<Complex32>()
                    .ok_or_else(|| read_refusal(tensor))?
                    .as_view()
            ),
            DType::C64 => copy_source!(
                C64,
                tensor
                    .as_typed::<Complex64>()
                    .ok_or_else(|| read_refusal(tensor))?
                    .as_view()
            ),
            // A caller-owned payload has no compact runtime read.
            DType::External(_) => Err(read_refusal(tensor)),
        },
        TensorRead::View(TensorView::F32(src)) => copy_source!(F32, src),
        TensorRead::View(TensorView::F64(src)) => copy_source!(F64, src),
        TensorRead::View(TensorView::I32(src)) => copy_source!(I32, src),
        TensorRead::View(TensorView::I64(src)) => copy_source!(I64, src),
        TensorRead::View(TensorView::Bool(src)) => copy_source!(Bool, src),
        TensorRead::View(TensorView::C32(src)) => copy_source!(C32, src),
        TensorRead::View(TensorView::C64(src)) => copy_source!(C64, src),
        // A caller-owned payload is opaque here, so it cannot be copied into a
    }
}

/// Clone one owned host tensor into a fresh allocation.
///
/// The accepted inputs are: host placement and a preset scalar
/// (`validate_cpu_host_placement` + `typed_host_data`). `tenferro-ad`'s eager
/// leaf path states the same acceptance in `cpu_host_owned_read` so it can skip
/// the session, and `host_leaf_materialization_matches_the_cpu_backend_acceptance`
/// pins the two against each other. Change both together.
fn clone_host_tensor_read(op: &'static str, tensor: &Tensor) -> crate::Result<Tensor> {
    macro_rules! clone_host {
        ($variant:ident, $tensor:expr) => {{
            structural::validate_cpu_host_placement(op, "source", $tensor.placement())?;
            typed_host_data(op, $tensor)?;
            $tensor
                .duplicate()
                .map(Tensor::from_typed::<preset_scalar!($variant)>)
        }};
    }

    match tensor.dtype() {
        DType::F32 => {
            let tensor = host_typed::<f32>(op, tensor)?;
            clone_host!(F32, tensor)
        }
        DType::F64 => {
            let tensor = host_typed::<f64>(op, tensor)?;
            clone_host!(F64, tensor)
        }
        DType::I32 => {
            let tensor = host_typed::<i32>(op, tensor)?;
            clone_host!(I32, tensor)
        }
        DType::I64 => {
            let tensor = host_typed::<i64>(op, tensor)?;
            clone_host!(I64, tensor)
        }
        DType::Bool => {
            let tensor = host_typed::<bool>(op, tensor)?;
            clone_host!(Bool, tensor)
        }
        DType::C32 => {
            let tensor = host_typed::<Complex32>(op, tensor)?;
            clone_host!(C32, tensor)
        }
        DType::C64 => {
            let tensor = host_typed::<Complex64>(op, tensor)?;
            clone_host!(C64, tensor)
        }
        // A caller-owned payload is a compact host tensor, so a contiguous copy is
        // the payload itself, copied into storage this value owns. Sharing the
        // payload would alias the caller's storage instead of copying it.
        DType::External(_) => Ok(Tensor::external_with_placement(
            tensor
                .external_payload()
                .ok_or_else(|| host_typed_error(op, tensor))?
                .duplicate(),
            tensor.placement().clone(),
        )),
    }
}

/// The typed tensor behind `tensor`, or this module's refusal for a dtype it cannot clone.
///
/// Callers reach this from a match on `tensor.dtype()`, so `None` means the tag table and the
/// runtime dtype disagree rather than a caller mistake.
fn host_typed<'a, T: TensorScalar>(
    op: &'static str,
    tensor: &'a Tensor,
) -> crate::Result<&'a TypedTensor<T>> {
    tensor
        .as_typed::<T>()
        .ok_or_else(|| host_typed_error(op, tensor))
}

/// The refusal the accessor reports when the tag and the runtime dtype disagree.
fn host_typed_error(op: &'static str, tensor: &Tensor) -> crate::Error {
    crate::Error::unsupported_dtype(
        op,
        tensor.dtype(),
        "the CPU host clone requires a preset scalar",
    )
}

fn materialize_tensor_view(
    buffers: &mut BufferPool,
    op: &'static str,
    view: TensorView<'_>,
) -> crate::Result<Tensor> {
    macro_rules! materialize {
        ($variant:ident, $view:expr) => {{
            Ok(Tensor::from_typed::<preset_scalar!($variant)>(
                structural::typed_materialize_view_with_pool(buffers, &$view, op)?,
            ))
        }};
    }

    match view {
        TensorView::F32(view) => materialize!(F32, view),
        TensorView::F64(view) => materialize!(F64, view),
        TensorView::I32(view) => materialize!(I32, view),
        TensorView::I64(view) => materialize!(I64, view),
        TensorView::Bool(view) => materialize!(Bool, view),
        TensorView::C32(view) => materialize!(C32, view),
        TensorView::C64(view) => materialize!(C64, view),
    }
}

/// Create an output array WITHOUT initializing element values.
///
/// # Safety
/// Caller must write every element before reading. The returned array
/// contains uninitialized data.
#[allow(clippy::uninit_vec)]
#[cfg(test)]
pub(crate) unsafe fn typed_array_uninit<T>(shape: &[usize]) -> StridedArray<T> {
    let total: usize = shape.iter().product();
    let strides = kernel_col_major_strides(shape);
    let mut data = Vec::with_capacity(total);
    // SAFETY: test-only helper is used for outputs whose elements are fully overwritten.
    unsafe { data.set_len(total) };
    // Invariant: `kernel_col_major_strides(shape)` and `total` describe the
    // compact column-major array for this validated test output shape.
    StridedArray::from_parts(data, shape, &strides, 0).expect("column-major output array")
}

#[cfg(test)]
pub(crate) fn tensor_from_array<T: Clone + tenferro_tensor::TensorScalar>(
    array: StridedArray<T>,
) -> TypedTensor<T> {
    // Invariant: `StridedArray` owns data whose length matches its validated dimensions.
    TypedTensor::from_vec_col_major(array.dims().to_vec(), array.into_data())
        .expect("strided array dimensions match owned data length")
}

// `provider-inject` owns call-through coverage in the serialized integration
// fixture, which registers every BLAS symbol before the first operation.  The
// broad unit suite selects the compiled default backend and therefore must not
// call an intentionally unregistered injected symbol.
#[cfg(all(test, not(feature = "provider-inject")))]
mod tests;
