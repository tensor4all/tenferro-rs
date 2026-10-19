//! cuBLASLt batched GEMM plans for small standard batched matmuls.
//!
//! cuBLAS' strided-batched call issues one physical kernel per batch at the
//! sizes this path targets: the 16-by-32x32 f64 benchmark reaches cuBLAS as 16
//! kernels. cuBLASLt has genuinely batched kernels, so a small standard batched
//! matmul is routed here and one `cublasLtMatmul` covers every slice.
//!
//! The layout/compute descriptors, the heuristic result, and the workspace are
//! geometry-specific, so the whole plan is cached per stream slot and keyed by
//! dtype and the detected GEMM parameters. The per-call path is then one
//! `cublasLtMatmul` on the runtime's existing CubeCL stream, with the same
//! cross-stream operand hand-off and no extra host synchronization.
//!
//! The plan cache is owned by the runtime state. Its device workspace retires
//! through the runtime's event queue exactly like cuTENSOR scratch; at runtime
//! teardown every stream has already been retired before this state is dropped,
//! so a workspace whose runtime can no longer be upgraded releases its handle
//! directly.

use std::collections::hash_map::DefaultHasher;
use std::ffi::c_void;
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::sync::{OnceLock, Weak};

use cubecl_runtime::server::Handle;
use cudarc::cublas::sys as cublas;
use cudarc::cublaslt::sys as cublaslt;

use super::batched_gemm::{BatchedGemmLaunch, BatchedGemmScalar};
use super::error::provider_status;
use super::interop::cuda_device_ptr_from_addr;
use super::plan_cache::LruPlanCache;
use super::runtime::CudaRuntimeState;
use super::CudaRuntime;

const OP: &str = "dot_general";
const PLAN_CACHE_ENTRIES: NonZeroUsize = match NonZeroUsize::new(64) {
    Some(entries) => entries,
    None => NonZeroUsize::MIN,
};
/// The largest workspace a heuristic may request for this fast path. A plan
/// that wants more scratch is still a valid batched kernel, but the
/// retained-memory cost belongs to the general path rather than a small-matrix
/// latency fix.
const MAX_WORKSPACE_BYTES: usize = 4 << 20;

#[cfg(test)]
thread_local! {
    static LT_GEMM_LAUNCHES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn lt_gemm_launches_for_test() -> usize {
    LT_GEMM_LAUNCHES.with(std::cell::Cell::get)
}

#[cfg(test)]
pub(crate) fn reset_lt_gemm_launches_for_test() {
    LT_GEMM_LAUNCHES.with(|launches| launches.set(0));
}

/// Report whether the cuBLASLt shared library can be dynamically loaded.
pub(super) fn cublaslt_library_present() -> bool {
    static PRESENT: OnceLock<bool> = OnceLock::new();
    // SAFETY: `is_culib_present` only probes candidate library names and does not
    // call cuBLASLt function pointers or retain a library handle.
    *PRESENT.get_or_init(|| unsafe { cublaslt::is_culib_present() })
}

fn lt_ok(status: cublaslt::cublasStatus_t) -> bool {
    matches!(status, cublaslt::cublasStatus_t::CUBLAS_STATUS_SUCCESS)
}

fn check_lt(
    op: &'static str,
    call: &'static str,
    status: cublaslt::cublasStatus_t,
) -> crate::Result<()> {
    if lt_ok(status) {
        Ok(())
    } else {
        Err(provider_status(op, "cuBLASLt", call, status as i32))
    }
}

/// One geometry's cuBLASLt descriptors and heuristic-chosen algorithm.
///
/// The descriptor handles are independent of the cuBLASLt handle; the plan is
/// still cached per stream slot because the algorithm and workspace belong to
/// that handle and stream.
struct LtPlan {
    a_layout: LtMatrixLayout,
    b_layout: LtMatrixLayout,
    c_layout: LtMatrixLayout,
    matmul_desc: LtMatmulDesc,
    algo: cublaslt::cublasLtMatmulAlgo_t,
    workspace: LtWorkspace,
}

impl LtPlan {
    fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>().saturating_add(self.workspace.size)
    }

    /// Build the descriptors, ask cuBLASLt for one algorithm, and allocate its
    /// workspace. `Ok(None)` means this geometry is not supported by the
    /// available cuBLASLt algorithms, which leaves the caller on cuBLAS/cuTENSOR.
    fn new<T: BatchedGemmScalar>(
        handle: cublaslt::cublasLtHandle_t,
        key: &LtPlanKey,
        rt: &CudaRuntime,
    ) -> crate::Result<Option<Self>> {
        let (a_rows, a_cols) = if key.trans_a {
            (key.k, key.m)
        } else {
            (key.m, key.k)
        };
        let (b_rows, b_cols) = if key.trans_b {
            (key.n, key.k)
        } else {
            (key.k, key.n)
        };
        let Some(a_layout) = LtMatrixLayout::new::<T>(a_rows, a_cols, key.lda)? else {
            return Ok(None);
        };
        let Some(b_layout) = LtMatrixLayout::new::<T>(b_rows, b_cols, key.ldb)? else {
            return Ok(None);
        };
        let Some(c_layout) = LtMatrixLayout::new::<T>(key.m, key.n, key.ldc)? else {
            return Ok(None);
        };
        a_layout.set_batch(key.batch_count, key.stride_a)?;
        b_layout.set_batch(key.batch_count, key.stride_b)?;
        c_layout.set_batch(key.batch_count, key.stride_c)?;

        let Some(matmul_desc) = LtMatmulDesc::new::<T>()? else {
            return Ok(None);
        };
        matmul_desc.set_transpose(
            key.trans_a,
            cublaslt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
        )?;
        matmul_desc.set_transpose(
            key.trans_b,
            cublaslt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
        )?;

        let Some(preference) = LtPreference::new()? else {
            return Ok(None);
        };
        let max_workspace = MAX_WORKSPACE_BYTES;
        if !lt_ok(unsafe {
            cublaslt::cublasLtMatmulPreferenceSetAttribute(
                preference.0,
                cublaslt::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                (&max_workspace as *const usize).cast(),
                std::mem::size_of::<usize>(),
            )
        }) {
            return Ok(None);
        }

        let mut heuristic =
            std::mem::MaybeUninit::<cublaslt::cublasLtMatmulHeuristicResult_t>::uninit();
        let mut returned = 0i32;
        let status = unsafe {
            cublaslt::cublasLtMatmulAlgoGetHeuristic(
                handle,
                matmul_desc.0,
                a_layout.0,
                b_layout.0,
                c_layout.0,
                c_layout.0,
                preference.0,
                1,
                heuristic.as_mut_ptr(),
                &mut returned,
            )
        };
        if !lt_ok(status) || returned == 0 {
            return Ok(None);
        }
        // SAFETY: the call succeeded and reported at least one initialized
        // heuristic result.
        let heuristic = unsafe { heuristic.assume_init() };
        if !lt_ok(heuristic.state) {
            return Ok(None);
        }
        let workspace = LtWorkspace::new(rt, heuristic.workspaceSize)?;

        Ok(Some(Self {
            a_layout,
            b_layout,
            c_layout,
            matmul_desc,
            algo: heuristic.algo,
            workspace,
        }))
    }
}

struct LtMatrixLayout(cublaslt::cublasLtMatrixLayout_t);

impl LtMatrixLayout {
    fn new<T: BatchedGemmScalar>(rows: i32, cols: i32, ld: i32) -> crate::Result<Option<Self>> {
        let Ok(rows) = u64::try_from(rows) else {
            return Ok(None);
        };
        let Ok(cols) = u64::try_from(cols) else {
            return Ok(None);
        };
        let mut layout = std::ptr::null_mut();
        let status = unsafe {
            cublaslt::cublasLtMatrixLayoutCreate(
                &mut layout,
                T::LT_DATA_TYPE,
                rows,
                cols,
                i64::from(ld),
            )
        };
        if !lt_ok(status) {
            return Ok(None);
        }
        Ok(Some(Self(layout)))
    }

    fn set_batch(&self, batch_count: i32, stride: i64) -> crate::Result<()> {
        check_lt(OP, "cublasLtMatrixLayoutSetAttribute", unsafe {
            cublaslt::cublasLtMatrixLayoutSetAttribute(
                self.0,
                cublaslt::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                (&batch_count as *const i32).cast(),
                std::mem::size_of::<i32>(),
            )
        })?;
        check_lt(OP, "cublasLtMatrixLayoutSetAttribute", unsafe {
            cublaslt::cublasLtMatrixLayoutSetAttribute(
                self.0,
                cublaslt::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                (&stride as *const i64).cast(),
                std::mem::size_of::<i64>(),
            )
        })
    }
}

impl Drop for LtMatrixLayout {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: this wrapper owns exactly one live layout.
            let _ = unsafe { cublaslt::cublasLtMatrixLayoutDestroy(self.0) };
        }
    }
}

struct LtMatmulDesc(cublaslt::cublasLtMatmulDesc_t);

impl LtMatmulDesc {
    fn new<T: BatchedGemmScalar>() -> crate::Result<Option<Self>> {
        let mut desc = std::ptr::null_mut();
        let status = unsafe {
            cublaslt::cublasLtMatmulDescCreate(&mut desc, T::LT_COMPUTE_TYPE, T::LT_SCALE_TYPE)
        };
        if !lt_ok(status) {
            return Ok(None);
        }
        Ok(Some(Self(desc)))
    }

    fn set_transpose(
        &self,
        transposed: bool,
        attr: cublaslt::cublasLtMatmulDescAttributes_t,
    ) -> crate::Result<()> {
        let op = if transposed {
            cublas::cublasOperation_t::CUBLAS_OP_T
        } else {
            cublas::cublasOperation_t::CUBLAS_OP_N
        };
        check_lt(OP, "cublasLtMatmulDescSetAttribute", unsafe {
            cublaslt::cublasLtMatmulDescSetAttribute(
                self.0,
                attr,
                (&op as *const cublas::cublasOperation_t).cast(),
                std::mem::size_of::<cublas::cublasOperation_t>(),
            )
        })
    }
}

impl Drop for LtMatmulDesc {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: this wrapper owns exactly one live descriptor.
            let _ = unsafe { cublaslt::cublasLtMatmulDescDestroy(self.0) };
        }
    }
}

struct LtPreference(cublaslt::cublasLtMatmulPreference_t);

impl LtPreference {
    fn new() -> crate::Result<Option<Self>> {
        let mut pref = std::ptr::null_mut();
        let status = unsafe { cublaslt::cublasLtMatmulPreferenceCreate(&mut pref) };
        if !lt_ok(status) {
            return Ok(None);
        }
        Ok(Some(Self(pref)))
    }
}

impl Drop for LtPreference {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: this wrapper owns exactly one live preference object.
            let _ = unsafe { cublaslt::cublasLtMatmulPreferenceDestroy(self.0) };
        }
    }
}

/// Device scratch retained by one cached plan.
struct LtWorkspace {
    handle: Option<Handle>,
    ptr: *mut c_void,
    size: usize,
    stream: u64,
    runtime: Weak<CudaRuntimeState>,
}

impl LtWorkspace {
    fn none() -> Self {
        Self {
            handle: None,
            ptr: std::ptr::null_mut(),
            size: 0,
            stream: 0,
            runtime: Weak::new(),
        }
    }

    fn new(rt: &CudaRuntime, size: usize) -> crate::Result<Self> {
        if size == 0 {
            return Ok(Self::none());
        }
        let handle = rt.client().empty(size);
        let resource = rt
            .client()
            .get_resource::<cubecl_cuda::ffi_interop::CudaServer>(handle.clone())
            .map_err(|err| crate::Error::backend_source(OP, err))?;
        let stream = rt.raw_cuda_stream()?;
        Ok(Self {
            handle: Some(handle),
            ptr: cuda_device_ptr_from_addr(resource.resource().ptr, OP)?,
            size,
            stream,
            runtime: rt.weak_state(),
        })
    }
}

impl Drop for LtWorkspace {
    fn drop(&mut self) {
        let Some(handle) = self.handle.take() else {
            return;
        };
        let Some(runtime) = self.runtime.upgrade() else {
            // Runtime teardown has already retired every stream before this
            // state is dropped, so the handle can return to the pool directly.
            drop(handle);
            return;
        };
        let mut retirements = runtime
            .workspace_retirements()
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        retirements.retire(&runtime, self.stream, handle);
    }
}

/// Dtype plus every GEMM parameter that changes the descriptors or algorithm.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct LtPlanKey {
    dtype: tenferro_tensor::DType,
    trans_a: bool,
    trans_b: bool,
    m: i32,
    n: i32,
    k: i32,
    lda: i32,
    ldb: i32,
    ldc: i32,
    stride_a: i64,
    stride_b: i64,
    stride_c: i64,
    batch_count: i32,
}

impl LtPlanKey {
    fn new<T: BatchedGemmScalar>(args: &BatchedGemmLaunch) -> Self {
        Self {
            dtype: T::dtype(),
            trans_a: args.trans_a == cublas::cublasOperation_t::CUBLAS_OP_T,
            trans_b: args.trans_b == cublas::cublasOperation_t::CUBLAS_OP_T,
            m: args.m,
            n: args.n,
            k: args.k,
            lda: args.lda,
            ldb: args.ldb,
            ldc: args.ldc,
            stride_a: args.stride_a,
            stride_b: args.stride_b,
            stride_c: args.stride_c,
            batch_count: args.batch_count,
        }
    }
}

fn plan_hash(key: &LtPlanKey) -> u64 {
    let mut hasher = DefaultHasher::new();
    key.hash(&mut hasher);
    hasher.finish()
}

/// One stream slot's cuBLASLt handle and its geometry-keyed plan cache.
pub(super) struct CudaLtStreamState {
    handle: cublaslt::cublasLtHandle_t,
    plans: LruPlanCache<LtPlanKey, Option<LtPlan>>,
}

impl CudaLtStreamState {
    /// Create the handle for this slot, or `None` when cuBLASLt cannot be used
    /// and the caller should stay on the cuBLAS strided-batched path.
    pub(super) fn new() -> Option<Self> {
        let mut handle = std::ptr::null_mut();
        let status = unsafe { cublaslt::cublasLtCreate(&mut handle) };
        if !lt_ok(status) {
            return None;
        }
        Some(Self {
            handle,
            plans: LruPlanCache::new(PLAN_CACHE_ENTRIES),
        })
    }

    /// Enqueue `C = op(A) * op(B)` for every batch slice, or answer `None` when
    /// no cuBLASLt plan covers this geometry.
    pub(super) fn try_launch<T: BatchedGemmScalar>(
        &mut self,
        rt: &CudaRuntime,
        args: &BatchedGemmLaunch,
        a: *const c_void,
        b: *const c_void,
        c: *mut c_void,
    ) -> crate::Result<Option<()>> {
        let key = LtPlanKey::new::<T>(args);
        let hash = plan_hash(&key);
        let handle = self.handle;
        self.plans.ensure(
            hash,
            |stored| stored == &key,
            || {
                let plan = LtPlan::new::<T>(handle, &key, rt)?;
                let retained = plan.as_ref().map_or(0, LtPlan::retained_bytes);
                Ok((key, plan, retained))
            },
        )?;
        let Some(plan) = self
            .plans
            .get(hash, |stored| stored == &key)
            .and_then(Option::as_ref)
        else {
            return Ok(None);
        };
        let (alpha, beta) = T::lt_alpha_beta();
        let stream = rt.raw_cuda_stream()? as usize as cublaslt::cudaStream_t;
        // SAFETY: the pointers were resolved from live allocations on this
        // runtime; the plan's descriptors, algorithm, and workspace belong to
        // this handle and stream; alpha/beta are stack locals live for the call.
        let status = unsafe {
            cublaslt::cublasLtMatmul(
                self.handle,
                plan.matmul_desc.0,
                (&alpha as *const T).cast(),
                a,
                plan.a_layout.0,
                b,
                plan.b_layout.0,
                (&beta as *const T).cast(),
                c,
                plan.c_layout.0,
                c,
                plan.c_layout.0,
                &plan.algo,
                plan.workspace.ptr,
                plan.workspace.size,
                stream,
            )
        };
        check_lt(OP, "cublasLtMatmul", status)?;
        #[cfg(test)]
        LT_GEMM_LAUNCHES.with(|launches| launches.set(launches.get() + 1));
        Ok(Some(()))
    }

    /// Destroy the cached plans and the handle. Called while the runtime's
    /// streams have already retired, so workspaces release without a barrier.
    pub(super) fn release(&mut self) {
        self.plans.clear();
        if !self.handle.is_null() {
            // SAFETY: this state owns exactly one live cuBLASLt handle.
            let _ = unsafe { cublaslt::cublasLtDestroy(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}

impl Drop for CudaLtStreamState {
    fn drop(&mut self) {
        self.release();
    }
}
