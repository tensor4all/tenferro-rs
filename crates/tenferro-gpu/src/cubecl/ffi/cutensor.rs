use std::ffi::{c_void, CStr};
use std::os::raw::c_char;
use std::sync::Arc;

use libloading::Library;

const OP: &str = "cuda_cutensor";
/// Default cuTENSOR search paths. Override with `TENFERRO_CUTENSOR_PATH` env var.
const CUTENSOR_DEFAULT_PATHS: &[&str] = &[
    "/usr/lib/x86_64-linux-gnu/libcutensor/12/libcutensor.so.2",
    "libcutensor.so.2",
    "libcutensor.so",
];

pub(crate) type CutensorHandleRaw = *mut c_void;
pub(crate) type CutensorTensorDescriptorRaw = *mut c_void;
pub(crate) type CutensorOperationDescriptorRaw = *mut c_void;
pub(crate) type CutensorPlanPreferenceRaw = *mut c_void;
pub(crate) type CutensorPlanRaw = *mut c_void;
pub(crate) type CutensorComputeDescriptor = *const c_void;
pub(crate) type CutensorCudaStream = *mut c_void;
type CutensorStatus = i32;

const CUTENSOR_STATUS_SUCCESS: CutensorStatus = 0;

#[derive(Debug, thiserror::Error)]
enum CutensorLoadError {
    #[error("failed to load cuTENSOR symbol {name}: {source}")]
    Symbol {
        name: String,
        #[source]
        source: libloading::Error,
    },
    #[error("failed to load cuTENSOR library (tried {paths}): {source}; attempts: {attempts}")]
    Library {
        paths: String,
        attempts: String,
        #[source]
        source: libloading::Error,
    },
}

#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum CudaDataType {
    R32F = 0,
    R64F = 1,
    C32F = 4,
    C64F = 5,
}

#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum CutensorOperator {
    Identity = 1,
    Conj = 9,
}

#[repr(i32)]
#[derive(Clone, Copy)]
pub(crate) enum CutensorAlgo {
    Default = -1,
}

#[repr(i32)]
#[derive(Clone, Copy)]
pub(crate) enum CutensorJitMode {
    None = 0,
}

#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum CutensorWorksizePreference {
    Default = 2,
}

type CutensorCreateFn = unsafe extern "C" fn(*mut CutensorHandleRaw) -> CutensorStatus;
type CutensorDestroyFn = unsafe extern "C" fn(CutensorHandleRaw) -> CutensorStatus;
type CutensorCreateTensorDescriptorFn = unsafe extern "C" fn(
    CutensorHandleRaw,
    *mut CutensorTensorDescriptorRaw,
    u32,
    *const i64,
    *const i64,
    CudaDataType,
    u32,
) -> CutensorStatus;
type CutensorDestroyTensorDescriptorFn =
    unsafe extern "C" fn(CutensorTensorDescriptorRaw) -> CutensorStatus;
type CutensorCreateContractionFn = unsafe extern "C" fn(
    CutensorHandleRaw,
    *mut CutensorOperationDescriptorRaw,
    CutensorTensorDescriptorRaw,
    *const i32,
    CutensorOperator,
    CutensorTensorDescriptorRaw,
    *const i32,
    CutensorOperator,
    CutensorTensorDescriptorRaw,
    *const i32,
    CutensorOperator,
    CutensorTensorDescriptorRaw,
    *const i32,
    CutensorComputeDescriptor,
) -> CutensorStatus;
type CutensorCreatePermutationFn = unsafe extern "C" fn(
    CutensorHandleRaw,
    *mut CutensorOperationDescriptorRaw,
    CutensorTensorDescriptorRaw,
    *const i32,
    CutensorOperator,
    CutensorTensorDescriptorRaw,
    *const i32,
    CutensorComputeDescriptor,
) -> CutensorStatus;
type CutensorDestroyOperationDescriptorFn =
    unsafe extern "C" fn(CutensorOperationDescriptorRaw) -> CutensorStatus;
type CutensorCreatePlanPreferenceFn = unsafe extern "C" fn(
    CutensorHandleRaw,
    *mut CutensorPlanPreferenceRaw,
    CutensorAlgo,
    CutensorJitMode,
) -> CutensorStatus;
type CutensorDestroyPlanPreferenceFn =
    unsafe extern "C" fn(CutensorPlanPreferenceRaw) -> CutensorStatus;
type CutensorEstimateWorkspaceSizeFn = unsafe extern "C" fn(
    CutensorHandleRaw,
    CutensorOperationDescriptorRaw,
    CutensorPlanPreferenceRaw,
    CutensorWorksizePreference,
    *mut u64,
) -> CutensorStatus;
type CutensorCreatePlanFn = unsafe extern "C" fn(
    CutensorHandleRaw,
    *mut CutensorPlanRaw,
    CutensorOperationDescriptorRaw,
    CutensorPlanPreferenceRaw,
    u64,
) -> CutensorStatus;
type CutensorDestroyPlanFn = unsafe extern "C" fn(CutensorPlanRaw) -> CutensorStatus;
type CutensorContractFn = unsafe extern "C" fn(
    CutensorHandleRaw,
    CutensorPlanRaw,
    *const c_void,
    *const c_void,
    *const c_void,
    *const c_void,
    *const c_void,
    *mut c_void,
    *mut c_void,
    u64,
    CutensorCudaStream,
) -> CutensorStatus;
type CutensorPermuteFn = unsafe extern "C" fn(
    CutensorHandleRaw,
    CutensorPlanRaw,
    *const c_void,
    *const c_void,
    *mut c_void,
    CutensorCudaStream,
) -> CutensorStatus;
type CutensorGetErrorStringFn = unsafe extern "C" fn(CutensorStatus) -> *const c_char;

struct CutensorVtable {
    create: CutensorCreateFn,
    destroy: CutensorDestroyFn,
    create_tensor_descriptor: CutensorCreateTensorDescriptorFn,
    destroy_tensor_descriptor: CutensorDestroyTensorDescriptorFn,
    create_contraction: CutensorCreateContractionFn,
    create_permutation: CutensorCreatePermutationFn,
    destroy_operation_descriptor: CutensorDestroyOperationDescriptorFn,
    create_plan_preference: CutensorCreatePlanPreferenceFn,
    destroy_plan_preference: CutensorDestroyPlanPreferenceFn,
    estimate_workspace_size: CutensorEstimateWorkspaceSizeFn,
    create_plan: CutensorCreatePlanFn,
    destroy_plan: CutensorDestroyPlanFn,
    contract: CutensorContractFn,
    permute: CutensorPermuteFn,
    get_error_string: CutensorGetErrorStringFn,
    compute_desc_32f: CutensorComputeDescriptor,
    compute_desc_64f: CutensorComputeDescriptor,
}

impl CutensorVtable {
    unsafe fn load(lib: &Library) -> crate::Result<Self> {
        Ok(Self {
            create: load_symbol(lib, b"cutensorCreate\0")?,
            destroy: load_symbol(lib, b"cutensorDestroy\0")?,
            create_tensor_descriptor: load_symbol(lib, b"cutensorCreateTensorDescriptor\0")?,
            destroy_tensor_descriptor: load_symbol(lib, b"cutensorDestroyTensorDescriptor\0")?,
            create_contraction: load_symbol(lib, b"cutensorCreateContraction\0")?,
            create_permutation: load_symbol(lib, b"cutensorCreatePermutation\0")?,
            destroy_operation_descriptor: load_symbol(
                lib,
                b"cutensorDestroyOperationDescriptor\0",
            )?,
            create_plan_preference: load_symbol(lib, b"cutensorCreatePlanPreference\0")?,
            destroy_plan_preference: load_symbol(lib, b"cutensorDestroyPlanPreference\0")?,
            estimate_workspace_size: load_symbol(lib, b"cutensorEstimateWorkspaceSize\0")?,
            create_plan: load_symbol(lib, b"cutensorCreatePlan\0")?,
            destroy_plan: load_symbol(lib, b"cutensorDestroyPlan\0")?,
            contract: load_symbol(lib, b"cutensorContract\0")?,
            permute: load_symbol(lib, b"cutensorPermute\0")?,
            get_error_string: load_symbol(lib, b"cutensorGetErrorString\0")?,
            compute_desc_32f: load_data_symbol(lib, b"CUTENSOR_COMPUTE_DESC_32F\0")?,
            compute_desc_64f: load_data_symbol(lib, b"CUTENSOR_COMPUTE_DESC_64F\0")?,
        })
    }
}

unsafe fn load_symbol<T: Copy>(lib: &Library, name: &[u8]) -> crate::Result<T> {
    // SAFETY: `name` is a NUL-terminated cuTENSOR symbol and the requested
    // type `T` is the exact FFI function-pointer type declared in this module.
    let symbol_name = String::from_utf8_lossy(name)
        .trim_end_matches('\0')
        .to_owned();
    let symbol = lib.get::<T>(name).map_err(|source| {
        crate::Error::io_source(
            OP,
            CutensorLoadError::Symbol {
                name: symbol_name,
                source,
            },
        )
    })?;
    Ok(*symbol)
}

unsafe fn load_data_symbol<T: Copy>(lib: &Library, name: &[u8]) -> crate::Result<T> {
    let symbol_name = String::from_utf8_lossy(name)
        .trim_end_matches('\0')
        .to_owned();
    // SAFETY: cuTENSOR compute descriptors are exported as static data symbols
    // whose value is a process-lifetime pointer-like descriptor.
    let symbol = lib.get::<*const T>(name).map_err(|source| {
        crate::Error::io_source(
            OP,
            CutensorLoadError::Symbol {
                name: symbol_name.clone(),
                source,
            },
        )
    })?;
    let ptr = *symbol;
    if ptr.is_null() {
        return Err(crate::Error::Internal(format!(
            "cuTENSOR data symbol {symbol_name} resolved to a null pointer"
        )));
    }
    if !ptr.is_aligned() {
        return Err(crate::Error::Internal(format!(
            "cuTENSOR data symbol {symbol_name} resolved to a misaligned pointer"
        )));
    }
    // SAFETY: null and alignment were checked above; cuTENSOR exports this as
    // static process-lifetime data and `T: Copy`, so reading copies the descriptor value.
    Ok(unsafe { std::ptr::read(ptr) })
}

struct CutensorLibrary {
    _lib: Library,
    vtable: CutensorVtable,
}

// SAFETY: `CutensorLibrary` keeps the dynamic library alive for the copied
// process-wide function/data symbols, and cuTENSOR entry points are documented
// as thread-safe when each call receives its explicit handle/descriptor state.
unsafe impl Send for CutensorLibrary {}
// SAFETY: See the `Send` rationale; shared references only expose immutable
// vtable/data-symbol access and call into cuTENSOR with explicit raw handles.
unsafe impl Sync for CutensorLibrary {}

impl CutensorLibrary {
    fn load() -> crate::Result<Arc<Self>> {
        let paths = super::library_search_paths("TENFERRO_CUTENSOR_PATH", CUTENSOR_DEFAULT_PATHS);
        Self::load_from_paths(paths)
    }

    fn load_from_paths(paths: Vec<String>) -> crate::Result<Arc<Self>> {
        let mut errors = Vec::new();
        let mut last_source = None;
        for path in &paths {
            let lib = match unsafe { Library::new(path) } {
                Ok(lib) => lib,
                Err(err) => {
                    errors.push(format!("{path}: {err}"));
                    last_source = Some(err);
                    continue;
                }
            };
            let vtable = unsafe { CutensorVtable::load(&lib) }?;
            return Ok(Arc::new(Self { _lib: lib, vtable }));
        }

        match last_source {
            Some(source) => Err(crate::Error::io_source(
                OP,
                CutensorLoadError::Library {
                    paths: paths.join(", "),
                    attempts: errors.join("; "),
                    source,
                },
            )),
            None => Err(crate::Error::io_source(
                OP,
                std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("no cuTENSOR library search paths configured for {OP}"),
                ),
            )),
        }
    }

    fn status_message(&self, status: CutensorStatus) -> String {
        // SAFETY: `cutensorGetErrorString` returns either null or a stable
        // NUL-terminated string owned by the loaded cuTENSOR library.
        let ptr = unsafe { (self.vtable.get_error_string)(status) };
        if ptr.is_null() {
            return format!("status code {status}");
        }
        // SAFETY: Null was checked above and cuTENSOR owns a valid
        // NUL-terminated status string for the library lifetime.
        unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned()
    }

    fn check_status(
        &self,
        status: CutensorStatus,
        op: &'static str,
        call: &'static str,
    ) -> crate::Result<()> {
        if status == CUTENSOR_STATUS_SUCCESS {
            return Ok(());
        }
        Err(super::super::error::provider_status(
            op, "cuTENSOR", call, status,
        ))
    }
}

#[cfg(test)]
// INVARIANT: the loader contract test is kept beside the private FFI loader;
// the production handle types intentionally remain below this test module.
#[allow(clippy::items_after_test_module)]
mod tests {
    #[test]
    fn cutensor_loader_missing_library_is_typed_io_error() {
        let err = match super::CutensorLibrary::load_from_paths(vec![
            "/definitely/not/libcutensor.so".to_string(),
        ]) {
            Ok(_) => panic!("unexpectedly loaded cuTENSOR from a nonexistent path"),
            Err(err) => err,
        };

        match err {
            crate::Error::IoSource { op, source } => {
                assert_eq!(op, "cuda_cutensor");
                assert!(source
                    .to_string()
                    .contains("failed to load cuTENSOR library"));
            }
            other => panic!("expected typed cuTENSOR IoSource, got {other:?}"),
        }
    }
}

#[cold]
fn report_cutensor_destroy_status(
    lib: &CutensorLibrary,
    status: CutensorStatus,
    call: &'static str,
) {
    if status != CUTENSOR_STATUS_SUCCESS {
        eprintln!(
            "tenferro-gpu: {call} failed during Drop with cuTENSOR {} ({status})",
            lib.status_message(status)
        );
    }
}

pub(crate) struct CutensorHandle {
    lib: Arc<CutensorLibrary>,
    raw: CutensorHandleRaw,
}

// SAFETY: The handle is an opaque cuTENSOR handle used through FFI calls that
// take `&self`; higher-level backend access serializes mutable use.
unsafe impl Send for CutensorHandle {}
// SAFETY: cuTENSOR calls receive explicit handle, descriptor, plan, workspace,
// and stream arguments. The loaded library vtable is immutable, and callers pass
// operation-local descriptor/plan values for each launch.
unsafe impl Sync for CutensorHandle {}

impl CutensorHandle {
    pub(crate) fn load() -> crate::Result<Self> {
        let lib = CutensorLibrary::load()?;
        let mut raw = std::ptr::null_mut();
        let status = unsafe { (lib.vtable.create)(&mut raw) };
        lib.check_status(status, OP, "cutensorCreate")?;
        Ok(Self { lib, raw })
    }

    pub(crate) fn compute_desc_32f(&self) -> CutensorComputeDescriptor {
        self.lib.vtable.compute_desc_32f
    }

    pub(crate) fn compute_desc_64f(&self) -> CutensorComputeDescriptor {
        self.lib.vtable.compute_desc_64f
    }

    fn as_raw(&self) -> CutensorHandleRaw {
        self.raw
    }

    pub(crate) fn estimate_workspace_size(
        &self,
        desc: &OperationDescriptor,
        pref: &PlanPreference,
        workspace_pref: CutensorWorksizePreference,
        op: &'static str,
    ) -> crate::Result<u64> {
        let mut workspace_size = 0u64;
        let status = unsafe {
            (self.lib.vtable.estimate_workspace_size)(
                self.as_raw(),
                desc.as_raw(),
                pref.as_raw(),
                workspace_pref,
                &mut workspace_size,
            )
        };
        self.lib
            .check_status(status, op, "cutensorEstimateWorkspaceSize")?;
        Ok(workspace_size)
    }

    // INVARIANT: cuTENSOR's C ABI fixes this argument list; keeping it
    // explicit preserves the provider call order at the unsafe boundary.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn contract(
        &self,
        plan: &Plan,
        alpha: *const c_void,
        a: *const c_void,
        b: *const c_void,
        beta: *const c_void,
        c: *const c_void,
        d: *mut c_void,
        workspace: *mut c_void,
        workspace_size: u64,
        stream: CutensorCudaStream,
        op: &'static str,
    ) -> crate::Result<()> {
        let status = (self.lib.vtable.contract)(
            self.as_raw(),
            plan.as_raw(),
            alpha,
            a,
            b,
            beta,
            c,
            d,
            workspace,
            workspace_size,
            stream,
        );
        self.lib.check_status(status, op, "cutensorContract")
    }

    pub(crate) unsafe fn permute(
        &self,
        plan: &Plan,
        alpha: *const c_void,
        a: *const c_void,
        b: *mut c_void,
        stream: CutensorCudaStream,
        op: &'static str,
    ) -> crate::Result<()> {
        let status = (self.lib.vtable.permute)(self.as_raw(), plan.as_raw(), alpha, a, b, stream);
        self.lib.check_status(status, op, "cutensorPermute")
    }
}

impl Drop for CutensorHandle {
    fn drop(&mut self) {
        let status = unsafe { (self.lib.vtable.destroy)(self.raw) };
        report_cutensor_destroy_status(&self.lib, status, "cutensorDestroy");
    }
}

pub(crate) struct TensorDescriptor {
    lib: Arc<CutensorLibrary>,
    raw: CutensorTensorDescriptorRaw,
}

impl TensorDescriptor {
    pub(crate) fn new(
        handle: &CutensorHandle,
        extents: &[i64],
        strides: &[i64],
        data_type: CudaDataType,
        alignment_requirement: u32,
        op: &'static str,
    ) -> crate::Result<Self> {
        let num_modes = u32::try_from(extents.len()).map_err(|_| {
            crate::Error::invalid_argument(op, "rank", "tensor rank exceeds cuTENSOR u32 limit")
        })?;
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            (handle.lib.vtable.create_tensor_descriptor)(
                handle.as_raw(),
                &mut raw,
                num_modes,
                extents.as_ptr(),
                strides.as_ptr(),
                data_type,
                alignment_requirement,
            )
        };
        handle
            .lib
            .check_status(status, op, "cutensorCreateTensorDescriptor")?;
        Ok(Self {
            lib: Arc::clone(&handle.lib),
            raw,
        })
    }

    fn as_raw(&self) -> CutensorTensorDescriptorRaw {
        self.raw
    }
}

impl Drop for TensorDescriptor {
    fn drop(&mut self) {
        let status = unsafe { (self.lib.vtable.destroy_tensor_descriptor)(self.raw) };
        report_cutensor_destroy_status(&self.lib, status, "cutensorDestroyTensorDescriptor");
    }
}

pub(crate) struct OperationDescriptor {
    lib: Arc<CutensorLibrary>,
    raw: CutensorOperationDescriptorRaw,
}

impl OperationDescriptor {
    // INVARIANT: these arguments map one-for-one to cuTENSOR's C descriptor
    // API; a parameter object would obscure the FFI order.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_contraction_with_ops(
        handle: &CutensorHandle,
        desc_a: &TensorDescriptor,
        mode_a: &[i32],
        op_a: CutensorOperator,
        desc_b: &TensorDescriptor,
        mode_b: &[i32],
        op_b: CutensorOperator,
        desc_c: &TensorDescriptor,
        mode_c: &[i32],
        desc_d: &TensorDescriptor,
        mode_d: &[i32],
        compute_desc: CutensorComputeDescriptor,
        op: &'static str,
    ) -> crate::Result<Self> {
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            (handle.lib.vtable.create_contraction)(
                handle.as_raw(),
                &mut raw,
                desc_a.as_raw(),
                mode_a.as_ptr(),
                op_a,
                desc_b.as_raw(),
                mode_b.as_ptr(),
                op_b,
                desc_c.as_raw(),
                mode_c.as_ptr(),
                CutensorOperator::Identity,
                desc_d.as_raw(),
                mode_d.as_ptr(),
                compute_desc,
            )
        };
        handle
            .lib
            .check_status(status, op, "cutensorCreateContraction")?;
        Ok(Self {
            lib: Arc::clone(&handle.lib),
            raw,
        })
    }

    // INVARIANT: these arguments map one-for-one to cuTENSOR's C descriptor
    // API; a parameter object would obscure the FFI order.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_permutation(
        handle: &CutensorHandle,
        desc_a: &TensorDescriptor,
        mode_a: &[i32],
        op_a: CutensorOperator,
        desc_b: &TensorDescriptor,
        mode_b: &[i32],
        compute_desc: CutensorComputeDescriptor,
        op: &'static str,
    ) -> crate::Result<Self> {
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            (handle.lib.vtable.create_permutation)(
                handle.as_raw(),
                &mut raw,
                desc_a.as_raw(),
                mode_a.as_ptr(),
                op_a,
                desc_b.as_raw(),
                mode_b.as_ptr(),
                compute_desc,
            )
        };
        handle
            .lib
            .check_status(status, op, "cutensorCreatePermutation")?;
        Ok(Self {
            lib: Arc::clone(&handle.lib),
            raw,
        })
    }

    fn as_raw(&self) -> CutensorOperationDescriptorRaw {
        self.raw
    }
}

impl Drop for OperationDescriptor {
    fn drop(&mut self) {
        let status = unsafe { (self.lib.vtable.destroy_operation_descriptor)(self.raw) };
        report_cutensor_destroy_status(&self.lib, status, "cutensorDestroyOperationDescriptor");
    }
}

pub(crate) struct PlanPreference {
    lib: Arc<CutensorLibrary>,
    raw: CutensorPlanPreferenceRaw,
}

impl PlanPreference {
    pub(crate) fn new_default(handle: &CutensorHandle, op: &'static str) -> crate::Result<Self> {
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            (handle.lib.vtable.create_plan_preference)(
                handle.as_raw(),
                &mut raw,
                CutensorAlgo::Default,
                CutensorJitMode::None,
            )
        };
        handle
            .lib
            .check_status(status, op, "cutensorCreatePlanPreference")?;
        Ok(Self {
            lib: Arc::clone(&handle.lib),
            raw,
        })
    }

    fn as_raw(&self) -> CutensorPlanPreferenceRaw {
        self.raw
    }
}

impl Drop for PlanPreference {
    fn drop(&mut self) {
        let status = unsafe { (self.lib.vtable.destroy_plan_preference)(self.raw) };
        report_cutensor_destroy_status(&self.lib, status, "cutensorDestroyPlanPreference");
    }
}

pub(crate) struct Plan {
    lib: Arc<CutensorLibrary>,
    raw: CutensorPlanRaw,
}

impl Plan {
    pub(crate) fn new(
        handle: &CutensorHandle,
        desc: &OperationDescriptor,
        pref: &PlanPreference,
        workspace_limit: u64,
        op: &'static str,
    ) -> crate::Result<Self> {
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            (handle.lib.vtable.create_plan)(
                handle.as_raw(),
                &mut raw,
                desc.as_raw(),
                pref.as_raw(),
                workspace_limit,
            )
        };
        handle.lib.check_status(status, op, "cutensorCreatePlan")?;
        Ok(Self {
            lib: Arc::clone(&handle.lib),
            raw,
        })
    }

    fn as_raw(&self) -> CutensorPlanRaw {
        self.raw
    }
}

impl Drop for Plan {
    fn drop(&mut self) {
        let status = unsafe { (self.lib.vtable.destroy_plan)(self.raw) };
        report_cutensor_destroy_status(&self.lib, status, "cutensorDestroyPlan");
    }
}
