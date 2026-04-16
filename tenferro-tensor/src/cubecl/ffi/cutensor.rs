use std::ffi::{c_void, CStr};
use std::os::raw::c_char;
use std::sync::Arc;

use libloading::Library;

const OP: &str = "dot_general";
const PRIMARY_LIBRARY_PATH: &str = "/usr/lib/x86_64-linux-gnu/libcutensor/12/libcutensor.so.2";
const FALLBACK_LIBRARY_PATH: &str = "libcutensor.so";

pub(crate) type CutensorHandleRaw = *mut c_void;
pub(crate) type CutensorTensorDescriptorRaw = *mut c_void;
pub(crate) type CutensorOperationDescriptorRaw = *mut c_void;
pub(crate) type CutensorPlanPreferenceRaw = *mut c_void;
pub(crate) type CutensorPlanRaw = *mut c_void;
pub(crate) type CutensorComputeDescriptor = *const c_void;
pub(crate) type CutensorCudaStream = *mut c_void;
type CutensorStatus = i32;

const CUTENSOR_STATUS_SUCCESS: CutensorStatus = 0;

#[repr(i32)]
#[derive(Clone, Copy)]
pub(crate) enum CudaDataType {
    R32F = 0,
    R64F = 1,
    C32F = 4,
    C64F = 5,
}

#[repr(i32)]
#[derive(Clone, Copy)]
pub(crate) enum CutensorOperator {
    Identity = 1,
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
#[derive(Clone, Copy)]
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
type CutensorGetErrorStringFn = unsafe extern "C" fn(CutensorStatus) -> *const c_char;

struct CutensorVtable {
    create: CutensorCreateFn,
    destroy: CutensorDestroyFn,
    create_tensor_descriptor: CutensorCreateTensorDescriptorFn,
    destroy_tensor_descriptor: CutensorDestroyTensorDescriptorFn,
    create_contraction: CutensorCreateContractionFn,
    destroy_operation_descriptor: CutensorDestroyOperationDescriptorFn,
    create_plan_preference: CutensorCreatePlanPreferenceFn,
    destroy_plan_preference: CutensorDestroyPlanPreferenceFn,
    estimate_workspace_size: CutensorEstimateWorkspaceSizeFn,
    create_plan: CutensorCreatePlanFn,
    destroy_plan: CutensorDestroyPlanFn,
    contract: CutensorContractFn,
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
            get_error_string: load_symbol(lib, b"cutensorGetErrorString\0")?,
            compute_desc_32f: load_data_symbol(lib, b"CUTENSOR_COMPUTE_DESC_32F\0")?,
            compute_desc_64f: load_data_symbol(lib, b"CUTENSOR_COMPUTE_DESC_64F\0")?,
        })
    }
}

unsafe fn load_symbol<T: Copy>(lib: &Library, name: &[u8]) -> crate::Result<T> {
    let symbol = lib
        .get::<T>(name)
        .map_err(|err| crate::Error::BackendFailure {
            op: OP,
            message: format!(
                "failed to load cuTENSOR symbol {}: {err}",
                String::from_utf8_lossy(name).trim_end_matches('\0')
            ),
        })?;
    Ok(*symbol)
}

unsafe fn load_data_symbol<T: Copy>(lib: &Library, name: &[u8]) -> crate::Result<T> {
    let symbol = lib
        .get::<*const T>(name)
        .map_err(|err| crate::Error::BackendFailure {
            op: OP,
            message: format!(
                "failed to load cuTENSOR data symbol {}: {err}",
                String::from_utf8_lossy(name).trim_end_matches('\0')
            ),
        })?;
    Ok(**symbol)
}

struct CutensorLibrary {
    _lib: Library,
    vtable: CutensorVtable,
}

impl CutensorLibrary {
    fn load() -> crate::Result<Arc<Self>> {
        let mut errors = Vec::new();
        for path in [PRIMARY_LIBRARY_PATH, FALLBACK_LIBRARY_PATH] {
            let lib = match unsafe { Library::new(path) } {
                Ok(lib) => lib,
                Err(err) => {
                    errors.push(format!("{path}: {err}"));
                    continue;
                }
            };
            let vtable = unsafe { CutensorVtable::load(&lib) }?;
            return Ok(Arc::new(Self { _lib: lib, vtable }));
        }

        Err(crate::Error::BackendFailure {
            op: OP,
            message: format!(
                "failed to load cuTENSOR library (tried {PRIMARY_LIBRARY_PATH}, {FALLBACK_LIBRARY_PATH}): {}",
                errors.join("; ")
            ),
        })
    }

    fn status_message(&self, status: CutensorStatus) -> String {
        let ptr = unsafe { (self.vtable.get_error_string)(status) };
        if ptr.is_null() {
            return format!("status code {status}");
        }
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
        Err(crate::Error::BackendFailure {
            op,
            message: format!(
                "{call} failed with cuTENSOR {} ({status})",
                self.status_message(status)
            ),
        })
    }
}

pub(crate) struct CutensorHandle {
    lib: Arc<CutensorLibrary>,
    raw: CutensorHandleRaw,
}

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
}

impl Drop for CutensorHandle {
    fn drop(&mut self) {
        let _ = unsafe { (self.lib.vtable.destroy)(self.raw) };
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
        let num_modes = u32::try_from(extents.len()).map_err(|_| crate::Error::BackendFailure {
            op,
            message: "tensor rank exceeds cuTENSOR u32 limit".into(),
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
        let _ = unsafe { (self.lib.vtable.destroy_tensor_descriptor)(self.raw) };
    }
}

pub(crate) struct OperationDescriptor {
    lib: Arc<CutensorLibrary>,
    raw: CutensorOperationDescriptorRaw,
}

impl OperationDescriptor {
    pub(crate) fn new_contraction(
        handle: &CutensorHandle,
        desc_a: &TensorDescriptor,
        mode_a: &[i32],
        desc_b: &TensorDescriptor,
        mode_b: &[i32],
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
                CutensorOperator::Identity,
                desc_b.as_raw(),
                mode_b.as_ptr(),
                CutensorOperator::Identity,
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

    fn as_raw(&self) -> CutensorOperationDescriptorRaw {
        self.raw
    }
}

impl Drop for OperationDescriptor {
    fn drop(&mut self) {
        let _ = unsafe { (self.lib.vtable.destroy_operation_descriptor)(self.raw) };
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
        let _ = unsafe { (self.lib.vtable.destroy_plan_preference)(self.raw) };
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
        let _ = unsafe { (self.lib.vtable.destroy_plan)(self.raw) };
    }
}
