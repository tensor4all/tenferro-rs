use std::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};

use tenferro_device::LogicalMemorySpace;
use tenferro_tensor::Tensor;

use crate::handle::{handle_take, tensor_to_handle, TfeTensorF64};
use crate::status::{
    finalize_ptr, map_device_error, set_last_error, tfe_status_t, TFE_INTERNAL_ERROR,
    TFE_INVALID_ARGUMENT,
};

/// DLPack version information.
#[repr(C)]
pub struct DLPackVersion {
    /// Major version (1 for DLPack v1.0).
    pub major: u32,
    /// Minor version.
    pub minor: u32,
}

/// DLPack device descriptor.
#[repr(C)]
pub struct DLDevice {
    /// Device type.
    pub device_type: i32,
    /// Device ID.
    pub device_id: i32,
}

/// DLPack data type descriptor.
#[repr(C)]
pub struct DLDataType {
    /// Type code.
    pub code: u8,
    /// Number of bits per element.
    pub bits: u8,
    /// Number of lanes.
    pub lanes: u16,
}

/// DLPack tensor descriptor (unmanaged).
#[repr(C)]
pub struct DLTensor {
    /// Pointer to the data.
    pub data: *mut c_void,
    /// Device where the data resides.
    pub device: DLDevice,
    /// Number of dimensions.
    pub ndim: i32,
    /// Data type.
    pub dtype: DLDataType,
    /// Shape array.
    pub shape: *mut i64,
    /// Strides array in element units.
    pub strides: *mut i64,
    /// Byte offset from `data`.
    pub byte_offset: u64,
}

/// DLPack managed tensor with version and ownership.
#[repr(C)]
pub struct DLManagedTensorVersioned {
    /// DLPack version.
    pub version: DLPackVersion,
    /// Opaque pointer for the producer's use.
    pub manager_ctx: *mut c_void,
    /// Callback to free the producer's resources.
    pub deleter: Option<unsafe extern "C" fn(*mut DLManagedTensorVersioned)>,
    /// Bitmask flags.
    pub flags: u64,
    /// The tensor descriptor.
    pub dl_tensor: DLTensor,
}

/// CPU device.
pub const KDLCPU: i32 = 1;
/// NVIDIA CUDA GPU device memory.
pub const KDLCUDA: i32 = 2;
/// Pinned CUDA CPU memory.
pub const KDLCUDA_HOST: i32 = 3;
/// AMD ROCm GPU device memory.
pub const KDLROCM: i32 = 10;
/// Pinned ROCm CPU memory.
pub const KDLROCM_HOST: i32 = 11;
/// CUDA managed/unified memory.
pub const KDLCUDA_MANAGED: i32 = 13;

/// Integer type code.
pub const KDLINT: u8 = 0;
/// Floating-point type code.
pub const KDLFLOAT: u8 = 2;
/// Complex type code.
pub const KDLCOMPLEX: u8 = 5;

/// Data is read-only.
pub const DLPACK_FLAG_BITMASK_READ_ONLY: u64 = 1 << 0;
/// Data was copied.
pub const DLPACK_FLAG_BITMASK_IS_COPIED: u64 = 1 << 1;

struct ExportedDLPackTensor {
    tensor: Box<Tensor<f64>>,
    shape: Box<[i64]>,
    strides: Box<[i64]>,
}

struct ImportedDLPackGuard {
    managed: *mut DLManagedTensorVersioned,
    armed: bool,
}

impl ImportedDLPackGuard {
    unsafe fn new(managed: *mut DLManagedTensorVersioned) -> Self {
        Self {
            managed,
            armed: true,
        }
    }

    fn as_ptr(&self) -> *mut DLManagedTensorVersioned {
        self.managed
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for ImportedDLPackGuard {
    fn drop(&mut self) {
        if !self.armed || self.managed.is_null() {
            return;
        }

        unsafe {
            if let Some(deleter) = (*self.managed).deleter {
                deleter(self.managed);
            } else {
                drop(Box::from_raw(self.managed));
            }
        }
    }
}

fn row_major_strides(dims: &[usize]) -> Vec<isize> {
    let ndim = dims.len();
    if ndim == 0 {
        return vec![];
    }
    let mut strides = vec![0isize; ndim];
    strides[ndim - 1] = 1;
    for i in (0..ndim - 1).rev() {
        strides[i] = strides[i + 1] * dims[i + 1] as isize;
    }
    strides
}

fn required_buffer_len(
    dims: &[usize],
    strides: &[isize],
    offset: isize,
) -> tenferro_device::Result<usize> {
    if dims.len() != strides.len() {
        return Err(tenferro_device::Error::InvalidArgument(format!(
            "strides length {} doesn't match dims length {}",
            strides.len(),
            dims.len()
        )));
    }
    if dims.iter().product::<usize>() == 0 {
        return Ok(0);
    }

    let mut min_pos = offset;
    let mut max_pos = offset;
    for (axis, (&dim, &stride)) in dims.iter().zip(strides.iter()).enumerate() {
        if dim == 0 {
            continue;
        }
        let extent = isize::try_from(dim - 1)
            .ok()
            .and_then(|d| d.checked_mul(stride))
            .ok_or_else(|| {
                tenferro_device::Error::StrideError(format!(
                    "extent overflow for dimension {axis} (size={dim}, stride={stride})"
                ))
            })?;
        if extent >= 0 {
            max_pos += extent;
        } else {
            min_pos += extent;
        }
    }

    if min_pos < 0 {
        return Err(tenferro_device::Error::StrideError(format!(
            "layout accesses negative buffer position {min_pos}"
        )));
    }

    Ok(max_pos as usize + 1)
}

fn logical_memory_space_to_dl_device(space: LogicalMemorySpace) -> DLDevice {
    match space {
        LogicalMemorySpace::MainMemory => DLDevice {
            device_type: KDLCPU,
            device_id: 0,
        },
        LogicalMemorySpace::PinnedMemory => DLDevice {
            device_type: KDLCUDA_HOST,
            device_id: 0,
        },
        LogicalMemorySpace::GpuMemory { device_id } => DLDevice {
            device_type: KDLCUDA,
            device_id: device_id as i32,
        },
        LogicalMemorySpace::ManagedMemory => DLDevice {
            device_type: KDLCUDA_MANAGED,
            device_id: 0,
        },
    }
}

unsafe extern "C" fn exported_dlpack_tensor_deleter(managed: *mut DLManagedTensorVersioned) {
    if managed.is_null() {
        return;
    }

    let managed = Box::from_raw(managed);
    if !managed.manager_ctx.is_null() {
        drop(Box::from_raw(
            managed.manager_ctx as *mut ExportedDLPackTensor,
        ));
    }
}

/// Export a tensor as a DLPack managed tensor (zero-copy).
///
/// The tensor handle is **consumed** by this call and must not be
/// used afterwards.
///
/// # Safety
///
/// - `tensor` must be a valid tensor pointer or NULL.
/// - `status` must be a valid, non-null pointer.
///
/// # Examples (C)
///
/// ```c
/// tfe_status_t status;
/// DLManagedTensorVersioned *dl = tfe_tensor_f64_to_dlpack(t, &status);
/// ```
#[no_mangle]
pub unsafe extern "C" fn tfe_tensor_f64_to_dlpack(
    tensor: *mut TfeTensorF64,
    status: *mut tfe_status_t,
) -> *mut DLManagedTensorVersioned {
    let result = catch_unwind(AssertUnwindSafe(
        || -> Result<*mut DLManagedTensorVersioned, tfe_status_t> {
            if tensor.is_null() {
                set_last_error("to_dlpack: tensor pointer is null");
                return Err(TFE_INVALID_ARGUMENT);
            }

            let tensor = handle_take(tensor);
            let shape = tensor
                .dims()
                .iter()
                .map(|&dim| i64::try_from(dim))
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|_| {
                    set_last_error("to_dlpack: shape dimension does not fit into i64");
                    TFE_INVALID_ARGUMENT
                })?
                .into_boxed_slice();
            let strides = tensor
                .strides()
                .iter()
                .map(|&stride| i64::try_from(stride))
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|_| {
                    set_last_error("to_dlpack: stride does not fit into i64");
                    TFE_INVALID_ARGUMENT
                })?
                .into_boxed_slice();

            let data_ptr = if let Some(ptr) = tensor.buffer().as_ptr() {
                ptr as *mut c_void
            } else if let Some(ptr) = tensor.buffer().as_device_ptr() {
                ptr as *mut c_void
            } else {
                set_last_error("to_dlpack: tensor buffer has no exported pointer");
                return Err(TFE_INTERNAL_ERROR);
            };

            let byte_offset =
                u64::try_from((tensor.offset() as i128) * (std::mem::size_of::<f64>() as i128))
                    .map_err(|_| {
                        set_last_error("to_dlpack: byte offset overflow");
                        TFE_INVALID_ARGUMENT
                    })?;

            let mut ctx = Box::new(ExportedDLPackTensor {
                tensor,
                shape,
                strides,
            });

            let shape_ptr = if ctx.shape.is_empty() {
                std::ptr::null_mut()
            } else {
                ctx.shape.as_mut_ptr()
            };
            let strides_ptr = if ctx.strides.is_empty() {
                std::ptr::null_mut()
            } else {
                ctx.strides.as_mut_ptr()
            };
            let ndim = i32::try_from(ctx.shape.len()).map_err(|_| {
                set_last_error("to_dlpack: rank does not fit into i32");
                TFE_INVALID_ARGUMENT
            })?;
            let device = logical_memory_space_to_dl_device(ctx.tensor.logical_memory_space());

            let managed = Box::new(DLManagedTensorVersioned {
                version: DLPackVersion { major: 1, minor: 0 },
                manager_ctx: Box::into_raw(ctx) as *mut c_void,
                deleter: Some(exported_dlpack_tensor_deleter),
                flags: 0,
                dl_tensor: DLTensor {
                    data: data_ptr,
                    device,
                    ndim,
                    dtype: DLDataType {
                        code: KDLFLOAT,
                        bits: 64,
                        lanes: 1,
                    },
                    shape: shape_ptr,
                    strides: strides_ptr,
                    byte_offset,
                },
            });

            Ok(Box::into_raw(managed))
        },
    ));
    finalize_ptr(result, status)
}

/// Import a DLPack managed tensor as a tenferro tensor (zero-copy).
///
/// Currently only `kDLCPU` device and float64 dtype are accepted.
///
/// # Safety
///
/// - `managed` must be a valid pointer to a `DLManagedTensorVersioned`.
/// - `status` must be a valid, non-null pointer.
///
/// # Examples (C)
///
/// ```c
/// tfe_status_t status;
/// tfe_tensor_f64 *t = tfe_tensor_f64_from_dlpack(dl, &status);
/// ```
#[no_mangle]
pub unsafe extern "C" fn tfe_tensor_f64_from_dlpack(
    managed: *mut DLManagedTensorVersioned,
    status: *mut tfe_status_t,
) -> *mut TfeTensorF64 {
    let result = catch_unwind(AssertUnwindSafe(
        || -> Result<*mut TfeTensorF64, tfe_status_t> {
            if managed.is_null() {
                set_last_error("from_dlpack: managed tensor pointer is null");
                return Err(TFE_INVALID_ARGUMENT);
            }

            let mut guard = ImportedDLPackGuard::new(managed);
            let managed_ref = &*guard.as_ptr();
            if managed_ref.version.major != 1 {
                set_last_error("from_dlpack: unsupported DLPack major version");
                return Err(TFE_INVALID_ARGUMENT);
            }

            let dl = &managed_ref.dl_tensor;
            if dl.device.device_type != KDLCPU || dl.device.device_id != 0 {
                set_last_error("from_dlpack: only kDLCPU device_id=0 is supported");
                return Err(TFE_INVALID_ARGUMENT);
            }
            if dl.dtype.code != KDLFLOAT || dl.dtype.bits != 64 || dl.dtype.lanes != 1 {
                set_last_error("from_dlpack: only float64 tensors are supported");
                return Err(TFE_INVALID_ARGUMENT);
            }
            if dl.ndim < 0 {
                set_last_error("from_dlpack: negative ndim is invalid");
                return Err(TFE_INVALID_ARGUMENT);
            }

            let ndim = dl.ndim as usize;
            if ndim > 0 && dl.shape.is_null() {
                set_last_error("from_dlpack: shape pointer is null for non-scalar tensor");
                return Err(TFE_INVALID_ARGUMENT);
            }

            let dims_i64 = if ndim == 0 {
                &[][..]
            } else {
                std::slice::from_raw_parts(dl.shape, ndim)
            };
            let dims = dims_i64
                .iter()
                .map(|&dim| usize::try_from(dim))
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|_| {
                    set_last_error("from_dlpack: shape contains negative or oversized dimension");
                    TFE_INVALID_ARGUMENT
                })?;

            let strides = if dl.strides.is_null() {
                row_major_strides(&dims)
            } else {
                std::slice::from_raw_parts(dl.strides, ndim)
                    .iter()
                    .map(|&stride| isize::try_from(stride))
                    .collect::<std::result::Result<Vec<_>, _>>()
                    .map_err(|_| {
                        set_last_error("from_dlpack: stride does not fit into isize");
                        TFE_INVALID_ARGUMENT
                    })?
            };

            let elem_size = std::mem::size_of::<f64>() as u64;
            if dl.byte_offset % elem_size != 0 {
                set_last_error("from_dlpack: byte offset is not aligned to f64 elements");
                return Err(TFE_INVALID_ARGUMENT);
            }
            let offset = isize::try_from(dl.byte_offset / elem_size).map_err(|_| {
                set_last_error("from_dlpack: element offset does not fit into isize");
                TFE_INVALID_ARGUMENT
            })?;
            let len = required_buffer_len(&dims, &strides, offset)
                .map_err(|err| map_device_error(&err))?;

            let data = dl.data as *const f64;
            if data.is_null() && len > 0 {
                set_last_error("from_dlpack: data pointer is null for non-empty tensor");
                return Err(TFE_INVALID_ARGUMENT);
            }

            let managed_addr = guard.as_ptr() as usize;
            let release = move || unsafe {
                let managed = managed_addr as *mut DLManagedTensorVersioned;
                if let Some(deleter) = (*managed).deleter {
                    deleter(managed);
                } else {
                    drop(Box::from_raw(managed));
                }
            };
            let tensor = Tensor::from_external_parts(data, len, &dims, &strides, offset, release)
                .map_err(|err| map_device_error(&err))?;
            guard.disarm();
            Ok(tensor_to_handle(tensor))
        },
    ));
    finalize_ptr(result, status)
}
