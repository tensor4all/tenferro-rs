//! Host-to-device and device-to-host transfers via CubeCL-managed allocations.

use cubecl::client::ComputeClient;
use cubecl::prelude::CubeElement;
use cubecl_cuda::CudaRuntime;
use num_complex::{Complex32, Complex64};

use crate::cubecl::runtime::CubeclRuntime;
use crate::types::{
    Buffer, ComputeDevice, CubeclBuffer, MemoryKind, Placement, Tensor, TypedTensor,
};

/// Upload a host tensor into a CubeCL-managed GPU allocation.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::cubecl::{upload_tensor, CubeclRuntime};
/// use tenferro_tensor::{Result, Tensor};
///
/// let _upload: fn(&CubeclRuntime, &Tensor) -> Result<Tensor> = upload_tensor;
/// ```
pub fn upload_tensor(rt: &CubeclRuntime, tensor: &Tensor) -> crate::Result<Tensor> {
    let client = rt.client();
    match tensor {
        Tensor::F64(t) => upload_typed::<f64>(client, rt.device_ordinal(), t).map(Tensor::F64),
        Tensor::F32(t) => upload_typed::<f32>(client, rt.device_ordinal(), t).map(Tensor::F32),
        Tensor::I64(t) => upload_typed::<i64>(client, rt.device_ordinal(), t).map(Tensor::I64),
        Tensor::C64(t) => {
            upload_typed::<Complex64>(client, rt.device_ordinal(), t).map(Tensor::C64)
        }
        Tensor::C32(t) => {
            upload_typed::<Complex32>(client, rt.device_ordinal(), t).map(Tensor::C32)
        }
    }
}

/// Download a CubeCL-managed GPU tensor back to host memory.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::cubecl::{download_tensor, CubeclRuntime};
/// use tenferro_tensor::{Result, Tensor};
///
/// let _download: fn(&CubeclRuntime, &Tensor) -> Result<Tensor> = download_tensor;
/// ```
pub fn download_tensor(rt: &CubeclRuntime, tensor: &Tensor) -> crate::Result<Tensor> {
    let client = rt.client();
    match tensor {
        Tensor::F64(t) => download_typed::<f64>(client, t).map(Tensor::F64),
        Tensor::F32(t) => download_typed::<f32>(client, t).map(Tensor::F32),
        Tensor::I64(t) => download_typed::<i64>(client, t).map(Tensor::I64),
        Tensor::C64(t) => download_typed::<Complex64>(client, t).map(Tensor::C64),
        Tensor::C32(t) => download_typed::<Complex32>(client, t).map(Tensor::C32),
    }
}

/// Extract the raw CUDA device pointer from a CubeCL-managed tensor allocation.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::cubecl::{device_ptr, CubeclRuntime};
/// use tenferro_tensor::{Result, Tensor};
///
/// let _device_ptr: fn(&CubeclRuntime, &Tensor) -> Result<u64> = device_ptr;
/// ```
pub fn device_ptr(rt: &CubeclRuntime, tensor: &Tensor) -> crate::Result<u64> {
    let handle = cubecl_handle(tensor)?;
    let resource =
        rt.client()
            .get_resource(handle)
            .map_err(|err| crate::Error::BackendFailure {
                op: "device_ptr",
                message: format!("{err:?}"),
            })?;
    Ok(resource.resource().ptr)
}

fn upload_typed<T: CubeElement + Clone>(
    client: &ComputeClient<CudaRuntime>,
    device_ordinal: usize,
    typed: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>> {
    let host_data = match &typed.buffer {
        Buffer::Host(data) => data,
        Buffer::Backend(_) => {
            return Err(crate::Error::BackendFailure {
                op: "upload",
                message: "expected host buffer".into(),
            });
        }
        Buffer::Cubecl(_) => {
            return Err(crate::Error::BackendFailure {
                op: "upload",
                message: "tensor is already backed by CubeCL storage".into(),
            });
        }
    };

    let handle = client.create_from_slice(T::as_bytes(host_data));
    Ok(TypedTensor {
        buffer: Buffer::Cubecl(CubeclBuffer::new(handle, host_data.len())),
        shape: typed.shape.clone(),
        placement: Placement {
            memory_kind: MemoryKind::Device,
            resident_device: Some(ComputeDevice {
                kind: "cuda".into(),
                ordinal: device_ordinal,
            }),
        },
    })
}

fn download_typed<T: CubeElement + Clone>(
    client: &ComputeClient<CudaRuntime>,
    typed: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>> {
    let handle = match &typed.buffer {
        Buffer::Host(_) => {
            return Err(crate::Error::BackendFailure {
                op: "download",
                message: "expected CubeCL buffer".into(),
            });
        }
        Buffer::Backend(_) => {
            return Err(crate::Error::BackendFailure {
                op: "download",
                message: "expected CubeCL buffer".into(),
            });
        }
        Buffer::Cubecl(buffer) => buffer.handle.clone(),
    };

    let bytes = client
        .read_one(handle)
        .map_err(|err| crate::Error::BackendFailure {
            op: "download",
            message: format!("{err:?}"),
        })?;
    let data = T::from_bytes(&bytes).to_vec();
    Ok(TypedTensor::from_vec(typed.shape.clone(), data))
}

fn cubecl_handle(tensor: &Tensor) -> crate::Result<cubecl::server::Handle> {
    match tensor {
        Tensor::F64(t) => cubecl_handle_from_buffer(&t.buffer),
        Tensor::F32(t) => cubecl_handle_from_buffer(&t.buffer),
        Tensor::I64(t) => cubecl_handle_from_buffer(&t.buffer),
        Tensor::C64(t) => cubecl_handle_from_buffer(&t.buffer),
        Tensor::C32(t) => cubecl_handle_from_buffer(&t.buffer),
    }
}

fn cubecl_handle_from_buffer<T>(buffer: &Buffer<T>) -> crate::Result<cubecl::server::Handle> {
    match buffer {
        Buffer::Host(_) | Buffer::Backend(_) => Err(crate::Error::BackendFailure {
            op: "cubecl_handle",
            message: "expected CubeCL buffer".into(),
        }),
        Buffer::Cubecl(buffer) => Ok(buffer.handle.clone()),
    }
}
