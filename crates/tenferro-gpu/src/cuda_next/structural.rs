//! The tensor surface on the new engine: `TensorStructural` for `cuda_next::CudaEngine`.
//!
//! This is the first slice of U5: the materialization boundary plus the two operations the
//! movement and conversion kernels already serve — a same-placement canonical copy and a
//! dtype cast. Every other structural operation answers a typed
//! [`Unsupported`](crate::Error) until its kernel slice lands, so nothing silently falls
//! back to a host round trip.

use tenferro_tensor::{
    DType, MemoryKind, StorageBuffer, Tensor, TensorRead, TensorScalar, TensorStructural,
    TensorWrite, TypedTensor,
};

use t4a_cuda_kernels::Entry;

use super::dispatch::Dispatcher;
use super::engine::CudaEngine;
use super::runtime::CudaRuntime;
use super::storage::{fresh_identity, CudaStorage};

/// Scalars the structural slice serves today.
macro_rules! with_scalar {
    ($dtype:expr, |$t:ident| $body:expr) => {
        match $dtype {
            DType::F32 => {
                #[allow(dead_code)]
                type $t = f32;
                $body
            }
            DType::F64 => {
                #[allow(dead_code)]
                type $t = f64;
                $body
            }
            other => Err(crate::Error::unsupported(
                "cuda structural",
                format!("the structural slice serves F32 and F64, not {other:?}"),
            )),
        }
    };
}

/// Allocate a zeroed device tensor of `shape` in the runtime's placement.
pub(crate) fn alloc_tensor(
    runtime: &CudaRuntime,
    dtype: DType,
    shape: &[usize],
) -> crate::Result<Tensor> {
    let count = element_count(shape)?;
    let bytes = with_scalar!(
        dtype,
        |T| Ok(count.saturating_mul(std::mem::size_of::<T>()))
    )?;
    let buffer = {
        let mut guard = runtime.submit();
        guard
            .alloc_zeroed(bytes, 256)
            .map_err(|error| crate::Error::runtime_state_source("cuda allocate", error))?
    };
    let (domain, id) = fresh_identity(runtime);
    let storage = CudaStorage {
        buffer,
        bytes,
        device: runtime.device_ordinal(),
        domain,
        id,
    };
    with_scalar!(dtype, |T| Ok(Tensor::from_typed(
        TypedTensor::<T>::from_buffer_col_major(
            shape.to_vec(),
            StorageBuffer::Backend(Box::new(storage)),
            tenferro_tensor::Placement::default(),
        )?
    )))
}

pub(crate) fn element_count(shape: &[usize]) -> crate::Result<usize> {
    let mut count = 1usize;
    for &extent in shape {
        count = count.checked_mul(extent).ok_or_else(|| {
            crate::Error::unsupported("cuda tensor", "the shape's element count overflows")
        })?;
    }
    Ok(count)
}

/// Whether `tensor` is host-resident rather than a backend allocation.
fn is_host(tensor: &Tensor) -> bool {
    !tensor.is_backend_buffer()
        && matches!(
            tensor.placement().memory_kind,
            MemoryKind::PinnedHost | MemoryKind::UnpinnedHost
        )
}

/// The CUDA allocation behind `tensor`, or a typed refusal.
fn storage_of<'a, T: TensorScalar>(
    tensor: &'a Tensor,
    op: &'static str,
) -> crate::Result<&'a CudaStorage> {
    let typed = tensor.as_typed::<T>().ok_or_else(|| {
        crate::Error::runtime_state(op, "the tensor payload is not this scalar type")
    })?;
    match typed.buffer() {
        StorageBuffer::Host(_) => Err(crate::Error::runtime_state(
            op,
            "expected a CUDA allocation, not host storage",
        )),
        StorageBuffer::Backend(buffer) => {
            buffer.as_ref().as_any().downcast_ref().ok_or_else(|| {
                crate::Error::runtime_state(op, "expected a CUDA allocation of this backend")
            })
        }
    }
}

/// The kernel crate's dtype tag for a tensor dtype.
fn kernel_dtype(dtype: DType, op: &'static str) -> crate::Result<t4a_cuda_kernels::DType> {
    Ok(match dtype {
        DType::F32 => t4a_cuda_kernels::DType::F32,
        DType::F64 => t4a_cuda_kernels::DType::F64,
        other => {
            return Err(crate::Error::unsupported(
                op,
                format!("the structural slice serves F32 and F64, not {other:?}"),
            ))
        }
    })
}

impl CudaEngine {
    fn allocation_of<'a, T: TensorScalar>(
        &self,
        tensor: &'a Tensor,
        op: &'static str,
    ) -> crate::Result<&'a t4a_cuda_runtime::Buffer> {
        let storage = storage_of::<T>(tensor, op)?;
        if storage.domain() != self.runtime().allocation_domain_id() {
            return Err(crate::Error::runtime_state(
                op,
                "expected a CUDA allocation of this runtime",
            ));
        }
        Ok(&storage.buffer)
    }

    /// A compact device copy of `source` into fresh storage of this runtime.
    fn copy_to_fresh(&mut self, source: &Tensor, op: &'static str) -> crate::Result<Tensor> {
        let dtype = source.dtype();
        let kernel = kernel_dtype(dtype, op)?;
        let shape = source.shape().to_vec();
        let len = i64::try_from(element_count(&shape)?)
            .map_err(|_| crate::Error::unsupported(op, "the element count exceeds i64"))?;
        let output = alloc_tensor(self.runtime(), dtype, &shape)?;
        let runtime = self.runtime().clone();
        let mut dispatcher = Dispatcher::for_entries(&runtime, &[Entry::CopyExact])?;
        with_scalar!(dtype, |T| {
            dispatcher.copy_exact(
                kernel,
                len,
                self.allocation_of::<T>(&output, op)?,
                self.allocation_of::<T>(source, op)?,
            )
        })?;
        // The returned tensor must be complete when the caller sees it.
        self.runtime()
            .synchronize()
            .map_err(|error| crate::Error::runtime_state_source(op, error))?;
        Ok(output)
    }
}

impl TensorStructural for CudaEngine {
    fn to_contiguous_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        match input {
            TensorRead::Tensor(tensor) => {
                if is_host(tensor) {
                    return tensor.duplicate();
                }
                self.copy_to_fresh(tensor, "to_contiguous_read")
            }
            TensorRead::View(view) => Err(crate::Error::unsupported(
                "to_contiguous_read",
                format!(
                    "a borrowed {} view is not served by the structural slice",
                    view.backend_family().map_or("host", |_| "backend")
                ),
            )),
        }
    }

    fn copy_read_into(&mut self, src: TensorRead<'_>, dst: TensorWrite<'_>) -> crate::Result<()> {
        let (source, destination) = match (src, dst) {
            (TensorRead::Tensor(source), TensorWrite::Tensor(destination)) => {
                (source, &*destination)
            }
            _ => {
                return Err(crate::Error::unsupported(
                    "copy_read_into",
                    "a borrowed view is not served by the structural slice",
                ))
            }
        };
        if is_host(source) || is_host(destination) {
            return Err(crate::Error::unsupported(
                "copy_read_into",
                "a host operand would need an implicit transfer",
            ));
        }
        if source.dtype() != destination.dtype() || source.shape() != destination.shape() {
            return Err(crate::Error::unsupported(
                "copy_read_into",
                "source and destination must share dtype and shape",
            ));
        }
        let dtype = source.dtype();
        let kernel = kernel_dtype(dtype, "copy_read_into")?;
        let len = i64::try_from(element_count(source.shape())?)
            .map_err(|_| crate::Error::unsupported("copy_read_into", "count exceeds i64"))?;
        let runtime = self.runtime().clone();
        let mut dispatcher = Dispatcher::for_entries(&runtime, &[Entry::CopyExact])?;
        with_scalar!(dtype, |T| {
            dispatcher.copy_exact(
                kernel,
                len,
                self.allocation_of::<T>(destination, "copy_read_into")?,
                self.allocation_of::<T>(source, "copy_read_into")?,
            )
        })?;
        self.runtime()
            .synchronize()
            .map_err(|error| crate::Error::runtime_state_source("copy_read_into", error))
    }

    fn cast(&mut self, input: &Tensor, to: DType) -> crate::Result<Tensor> {
        if is_host(input) {
            return Err(crate::Error::unsupported(
                "cast",
                "a host tensor would need an implicit transfer",
            ));
        }
        let from = input.dtype();
        let (kernel_from, kernel_to) = (kernel_dtype(from, "cast")?, kernel_dtype(to, "cast")?);
        let shape = input.shape().to_vec();
        let len = i64::try_from(element_count(&shape)?)
            .map_err(|_| crate::Error::unsupported("cast", "the element count exceeds i64"))?;
        let output = alloc_tensor(self.runtime(), to, &shape)?;
        let runtime = self.runtime().clone();
        let mut dispatcher = Dispatcher::for_entries(&runtime, &[Entry::Convert])?;
        with_scalar!(from, |T| {
            let source = self.allocation_of::<T>(input, "cast")?;
            with_scalar!(to, |U| {
                dispatcher.convert(
                    kernel_to,
                    kernel_from,
                    len,
                    self.allocation_of::<U>(&output, "cast")?,
                    source,
                )
            })
        })?;
        self.runtime()
            .synchronize()
            .map_err(|error| crate::Error::runtime_state_source("cast", error))?;
        Ok(output)
    }

    fn transpose_read(&mut self, _input: TensorRead<'_>, _perm: &[usize]) -> crate::Result<Tensor> {
        Err(crate::Error::unsupported(
            "transpose_read",
            "the permutation slice is not on the new path yet",
        ))
    }

    fn reshape_read(&mut self, _input: TensorRead<'_>, _shape: &[usize]) -> crate::Result<Tensor> {
        Err(crate::Error::unsupported(
            "reshape_read",
            "the reshape slice is not on the new path yet",
        ))
    }

    fn broadcast_in_dim_read(
        &mut self,
        _input: TensorRead<'_>,
        _shape: &[usize],
        _dims: &[usize],
    ) -> crate::Result<Tensor> {
        Err(crate::Error::unsupported(
            "broadcast_in_dim_read",
            "the broadcast slice is not on the new path yet",
        ))
    }

    fn extract_diagonal(
        &mut self,
        _input: &Tensor,
        _axis_a: usize,
        _axis_b: usize,
    ) -> crate::Result<Tensor> {
        Err(crate::Error::unsupported(
            "extract_diagonal",
            "the diagonal slice is not on the new path yet",
        ))
    }

    fn embed_diagonal(
        &mut self,
        _input: &Tensor,
        _axis_a: usize,
        _axis_b: usize,
    ) -> crate::Result<Tensor> {
        Err(crate::Error::unsupported(
            "embed_diagonal",
            "the diagonal slice is not on the new path yet",
        ))
    }

    fn tril(&mut self, _input: &Tensor, _k: i64) -> crate::Result<Tensor> {
        Err(crate::Error::unsupported(
            "tril",
            "the triangular slice is not on the new path yet",
        ))
    }

    fn triu(&mut self, _input: &Tensor, _k: i64) -> crate::Result<Tensor> {
        Err(crate::Error::unsupported(
            "triu",
            "the triangular slice is not on the new path yet",
        ))
    }
}

#[cfg(test)]
#[path = "structural/tests.rs"]
mod tests;
