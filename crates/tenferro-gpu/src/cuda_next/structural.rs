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

use t4a_cuda_kernels::{
    plan_broadcast_in_dim, plan_materialize_strided, CAxisValues, CLayout, Entry, Layout,
};
use t4a_cuda_runtime::{Arg, LaunchConfig};

use super::dispatch::{pod_arg, Dispatcher};
use super::engine::CudaEngine;
use super::launch::layout_arg;
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
    /// The element strides of an owned tensor, through its typed payload.
    fn strides_of(&self, tensor: &Tensor, op: &'static str) -> crate::Result<Vec<isize>> {
        with_scalar!(tensor.dtype(), |T| {
            let typed = tensor.as_typed::<T>().ok_or_else(|| {
                crate::Error::runtime_state(op, "the tensor payload is not this scalar type")
            })?;
            Ok(typed.strides().to_vec())
        })
    }

    /// The device tensor behind `input`, or a typed refusal for a view or host tensor.
    fn device_read<'a>(
        &self,
        input: TensorRead<'a>,
        op: &'static str,
    ) -> crate::Result<&'a Tensor> {
        match input {
            TensorRead::Tensor(tensor) if !is_host(tensor) => Ok(tensor),
            TensorRead::Tensor(_) => Err(crate::Error::unsupported(
                op,
                "a host tensor would need an implicit transfer",
            )),
            TensorRead::View(_) => Err(crate::Error::unsupported(
                op,
                "a borrowed view is not served by the structural slice",
            )),
        }
    }

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

    /// A compact device copy of `source` into fresh storage of this runtime, labelled with
    /// `shape` (which must hold the same element count).
    fn copy_to_fresh(
        &mut self,
        source: &Tensor,
        shape: Vec<usize>,
        op: &'static str,
    ) -> crate::Result<Tensor> {
        let dtype = source.dtype();
        let kernel = kernel_dtype(dtype, op)?;
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

impl CudaEngine {
    /// Launch one planned kernel on `out`/`input` through this engine's own module set.
    fn launch_planned(
        &mut self,
        entry: Entry,
        plan: t4a_cuda_kernels::Launch,
        args: &[Arg<'_>],
        op: &'static str,
    ) -> crate::Result<()> {
        let mut modules = super::launch::KernelModules::for_entries(self.runtime(), &[entry])?;
        // SAFETY: the planner proved each operand's span for this dtype and geometry, and
        // both allocations belong to this runtime, so they outlive the scope's completion.
        unsafe { modules.launch(entry, LaunchConfig::new(plan.grid[0], plan.block[0]), args) }
            .map_err(|error| crate::Error::runtime_state_source(op, error))?;
        self.runtime()
            .synchronize()
            .map_err(|error| crate::Error::runtime_state_source(op, error))
    }
}

impl TensorStructural for CudaEngine {
    fn to_contiguous_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        match input {
            TensorRead::Tensor(tensor) => {
                if is_host(tensor) {
                    return tensor.duplicate();
                }
                let shape = tensor.shape().to_vec();
                self.copy_to_fresh(tensor, shape, "to_contiguous_read")
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

    fn transpose_read(&mut self, input: TensorRead<'_>, perm: &[usize]) -> crate::Result<Tensor> {
        let source = self.device_read(input, "transpose_read")?;
        let shape = source.shape().to_vec();
        let strides = self.strides_of(source, "transpose_read")?;
        let rank = shape.len();
        if perm.len() != rank || {
            let mut seen = vec![false; rank];
            perm.iter()
                .any(|&axis| axis >= rank || std::mem::replace(&mut seen[axis], true))
        } {
            return Err(crate::Error::unsupported(
                "transpose_read",
                "the permutation must list every axis exactly once",
            ));
        }
        let dtype = source.dtype();
        let kernel = kernel_dtype(dtype, "transpose_read")?;
        // The output shape is the input's, permuted; the view over the source reads each
        // output axis from the input axis the permutation names.
        let out_shape: Vec<usize> = perm.iter().map(|&axis| shape[axis]).collect();
        let view_extents: Vec<i64> = out_shape.iter().map(|&extent| extent as i64).collect();
        let view_strides: Vec<i64> = perm.iter().map(|&axis| strides[axis] as i64).collect();
        let source_len = i64::try_from(element_count(&shape)?)
            .map_err(|_| crate::Error::unsupported("transpose_read", "count exceeds i64"))?;
        let view = Layout::new(&view_extents, &view_strides, 0)
            .map_err(|error| crate::Error::runtime_state_source("transpose_read", error))?;
        let output = alloc_tensor(self.runtime(), dtype, &out_shape)?;
        let plan = plan_materialize_strided(kernel, &view, source_len)
            .map_err(|error| crate::Error::runtime_state_source("transpose_read", error))?;
        let layout = CLayout::new(&view);
        let (destination, input) = with_scalar!(dtype, |T| Ok((
            self.allocation_of::<T>(&output, "transpose_read")? as *const _,
            self.allocation_of::<T>(source, "transpose_read")? as *const _,
        )))?;
        // SAFETY: both pointers name allocations of this runtime taken above; the planner
        // proved the view's span against the source length and the compact destination.
        let (destination, input) = unsafe { (&*destination, &*input) };
        self.launch_planned(
            Entry::MaterializeStrided,
            plan,
            &[
                Arg::DevicePtr(destination),
                Arg::DevicePtr(input),
                Arg::U32(kernel.tag() as u32),
                layout_arg(&layout),
                Arg::U64(source_len as u64),
            ],
            "transpose_read",
        )?;
        Ok(output)
    }

    fn reshape_read(&mut self, input: TensorRead<'_>, shape: &[usize]) -> crate::Result<Tensor> {
        let source = self.device_read(input, "reshape_read")?;
        let count = element_count(source.shape())?;
        if element_count(shape)? != count {
            return Err(crate::Error::unsupported(
                "reshape_read",
                "the new shape must hold the same element count",
            ));
        }
        // A compact copy carries the same elements in the same order; only the destination
        // allocation's label changes, so the copy is taken at the new shape.
        self.copy_to_fresh(source, shape.to_vec(), "reshape_read")
    }

    fn broadcast_in_dim_read(
        &mut self,
        input: TensorRead<'_>,
        shape: &[usize],
        dims: &[usize],
    ) -> crate::Result<Tensor> {
        let source = self.device_read(input, "broadcast_in_dim_read")?;
        let in_shape = source.shape().to_vec();
        if dims.len() != in_shape.len() {
            return Err(crate::Error::unsupported(
                "broadcast_in_dim_read",
                "one destination axis per input axis is required",
            ));
        }
        let dtype = source.dtype();
        let kernel = kernel_dtype(dtype, "broadcast_in_dim_read")?;
        let out_extents: Vec<i64> = shape.iter().map(|&extent| extent as i64).collect();
        let in_extents: Vec<i64> = in_shape.iter().map(|&extent| extent as i64).collect();
        let in_strides: Vec<i64> = self
            .strides_of(source, "broadcast_in_dim_read")?
            .iter()
            .map(|&stride| i64::try_from(stride).unwrap_or(i64::MAX))
            .collect();
        let out_len = i64::try_from(element_count(shape)?)
            .map_err(|_| crate::Error::unsupported("broadcast_in_dim_read", "count exceeds i64"))?;
        let in_len = i64::try_from(element_count(&in_shape)?)
            .map_err(|_| crate::Error::unsupported("broadcast_in_dim_read", "count exceeds i64"))?;
        let out_layout = Layout::compact(&out_extents)
            .map_err(|error| crate::Error::runtime_state_source("broadcast_in_dim_read", error))?;
        let in_layout = Layout::new(&in_extents, &in_strides, 0)
            .map_err(|error| crate::Error::runtime_state_source("broadcast_in_dim_read", error))?;
        let axis_values: Vec<i64> = dims.iter().map(|&axis| axis as i64).collect();
        let axes = CAxisValues::new(&axis_values)
            .map_err(|error| crate::Error::runtime_state_source("broadcast_in_dim_read", error))?;
        let output = alloc_tensor(self.runtime(), dtype, shape)?;
        let plan = plan_broadcast_in_dim(
            kernel,
            &out_layout,
            out_len,
            &in_layout,
            in_len,
            &axis_values,
        )
        .map_err(|error| crate::Error::runtime_state_source("broadcast_in_dim_read", error))?;
        let (destination, source_ptr) = with_scalar!(dtype, |T| Ok((
            self.allocation_of::<T>(&output, "broadcast_in_dim_read")? as *const _,
            self.allocation_of::<T>(source, "broadcast_in_dim_read")? as *const _,
        )))?;
        // SAFETY: both pointers name allocations of this runtime taken above; the planner
        // proved the input's span and the compact destination.
        let (destination, source_ptr) = unsafe { (&*destination, &*source_ptr) };
        let out_cl = CLayout::new(&out_layout);
        let in_cl = CLayout::new(&in_layout);
        self.launch_planned(
            Entry::BroadcastInDim,
            plan,
            &[
                Arg::DevicePtr(destination),
                Arg::DevicePtr(source_ptr),
                Arg::U32(kernel.tag() as u32),
                layout_arg(&out_cl),
                layout_arg(&in_cl),
                pod_arg(&axes),
                Arg::U64(out_len as u64),
            ],
            "broadcast_in_dim_read",
        )?;
        Ok(output)
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
