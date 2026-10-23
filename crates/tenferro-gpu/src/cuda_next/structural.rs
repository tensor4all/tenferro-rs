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
    plan_broadcast_in_dim, plan_embed_diagonal_copy, plan_extract_diagonal,
    plan_materialize_strided, plan_tril, plan_triu, CAxisValues, CLayout, Entry, Layout,
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
            DType::Bool => {
                #[allow(dead_code)]
                type $t = bool;
                $body
            }
            DType::I32 => {
                #[allow(dead_code)]
                type $t = i32;
                $body
            }
            DType::I64 => {
                #[allow(dead_code)]
                type $t = i64;
                $body
            }
            other => Err(crate::Error::unsupported(
                "cuda structural",
                format!("the structural slice does not serve {other:?}"),
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
        DType::Bool => t4a_cuda_kernels::DType::Bool,
        DType::I32 => t4a_cuda_kernels::DType::I32,
        DType::I64 => t4a_cuda_kernels::DType::I64,
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
        &self,
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
        input: &Tensor,
        axis_a: usize,
        axis_b: usize,
    ) -> crate::Result<Tensor> {
        let shape = input.shape().to_vec();
        let kernel = kernel_dtype(input.dtype(), "extract_diagonal")?;
        if axis_a >= shape.len() || axis_b >= shape.len() || axis_a == axis_b {
            return Err(crate::Error::unsupported(
                "extract_diagonal",
                "two distinct in-range axes are required",
            ));
        }
        // The output is the input's shape without the second axis; its length is the
        // smaller of the two axes.
        let mut out_shape: Vec<usize> = shape.clone();
        let removed = out_shape.remove(axis_b);
        out_shape[axis_a] = out_shape[axis_a].min(removed);
        let strides = self.strides_of(input, "extract_diagonal")?;
        let in_layout = compact_layout(&shape, &strides, "extract_diagonal")?;
        let out_layout =
            compact_layout(&out_shape, &compact_strides(&out_shape), "extract_diagonal")?;
        let in_len = count_i64(&shape, "extract_diagonal")?;
        let out_len = count_i64(&out_shape, "extract_diagonal")?;
        let output = alloc_tensor(self.runtime(), input.dtype(), &out_shape)?;
        let plan = plan_extract_diagonal(
            kernel,
            &in_layout,
            in_len,
            &out_layout,
            out_len,
            axis_a as u32,
            axis_b as u32,
        )
        .map_err(|error| crate::Error::runtime_state_source("extract_diagonal", error))?;
        let (destination, source) = self.two_allocations(&output, input, "extract_diagonal")?;
        let (in_cl, out_cl) = (CLayout::new(&in_layout), CLayout::new(&out_layout));
        self.launch_planned(
            Entry::ExtractDiagonal,
            plan,
            &[
                Arg::DevicePtr(destination),
                Arg::DevicePtr(source),
                Arg::U32(kernel.tag() as u32),
                layout_arg(&out_cl),
                layout_arg(&in_cl),
                Arg::I32(axis_a as i32),
                Arg::I32(axis_b as i32),
                Arg::U64(out_len as u64),
            ],
            "extract_diagonal",
        )?;
        Ok(output)
    }

    fn embed_diagonal(
        &mut self,
        input: &Tensor,
        axis_a: usize,
        axis_b: usize,
    ) -> crate::Result<Tensor> {
        let shape = input.shape().to_vec();
        let kernel = kernel_dtype(input.dtype(), "embed_diagonal")?;
        if axis_a >= shape.len() || axis_b > shape.len() {
            return Err(crate::Error::unsupported(
                "embed_diagonal",
                "the insertion position must be in range",
            ));
        }
        // The output re-inserts an axis at `axis_b` whose extent is the input's `axis_a`
        // extent; the diagonal positions are the only ones written, so the destination is
        // zero-filled by the allocation.
        let mut out_shape = shape.clone();
        out_shape.insert(axis_b, shape[axis_a]);
        let strides = self.strides_of(input, "embed_diagonal")?;
        let in_layout = compact_layout(&shape, &strides, "embed_diagonal")?;
        let out_layout =
            compact_layout(&out_shape, &compact_strides(&out_shape), "embed_diagonal")?;
        let in_len = count_i64(&shape, "embed_diagonal")?;
        let out_len = count_i64(&out_shape, "embed_diagonal")?;
        let output = alloc_tensor(self.runtime(), input.dtype(), &out_shape)?;
        let plan = plan_embed_diagonal_copy(
            kernel,
            &in_layout,
            in_len,
            &out_layout,
            out_len,
            axis_a as u32,
            axis_b as u32,
        )
        .map_err(|error| crate::Error::runtime_state_source("embed_diagonal", error))?;
        let (destination, source) = self.two_allocations(&output, input, "embed_diagonal")?;
        let (in_cl, out_cl) = (CLayout::new(&in_layout), CLayout::new(&out_layout));
        self.launch_planned(
            Entry::EmbedDiagonalCopy,
            plan,
            &[
                Arg::DevicePtr(destination),
                Arg::DevicePtr(source),
                Arg::U32(kernel.tag() as u32),
                layout_arg(&out_cl),
                layout_arg(&in_cl),
                Arg::I32(axis_a as i32),
                Arg::I32(axis_b as i32),
                Arg::U64(in_len as u64),
            ],
            "embed_diagonal",
        )?;
        Ok(output)
    }

    fn tril(&mut self, input: &Tensor, k: i64) -> crate::Result<Tensor> {
        self.triangular(input, k, true)
    }

    fn triu(&mut self, input: &Tensor, k: i64) -> crate::Result<Tensor> {
        self.triangular(input, k, false)
    }
}

impl CudaEngine {
    /// `tril`/`triu` over a rank-2-or-higher compact tensor, batching on higher axes.
    fn triangular(&mut self, input: &Tensor, k: i64, lower: bool) -> crate::Result<Tensor> {
        let op = if lower { "tril" } else { "triu" };
        let shape = input.shape().to_vec();
        if shape.len() < 2 {
            return Err(crate::Error::unsupported(
                op,
                "a triangular mask needs at least two axes",
            ));
        }
        let kernel = kernel_dtype(input.dtype(), op)?;
        let len = count_i64(&shape, op)?;
        let (rows, cols) = (shape[0] as i64, shape[1] as i64);
        let output = alloc_tensor(self.runtime(), input.dtype(), &shape)?;
        let plan = if lower {
            plan_tril(kernel, rows, cols, len)
        } else {
            plan_triu(kernel, rows, cols, len)
        }
        .map_err(|error| crate::Error::runtime_state_source(op, error))?;
        let (destination, source) = self.two_allocations(&output, input, op)?;
        self.launch_planned(
            if lower { Entry::Tril } else { Entry::Triu },
            plan,
            &[
                Arg::DevicePtr(destination),
                Arg::DevicePtr(source),
                Arg::U32(kernel.tag() as u32),
                pod_arg(&k),
                Arg::U64(rows as u64),
                Arg::U64(cols as u64),
                Arg::U64(len as u64),
            ],
            op,
        )?;
        Ok(output)
    }

    /// The allocations of a fresh `output` and an existing `source`, both of this runtime.
    fn two_allocations<'a>(
        &'a self,
        output: &'a Tensor,
        source: &'a Tensor,
        op: &'static str,
    ) -> crate::Result<(&'a t4a_cuda_runtime::Buffer, &'a t4a_cuda_runtime::Buffer)> {
        with_scalar!(output.dtype(), |T| {
            let destination = self.allocation_of::<T>(output, op)?;
            let input = self.allocation_of::<T>(source, op)?;
            Ok((destination, input))
        })
    }
}

/// A column-major compact stride list for `shape`.
fn compact_strides(shape: &[usize]) -> Vec<isize> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut stride = 1isize;
    for &extent in shape {
        strides.push(stride);
        stride *= extent as isize;
    }
    strides
}

fn compact_layout(shape: &[usize], strides: &[isize], op: &'static str) -> crate::Result<Layout> {
    let extents: Vec<i64> = shape.iter().map(|&extent| extent as i64).collect();
    let strides: Vec<i64> = strides.iter().map(|&stride| stride as i64).collect();
    Layout::new(&extents, &strides, 0)
        .map_err(|error| crate::Error::runtime_state_source(op, error))
}

fn count_i64(shape: &[usize], op: &'static str) -> crate::Result<i64> {
    i64::try_from(element_count(shape)?)
        .map_err(|_| crate::Error::unsupported(op, "the element count exceeds i64"))
}

#[cfg(test)]
#[path = "structural/tests.rs"]
mod tests;

/// One `TensorReduction` request: the axes and the kernel operation.
fn reduction_op(op: ReduceOp) -> t4a_cuda_kernels::ReduceOp {
    match op {
        ReduceOp::Sum => t4a_cuda_kernels::ReduceOp::Sum,
        ReduceOp::Prod => t4a_cuda_kernels::ReduceOp::Prod,
        ReduceOp::Max => t4a_cuda_kernels::ReduceOp::Max,
        ReduceOp::Min => t4a_cuda_kernels::ReduceOp::Min,
        ReduceOp::SumSquares => t4a_cuda_kernels::ReduceOp::SumSquares,
    }
}

/// The reduction family the tensor layer asks for.
#[derive(Clone, Copy)]
enum ReduceOp {
    Sum,
    Prod,
    Max,
    Min,
    SumSquares,
}

impl CudaEngine {
    /// Reduce `axes` one axis at a time, largest axis first so the indices stay valid.
    fn reduce_axes(
        &mut self,
        input: TensorRead<'_>,
        axes: &[usize],
        op: ReduceOp,
        name: &'static str,
    ) -> crate::Result<Tensor> {
        let source = self.device_read(input, name)?;
        let dtype = source.dtype();
        let kernel = kernel_dtype(dtype, name)?;
        let kernel_op = reduction_op(op);
        let mut shape = source.shape().to_vec();
        let mut ordered: Vec<usize> = axes.to_vec();
        ordered.sort_unstable();
        ordered.dedup();
        if let Some(&axis) = ordered.iter().find(|&&axis| axis >= shape.len()) {
            return Err(crate::Error::unsupported(
                name,
                format!("axis {axis} is outside the tensor's rank {}", shape.len()),
            ));
        }
        if ordered.is_empty() {
            let shape = source.shape().to_vec();
            return self.copy_to_fresh(source, shape, name);
        }
        // The first pass reads the caller's tensor; later passes read the previous result.
        let mut owned: Option<Tensor> = None;
        let width = 32u32;
        // The reduction kernel is keepdims: the destination has the input's rank with an
        // extent of one on the reduced axis, so the unit axes are squeezed off at the end.
        for &axis in ordered.iter().rev() {
            let out_shape: Vec<usize> = {
                let mut next = shape.clone();
                next[axis] = 1;
                next
            };
            let out_layout = compact_layout(&out_shape, &compact_strides(&out_shape), name)?;
            let in_layout = compact_layout(&shape, &compact_strides(&shape), name)?;
            let out_len = count_i64(&out_shape, name)?;
            let in_len = count_i64(&shape, name)?;
            let output = alloc_tensor(self.runtime(), dtype, &out_shape)?;
            // Partials hold `splits * out_len` at most; the crate's cap is the input length.
            let partials = alloc_tensor(self.runtime(), dtype, &[in_len as usize])?;
            let source_tensor = owned.as_ref().unwrap_or(source);
            let runtime = self.runtime().clone();
            let mut dispatcher =
                Dispatcher::for_entries(&runtime, &[Entry::Reduce, Entry::ReducePartials])?;
            with_scalar!(dtype, |T| {
                dispatcher.reduce_auto(
                    kernel,
                    kernel_op,
                    &out_layout,
                    out_len,
                    &in_layout,
                    in_len,
                    axis as u32,
                    width,
                    self.allocation_of::<T>(&output, name)?,
                    self.allocation_of::<T>(source_tensor, name)?,
                    self.allocation_of::<T>(&partials, name)?,
                )
            })?;
            self.runtime()
                .synchronize()
                .map_err(|error| crate::Error::runtime_state_source(name, error))?;
            drop(partials);
            shape = out_shape;
            owned = Some(output);
        }
        let intermediate = owned.expect("at least one axis was reduced");
        let squeezed: Vec<usize> = (0..intermediate.shape().len())
            .filter(|axis| !ordered.contains(axis))
            .map(|axis| intermediate.shape()[axis])
            .collect();
        if squeezed == intermediate.shape() {
            return Ok(intermediate);
        }
        self.copy_to_fresh(&intermediate, squeezed, name)
    }
}

impl tenferro_tensor::TensorReduction for CudaEngine {
    fn reduce_sum_read(&mut self, input: TensorRead<'_>, axes: &[usize]) -> crate::Result<Tensor> {
        self.reduce_axes(input, axes, ReduceOp::Sum, "reduce_sum_read")
    }

    fn reduce_prod_read(&mut self, input: TensorRead<'_>, axes: &[usize]) -> crate::Result<Tensor> {
        self.reduce_axes(input, axes, ReduceOp::Prod, "reduce_prod_read")
    }

    fn reduce_max_read(&mut self, input: TensorRead<'_>, axes: &[usize]) -> crate::Result<Tensor> {
        self.reduce_axes(input, axes, ReduceOp::Max, "reduce_max_read")
    }

    fn reduce_min_read(&mut self, input: TensorRead<'_>, axes: &[usize]) -> crate::Result<Tensor> {
        self.reduce_axes(input, axes, ReduceOp::Min, "reduce_min_read")
    }

    fn reduce_sum_squares_read(
        &mut self,
        input: TensorRead<'_>,
        axes: &[usize],
    ) -> crate::Result<Tensor> {
        self.reduce_axes(input, axes, ReduceOp::SumSquares, "reduce_sum_squares_read")
    }
}

/// The elementwise family the tensor layer asks for.
#[derive(Clone, Copy)]
enum Elementwise {
    Add,
    Sub,
    Mul,
    Div,
}

impl CudaEngine {
    /// One same-shape binary elementwise operation over two device tensors.
    fn elementwise_binary(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        op: Elementwise,
        name: &'static str,
    ) -> crate::Result<Tensor> {
        let left = self.device_read(lhs, name)?;
        let right = self.device_read(rhs, name)?;
        if left.shape() != right.shape() {
            return Err(crate::Error::unsupported(
                name,
                "both operands must share a shape on this path",
            ));
        }
        let dtype = left.dtype();
        if right.dtype() != dtype {
            return Err(crate::Error::unsupported(
                name,
                "both operands must share a dtype on this path",
            ));
        }
        let kernel = kernel_dtype(dtype, name)?;
        let shape = left.shape().to_vec();
        let len = count_i64(&shape, name)?;
        let output = alloc_tensor(self.runtime(), dtype, &shape)?;
        let runtime = self.runtime().clone();
        let entry = match op {
            Elementwise::Add => Entry::Add,
            Elementwise::Sub => Entry::Sub,
            Elementwise::Mul => Entry::Mul,
            Elementwise::Div => Entry::Div,
        };
        let mut dispatcher = Dispatcher::for_entries(&runtime, &[entry])?;
        with_scalar!(dtype, |T| {
            dispatcher.arithmetic(
                entry,
                kernel,
                len,
                self.allocation_of::<T>(&output, name)?,
                self.allocation_of::<T>(left, name)?,
                self.allocation_of::<T>(right, name)?,
            )
        })?;
        self.runtime()
            .synchronize()
            .map_err(|error| crate::Error::runtime_state_source(name, error))?;
        Ok(output)
    }

    /// One unary elementwise operation over a device tensor.
    fn elementwise_unary(
        &mut self,
        input: TensorRead<'_>,
        op: t4a_cuda_kernels::UnaryOp,
        name: &'static str,
    ) -> crate::Result<Tensor> {
        let source = self.device_read(input, name)?;
        let dtype = source.dtype();
        let kernel = kernel_dtype(dtype, name)?;
        let shape = source.shape().to_vec();
        let len = count_i64(&shape, name)?;
        let output = alloc_tensor(self.runtime(), dtype, &shape)?;
        let runtime = self.runtime().clone();
        let mut dispatcher = Dispatcher::for_entries(&runtime, &[Entry::Unary])?;
        with_scalar!(dtype, |T| {
            dispatcher.unary(
                op,
                kernel,
                kernel,
                len,
                self.allocation_of::<T>(&output, name)?,
                self.allocation_of::<T>(source, name)?,
            )
        })?;
        self.runtime()
            .synchronize()
            .map_err(|error| crate::Error::runtime_state_source(name, error))?;
        Ok(output)
    }

    /// One binary selection (`maximum`/`minimum`) over two device tensors.
    fn elementwise_extreme(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        op: t4a_cuda_kernels::BinaryOp,
        name: &'static str,
    ) -> crate::Result<Tensor> {
        let left = self.device_read(lhs, name)?;
        let right = self.device_read(rhs, name)?;
        if left.shape() != right.shape() || left.dtype() != right.dtype() {
            return Err(crate::Error::unsupported(
                name,
                "both operands must share a shape and a dtype on this path",
            ));
        }
        let dtype = left.dtype();
        let kernel = kernel_dtype(dtype, name)?;
        let shape = left.shape().to_vec();
        let len = count_i64(&shape, name)?;
        let output = alloc_tensor(self.runtime(), dtype, &shape)?;
        let runtime = self.runtime().clone();
        let mut dispatcher = Dispatcher::for_entries(&runtime, &[Entry::Binary])?;
        // The entry always takes a status word for its checked-integer route; a
        // comparison of two full buffers never sets it.
        let status = runtime
            .upload_bytes(&[0u8; 4])
            .map_err(|error| crate::Error::runtime_state_source(name, error))?;
        with_scalar!(dtype, |T| {
            dispatcher.binary(
                op,
                kernel,
                t4a_cuda_kernels::ScalarSide::Both,
                len,
                self.allocation_of::<T>(&output, name)?,
                self.allocation_of::<T>(left, name)?,
                self.allocation_of::<T>(right, name)?,
                &status,
            )
        })?;
        self.runtime()
            .synchronize()
            .map_err(|error| crate::Error::runtime_state_source(name, error))?;
        Ok(output)
    }
}

impl CudaEngine {
    /// Remainder of two same-shaped device tensors.
    ///
    /// Floats use the plain remainder entry; integers use the checked one, whose
    /// divide-by-zero flag is read back and reported rather than silently producing a zero.
    fn remainder(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        name: &'static str,
    ) -> crate::Result<Tensor> {
        let left = self.device_read(lhs, name)?;
        let right = self.device_read(rhs, name)?;
        if left.shape() != right.shape() || left.dtype() != right.dtype() {
            return Err(crate::Error::unsupported(
                name,
                "both operands must share a shape and a dtype on this path",
            ));
        }
        let dtype = left.dtype();
        let kernel = kernel_dtype(dtype, name)?;
        let op = match dtype {
            DType::F32 | DType::F64 => t4a_cuda_kernels::BinaryOp::Rem,
            DType::I32 | DType::I64 => t4a_cuda_kernels::BinaryOp::RemChecked,
            other => {
                return Err(crate::Error::unsupported(
                    name,
                    format!("the remainder is not defined for {other:?}"),
                ))
            }
        };
        let shape = left.shape().to_vec();
        let len = count_i64(&shape, name)?;
        let output = alloc_tensor(self.runtime(), dtype, &shape)?;
        let runtime = self.runtime().clone();
        let mut dispatcher = Dispatcher::for_entries(&runtime, &[Entry::Binary])?;
        let status = runtime
            .upload_bytes(&[0u8; 4])
            .map_err(|error| crate::Error::runtime_state_source(name, error))?;
        with_scalar!(dtype, |T| {
            dispatcher.binary(
                op,
                kernel,
                t4a_cuda_kernels::ScalarSide::Both,
                len,
                self.allocation_of::<T>(&output, name)?,
                self.allocation_of::<T>(left, name)?,
                self.allocation_of::<T>(right, name)?,
                &status,
            )
        })?;
        self.runtime()
            .synchronize()
            .map_err(|error| crate::Error::runtime_state_source(name, error))?;
        let flag = self
            .runtime()
            .download_bytes(&status)
            .map_err(|error| crate::Error::runtime_state_source(name, error))?;
        if flag.iter().any(|byte| *byte != 0) {
            return Err(crate::Error::unsupported(
                name,
                "the checked remainder reported a division by zero",
            ));
        }
        Ok(output)
    }
}

impl tenferro_tensor::TensorElementwise for CudaEngine {
    fn add_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        self.elementwise_binary(lhs, rhs, Elementwise::Add, "add_read")
    }

    fn sub_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        self.elementwise_binary(lhs, rhs, Elementwise::Sub, "sub_read")
    }

    fn mul_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        self.elementwise_binary(lhs, rhs, Elementwise::Mul, "mul_read")
    }

    fn div_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        self.elementwise_binary(lhs, rhs, Elementwise::Div, "div_read")
    }

    fn neg_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.elementwise_unary(input, t4a_cuda_kernels::UnaryOp::Neg, "neg_read")
    }

    fn conj_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.elementwise_unary(input, t4a_cuda_kernels::UnaryOp::Conj, "conj_read")
    }

    fn abs_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.elementwise_unary(input, t4a_cuda_kernels::UnaryOp::Abs, "abs_read")
    }

    fn sign_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.elementwise_unary(input, t4a_cuda_kernels::UnaryOp::Sign, "sign_read")
    }

    fn maximum_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        self.elementwise_extreme(lhs, rhs, t4a_cuda_kernels::BinaryOp::Max, "maximum_read")
    }

    fn minimum_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        self.elementwise_extreme(lhs, rhs, t4a_cuda_kernels::BinaryOp::Min, "minimum_read")
    }

    fn clamp_read(
        &mut self,
        input: TensorRead<'_>,
        lower: TensorRead<'_>,
        upper: TensorRead<'_>,
    ) -> crate::Result<Tensor> {
        let name = "clamp_read";
        let source = self.device_read(input, name)?;
        let low = self.device_read(lower, name)?;
        let high = self.device_read(upper, name)?;
        if source.shape() != low.shape()
            || source.shape() != high.shape()
            || source.dtype() != low.dtype()
            || source.dtype() != high.dtype()
        {
            return Err(crate::Error::unsupported(
                name,
                "input and both per-element bounds must share a shape and a dtype",
            ));
        }
        let dtype = source.dtype();
        let kernel = kernel_dtype(dtype, name)?;
        let shape = source.shape().to_vec();
        let len = count_i64(&shape, name)?;
        let output = alloc_tensor(self.runtime(), dtype, &shape)?;
        let runtime = self.runtime().clone();
        let mut dispatcher = Dispatcher::for_entries(&runtime, &[Entry::Clamp])?;
        with_scalar!(dtype, |T| {
            dispatcher.clamp_buffers(
                kernel,
                len,
                self.allocation_of::<T>(&output, name)?,
                self.allocation_of::<T>(source, name)?,
                self.allocation_of::<T>(low, name)?,
                self.allocation_of::<T>(high, name)?,
            )
        })?;
        self.runtime()
            .synchronize()
            .map_err(|error| crate::Error::runtime_state_source(name, error))?;
        Ok(output)
    }

    fn compare_read(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        dir: &tenferro_tensor::CompareDir,
    ) -> crate::Result<Tensor> {
        let name = "compare_read";
        let left = self.device_read(lhs, name)?;
        let right = self.device_read(rhs, name)?;
        if left.shape() != right.shape() || left.dtype() != right.dtype() {
            return Err(crate::Error::unsupported(
                name,
                "both operands must share a shape and a dtype on this path",
            ));
        }
        let dtype = left.dtype();
        let kernel = kernel_dtype(dtype, name)?;
        let op = match dir {
            tenferro_tensor::CompareDir::Eq => t4a_cuda_kernels::CompareOp::Eq,
            tenferro_tensor::CompareDir::Lt => t4a_cuda_kernels::CompareOp::Lt,
            tenferro_tensor::CompareDir::Le => t4a_cuda_kernels::CompareOp::Le,
            tenferro_tensor::CompareDir::Gt => t4a_cuda_kernels::CompareOp::Gt,
            tenferro_tensor::CompareDir::Ge => t4a_cuda_kernels::CompareOp::Ge,
        };
        let shape = left.shape().to_vec();
        let len = count_i64(&shape, name)?;
        // The comparison writes one Bool byte per element pair.
        let output = alloc_tensor(self.runtime(), DType::Bool, &shape)?;
        let runtime = self.runtime().clone();
        let mut dispatcher = Dispatcher::for_entries(&runtime, &[Entry::Compare])?;
        let out = self.allocation_of::<bool>(&output, name)?;
        with_scalar!(dtype, |T| {
            dispatcher.compare(
                op,
                kernel,
                len,
                out,
                self.allocation_of::<T>(left, name)?,
                self.allocation_of::<T>(right, name)?,
            )
        })?;
        self.runtime()
            .synchronize()
            .map_err(|error| crate::Error::runtime_state_source(name, error))?;
        Ok(output)
    }

    fn select_read(
        &mut self,
        pred: TensorRead<'_>,
        on_true: TensorRead<'_>,
        on_false: TensorRead<'_>,
    ) -> crate::Result<Tensor> {
        let name = "select_read";
        let predicate = self.device_read(pred, name)?;
        let when_true = self.device_read(on_true, name)?;
        let when_false = self.device_read(on_false, name)?;
        if predicate.dtype() != DType::Bool {
            return Err(crate::Error::unsupported(
                name,
                "the predicate must be a boolean tensor",
            ));
        }
        if when_true.shape() != when_false.shape() || when_true.dtype() != when_false.dtype() {
            return Err(crate::Error::unsupported(
                name,
                "the two branches must share a shape and a dtype",
            ));
        }
        if predicate.shape() != when_true.shape() {
            return Err(crate::Error::unsupported(
                name,
                "the predicate must share the branches' shape on this path",
            ));
        }
        let dtype = when_true.dtype();
        let kernel = kernel_dtype(dtype, name)?;
        let shape = when_true.shape().to_vec();
        let len = count_i64(&shape, name)?;
        let output = alloc_tensor(self.runtime(), dtype, &shape)?;
        let runtime = self.runtime().clone();
        let mut dispatcher = Dispatcher::for_entries(&runtime, &[Entry::Select])?;
        let predicate_buffer = self.allocation_of::<bool>(predicate, name)?;
        with_scalar!(dtype, |T| {
            dispatcher.select(
                kernel,
                len,
                self.allocation_of::<T>(&output, name)?,
                predicate_buffer,
                self.allocation_of::<T>(when_true, name)?,
                self.allocation_of::<T>(when_false, name)?,
            )
        })?;
        self.runtime()
            .synchronize()
            .map_err(|error| crate::Error::runtime_state_source(name, error))?;
        Ok(output)
    }

    fn elementwise_read_into(
        &mut self,
        op: tenferro_tensor::ElementwiseReadOp,
        inputs: &[TensorRead<'_>],
        out: TensorWrite<'_>,
    ) -> crate::Result<()> {
        let name = "elementwise_read_into";
        let destination = match out {
            TensorWrite::Tensor(tensor) => tensor,
            TensorWrite::View(_) => {
                return Err(crate::Error::unsupported(
                    name,
                    "a borrowed view destination is not served by the structural slice",
                ))
            }
        };
        if is_host(destination) {
            return Err(crate::Error::unsupported(
                name,
                "a host destination would need an implicit transfer",
            ));
        }
        let dtype = destination.dtype();
        let kernel = kernel_dtype(dtype, name)?;
        let shape = destination.shape().to_vec();
        let len = count_i64(&shape, name)?;
        // Each shape carries the operands it needs: one for the unary forms, two for the
        // binary ones.
        let unary = matches!(
            op,
            tenferro_tensor::ElementwiseReadOp::Negate | tenferro_tensor::ElementwiseReadOp::Conj
        );
        let expected = if unary { 1 } else { 2 };
        if inputs.len() != expected {
            return Err(crate::Error::unsupported(
                name,
                format!(
                    "this operation takes {expected} operand(s), got {}",
                    inputs.len()
                ),
            ));
        }
        let sources: Vec<&Tensor> = inputs
            .iter()
            .map(|input| self.device_read(input.clone(), name))
            .collect::<crate::Result<_>>()?;
        for source in &sources {
            if source.shape() != shape.as_slice() || source.dtype() != dtype {
                return Err(crate::Error::unsupported(
                    name,
                    "every operand must share the destination's shape and dtype on this path",
                ));
            }
        }
        let runtime = self.runtime().clone();
        match op {
            tenferro_tensor::ElementwiseReadOp::Negate => {
                let mut dispatcher = Dispatcher::for_entries(&runtime, &[Entry::Unary])?;
                with_scalar!(dtype, |T| {
                    dispatcher.unary(
                        t4a_cuda_kernels::UnaryOp::Neg,
                        kernel,
                        kernel,
                        len,
                        self.allocation_of::<T>(destination, name)?,
                        self.allocation_of::<T>(sources[0], name)?,
                    )
                })?;
            }
            tenferro_tensor::ElementwiseReadOp::Conj => {
                let mut dispatcher = Dispatcher::for_entries(&runtime, &[Entry::Unary])?;
                with_scalar!(dtype, |T| {
                    dispatcher.unary(
                        t4a_cuda_kernels::UnaryOp::Conj,
                        kernel,
                        kernel,
                        len,
                        self.allocation_of::<T>(destination, name)?,
                        self.allocation_of::<T>(sources[0], name)?,
                    )
                })?;
            }
            binary => {
                let entry = match binary {
                    tenferro_tensor::ElementwiseReadOp::Add => Entry::Add,
                    tenferro_tensor::ElementwiseReadOp::Subtract => Entry::Sub,
                    tenferro_tensor::ElementwiseReadOp::Multiply => Entry::Mul,
                    _ => Entry::Div,
                };
                let mut dispatcher = Dispatcher::for_entries(&runtime, &[entry])?;
                with_scalar!(dtype, |T| {
                    dispatcher.arithmetic(
                        entry,
                        kernel,
                        len,
                        self.allocation_of::<T>(destination, name)?,
                        self.allocation_of::<T>(sources[0], name)?,
                        self.allocation_of::<T>(sources[1], name)?,
                    )
                })?;
            }
        }
        self.runtime()
            .synchronize()
            .map_err(|error| crate::Error::runtime_state_source(name, error))
    }

    fn rem(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
        self.remainder(
            TensorRead::from_tensor(lhs),
            TensorRead::from_tensor(rhs),
            "rem",
        )
    }

    fn rem_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        self.remainder(lhs, rhs, "rem_read")
    }
}
