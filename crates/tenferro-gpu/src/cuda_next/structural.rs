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
    plan_materialize_strided, plan_tril, plan_triu, CAxisValues, CLayout, Entry, GatherPlan,
    Layout, ScatterPlan,
};
use t4a_cuda_runtime::{Arg, LaunchConfig};

use super::cutensor::{Contraction, Operand};
use super::dispatch::{pod_arg, Dispatcher};
use super::engine::CudaEngine;
use super::launch::layout_arg;
use super::runtime::CudaRuntime;
use super::storage::CudaStorage;

/// Scalars the structural slice carries; a kernel family still refuses the dtypes its
/// catalogue does not list.
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
            DType::C32 => {
                #[allow(dead_code)]
                type $t = tenferro_tensor::Complex32;
                $body
            }
            DType::C64 => {
                #[allow(dead_code)]
                type $t = tenferro_tensor::Complex64;
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
    let storage = super::storage::fresh_storage(runtime, buffer);
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
        Ok(storage.allocation())
    }

    /// A compact device copy of `source` into fresh storage, enqueued on the stream and
    /// **not** waited for, for a route whose own submissions order after it.
    fn enqueue_copy(
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
        Ok(output)
    }

    /// A compact device copy of `source` into fresh storage of this runtime, labelled with
    /// `shape` (which must hold the same element count), complete when it returns.
    fn copy_to_fresh(
        &mut self,
        source: &Tensor,
        shape: Vec<usize>,
        op: &'static str,
    ) -> crate::Result<Tensor> {
        let output = self.enqueue_copy(source, shape, op)?;
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
        if plan.grid[0] == 0 {
            // Nothing to visit: the destination keeps the state the allocation left it in.
            return Ok(());
        }
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

/// The kernel crate's dtype tag for a contraction, which cuTENSOR serves for all four.
fn contraction_dtype(dtype: DType, op: &'static str) -> crate::Result<t4a_cuda_kernels::DType> {
    Ok(match dtype {
        DType::F32 => t4a_cuda_kernels::DType::F32,
        DType::F64 => t4a_cuda_kernels::DType::F64,
        DType::C32 => t4a_cuda_kernels::DType::C32,
        DType::C64 => t4a_cuda_kernels::DType::C64,
        other => {
            return Err(crate::Error::unsupported(
                op,
                format!("a contraction serves F32, F64, C32 and C64, not {other:?}"),
            ))
        }
    })
}

impl CudaEngine {
    /// The kernel layout of an owned device tensor: its shape, element strides and no
    /// offset.
    fn layout_of(&self, tensor: &Tensor, op: &'static str) -> crate::Result<Layout> {
        let extents: Vec<i64> = tensor.shape().iter().map(|&extent| extent as i64).collect();
        let strides: Vec<i64> = self
            .strides_of(tensor, op)?
            .iter()
            .map(|&stride| stride as i64)
            .collect();
        Layout::new(&extents, &strides, 0)
            .map_err(|error| crate::Error::runtime_state_source(op, error))
    }
}

/// The window a [`SliceConfig`](tenferro_tensor::SliceConfig) describes, checked the way the
/// kernel crate's planner checks it.
fn slice_window(
    shape: &[usize],
    config: &tenferro_tensor::SliceConfig,
) -> crate::Result<Vec<usize>> {
    let op = "slice";
    let rank = shape.len();
    if config.starts.len() != rank || config.limits.len() != rank || config.strides.len() != rank {
        return Err(crate::Error::rank_mismatch(op, rank, config.starts.len()));
    }
    config
        .starts
        .iter()
        .zip(&config.limits)
        .zip(&config.strides)
        .enumerate()
        .map(|(axis, ((&start, &limit), &stride))| {
            if start > limit {
                return Err(crate::Error::invalid_argument(
                    op,
                    "configuration",
                    format!("start exceeds limit on axis {axis}"),
                ));
            }
            if limit > shape[axis] {
                return Err(crate::Error::invalid_argument(
                    op,
                    "configuration",
                    format!(
                        "limit {limit} on axis {axis} exceeds the dimension {}",
                        shape[axis]
                    ),
                ));
            }
            if stride == 0 {
                return Err(crate::Error::invalid_argument(
                    op,
                    "configuration",
                    format!("stride must be positive on axis {axis}"),
                ));
            }
            Ok((limit - start).div_ceil(stride))
        })
        .collect()
}

/// The padded extents a [`PadConfig`](tenferro_tensor::PadConfig) describes.
fn padded_shape(shape: &[usize], config: &tenferro_tensor::PadConfig) -> crate::Result<Vec<usize>> {
    let op = "pad";
    let rank = shape.len();
    if config.edge_padding_low.len() != rank
        || config.edge_padding_high.len() != rank
        || config.interior_padding.len() != rank
    {
        return Err(crate::Error::rank_mismatch(
            op,
            rank,
            config.interior_padding.len(),
        ));
    }
    (0..rank)
        .map(|axis| {
            let interior = config.interior_padding[axis];
            if interior < 0 {
                return Err(crate::Error::invalid_argument(
                    op,
                    "configuration",
                    format!("interior padding must be non-negative on axis {axis}"),
                ));
            }
            // (n - 1) * (interior + 1) + 1 for a non-empty axis, zero for an empty one; the
            // kernel crate's planner derives the same base, so a mismatch is refused there.
            let overflow = || {
                crate::Error::invalid_argument(
                    op,
                    "configuration",
                    format!("the padded extent overflows on axis {axis}"),
                )
            };
            let spacing = interior.checked_add(1).ok_or_else(overflow)?;
            let extent = shape[axis] as i64;
            let spaced = if extent == 0 {
                0
            } else {
                (extent - 1)
                    .checked_mul(spacing)
                    .and_then(|value| value.checked_add(1))
                    .ok_or_else(overflow)?
            };
            let extent = spaced
                .checked_add(config.edge_padding_low[axis])
                .and_then(|value| value.checked_add(config.edge_padding_high[axis]))
                .ok_or_else(|| {
                    crate::Error::invalid_argument(
                        op,
                        "configuration",
                        format!("the padded extent overflows on axis {axis}"),
                    )
                })?;
            usize::try_from(extent).map_err(|_| {
                crate::Error::invalid_argument(
                    op,
                    "configuration",
                    format!("the padded extent is negative on axis {axis}"),
                )
            })
        })
        .collect()
}

/// The index dtype of a device index tensor, or a refusal naming the missing exactness check.
fn index_dtype_of(index_dtype: DType, op: &'static str) -> crate::Result<DType> {
    if matches!(index_dtype, DType::I32 | DType::I64) {
        return Ok(index_dtype);
    }
    Err(crate::Error::unsupported(
        op,
        format!(
            "device {index_dtype:?} indices need the exactness check this path does not run \
             yet; pass I32 or I64 indices"
        ),
    ))
}

/// Refuse a dimension list that repeats an axis or names one past `rank`.
fn require_axes(
    op: &'static str,
    argument: &'static str,
    axes: &[usize],
    rank: usize,
) -> crate::Result<()> {
    for (index, &axis) in axes.iter().enumerate() {
        if axis >= rank {
            return Err(crate::Error::invalid_argument(
                op,
                argument,
                format!("axis {axis} is out of range for rank {rank}"),
            ));
        }
        if axes[..index].contains(&axis) {
            return Err(crate::Error::invalid_argument(
                op,
                argument,
                format!("axis {axis} is listed twice"),
            ));
        }
    }
    Ok(())
}

/// The gather plan the kernel crate consumes, from the tensor surface's configuration.
fn gather_plan(config: &tenferro_tensor::GatherConfig, operand_rank: u32) -> GatherPlan {
    let collapsed: Vec<u32> = config
        .collapsed_slice_dims
        .iter()
        .map(|&axis| axis as u32)
        .collect();
    GatherPlan {
        window_dims: t4a_cuda_kernels::window_dims(operand_rank, &collapsed),
        offset_dims: config.offset_dims.iter().map(|&axis| axis as u32).collect(),
        start_index_map: config
            .start_index_map
            .iter()
            .map(|&axis| axis as u32)
            .collect(),
        slice_sizes: config
            .slice_sizes
            .iter()
            .map(|&extent| extent as i64)
            .collect(),
        index_vector_dim: config.index_vector_dim as u32,
    }
}

/// The output extents StableHLO gather produces: the window extents on the offset axes,
/// the index batch extents on the rest.
fn gather_shape(
    indices_shape: &[usize],
    plan: &GatherPlan,
    config: &tenferro_tensor::GatherConfig,
) -> crate::Result<Vec<usize>> {
    let op = "gather";
    let rank = indices_shape.len();
    let vector_axis = config.index_vector_dim;
    let has_vector = vector_axis < rank;
    let out_rank = rank - usize::from(has_vector) + plan.offset_dims.len();
    let mut batch = indices_shape
        .iter()
        .enumerate()
        .filter(|(axis, _)| !has_vector || *axis != vector_axis)
        .map(|(_, &extent)| extent);
    let mut out_shape = Vec::with_capacity(out_rank);
    for axis in 0..out_rank {
        match plan
            .offset_dims
            .iter()
            .position(|&dim| dim as usize == axis)
        {
            Some(position) => {
                let window = plan.window_dims.get(position).copied().ok_or_else(|| {
                    crate::Error::invalid_argument(
                        op,
                        "configuration",
                        format!("the output offset axis {axis} has no operand window axis"),
                    )
                })?;
                out_shape.push(config.slice_sizes[window as usize]);
            }
            None => out_shape.push(batch.next().ok_or_else(|| {
                crate::Error::invalid_argument(
                    op,
                    "configuration",
                    format!("the output axis {axis} has no index batch extent"),
                )
            })?),
        }
    }
    if let Some(extra) = batch.next() {
        return Err(crate::Error::invalid_argument(
            op,
            "configuration",
            format!("the index batch extent {extra} has no output axis"),
        ));
    }
    Ok(out_shape)
}

/// The scatter plan the kernel crate consumes, from the tensor surface's configuration.
fn scatter_plan(config: &tenferro_tensor::ScatterConfig, operand_rank: u32) -> ScatterPlan {
    let inserted: Vec<u32> = config
        .inserted_window_dims
        .iter()
        .map(|&axis| axis as u32)
        .collect();
    ScatterPlan {
        window_dims: t4a_cuda_kernels::window_dims(operand_rank, &inserted),
        update_window_dims: config
            .update_window_dims
            .iter()
            .map(|&axis| axis as u32)
            .collect(),
        scatter_dims: config
            .scatter_dims_to_operand_dims
            .iter()
            .map(|&axis| axis as u32)
            .collect(),
        index_vector_dim: config.index_vector_dim as u32,
    }
}

impl tenferro_tensor::TensorIndexing for CudaEngine {
    fn gather(
        &mut self,
        operand: &Tensor,
        start_indices: &Tensor,
        config: &tenferro_tensor::GatherConfig,
    ) -> crate::Result<Tensor> {
        let op = "gather";
        if is_host(operand) || is_host(start_indices) {
            return Err(crate::Error::unsupported(
                op,
                "a host operand would need an implicit transfer",
            ));
        }
        let dtype = operand.dtype();
        let kernel = kernel_dtype(dtype, op)?;
        let index_dtype = index_dtype_of(start_indices.dtype(), op)?;
        let index_kernel = kernel_dtype(index_dtype, op)?;
        let operand_shape = operand.shape().to_vec();
        let operand_rank = u32::try_from(operand_shape.len())
            .map_err(|_| crate::Error::unsupported(op, "the rank exceeds the ABI's int"))?;
        require_axes(
            op,
            "collapsed_slice_dims",
            &config.collapsed_slice_dims,
            operand_rank as usize,
        )?;
        // The output rank is not known yet; `gather_shape` checks these axes against it.
        require_axes(op, "offset_dims", &config.offset_dims, usize::MAX)?;
        require_axes(
            op,
            "start_index_map",
            &config.start_index_map,
            operand_rank as usize,
        )?;
        if config.slice_sizes.len() != operand_shape.len() {
            return Err(crate::Error::rank_mismatch(
                op,
                operand_shape.len(),
                config.slice_sizes.len(),
            ));
        }
        let plan = gather_plan(config, operand_rank);
        let indices_shape = start_indices.shape().to_vec();
        let out_shape = gather_shape(&indices_shape, &plan, config)?;
        let output = alloc_tensor(self.runtime(), dtype, &out_shape)?;
        let out_layout = self.layout_of(&output, op)?;
        let operand_layout = self.layout_of(operand, op)?;
        let indices_layout = self.layout_of(start_indices, op)?;
        let out_len = count_i64(&out_shape, op)?;
        let operand_len = count_i64(&operand_shape, op)?;
        let indices_len = count_i64(&indices_shape, op)?;
        let runtime = self.runtime().clone();
        let mut dispatcher = Dispatcher::for_entries(&runtime, &[Entry::Gather])?;
        with_scalar!(index_dtype, |I| {
            let indices_buffer = self.allocation_of::<I>(start_indices, op)?;
            with_scalar!(dtype, |T| {
                dispatcher.gather(
                    kernel,
                    index_kernel,
                    &out_layout,
                    out_len,
                    &operand_layout,
                    operand_len,
                    &indices_layout,
                    indices_len,
                    &plan,
                    self.allocation_of::<T>(&output, op)?,
                    self.allocation_of::<T>(operand, op)?,
                    indices_buffer,
                )
            })
        })?;
        self.runtime()
            .synchronize()
            .map_err(|error| crate::Error::runtime_state_source(op, error))?;
        Ok(output)
    }

    fn scatter(
        &mut self,
        operand: &Tensor,
        scatter_indices: &Tensor,
        updates: &Tensor,
        config: &tenferro_tensor::ScatterConfig,
    ) -> crate::Result<Tensor> {
        let op = "scatter";
        if is_host(operand) || is_host(scatter_indices) || is_host(updates) {
            return Err(crate::Error::unsupported(
                op,
                "a host operand would need an implicit transfer",
            ));
        }
        if updates.dtype() != operand.dtype() {
            return Err(crate::Error::dtype_mismatch(
                op,
                operand.dtype(),
                updates.dtype(),
            ));
        }
        let dtype = operand.dtype();
        let kernel = kernel_dtype(dtype, op)?;
        let index_dtype = index_dtype_of(scatter_indices.dtype(), op)?;
        let index_kernel = kernel_dtype(index_dtype, op)?;
        let operand_shape = operand.shape().to_vec();
        let operand_rank = u32::try_from(operand_shape.len())
            .map_err(|_| crate::Error::unsupported(op, "the rank exceeds the ABI's int"))?;
        let operand_rank_axes = operand_rank as usize;
        require_axes(
            op,
            "inserted_window_dims",
            &config.inserted_window_dims,
            operand_rank_axes,
        )?;
        require_axes(
            op,
            "scatter_dims_to_operand_dims",
            &config.scatter_dims_to_operand_dims,
            operand_rank_axes,
        )?;
        let plan = scatter_plan(config, operand_rank);
        // Additive scatter accumulates into the operand copy. The copy is only enqueued:
        // the accumulation below is submitted on the same stream, so stream order already
        // covers it and no host wait is needed.
        let output = self.enqueue_copy(operand, operand_shape.clone(), op)?;
        let out_layout = self.layout_of(&output, op)?;
        let indices_shape = scatter_indices.shape().to_vec();
        let updates_shape = updates.shape().to_vec();
        let indices_layout = self.layout_of(scatter_indices, op)?;
        let updates_layout = self.layout_of(updates, op)?;
        let out_len = count_i64(&operand_shape, op)?;
        let indices_len = count_i64(&indices_shape, op)?;
        let updates_len = count_i64(&updates_shape, op)?;
        let runtime = self.runtime().clone();
        let mut dispatcher = Dispatcher::for_entries(&runtime, &[Entry::ScatterAdd])?;
        with_scalar!(index_dtype, |I| {
            let indices_buffer = self.allocation_of::<I>(scatter_indices, op)?;
            with_scalar!(dtype, |T| {
                dispatcher.scatter_add(
                    kernel,
                    index_kernel,
                    &out_layout,
                    out_len,
                    &indices_layout,
                    indices_len,
                    &updates_layout,
                    updates_len,
                    &plan,
                    self.allocation_of::<T>(&output, op)?,
                    indices_buffer,
                    self.allocation_of::<T>(updates, op)?,
                )
            })
        })?;
        self.runtime()
            .synchronize()
            .map_err(|error| crate::Error::runtime_state_source(op, error))?;
        Ok(output)
    }

    fn slice(
        &mut self,
        input: &Tensor,
        config: &tenferro_tensor::SliceConfig,
    ) -> crate::Result<Tensor> {
        let op = "slice";
        if is_host(input) {
            return Err(crate::Error::unsupported(
                op,
                "a host operand would need an implicit transfer",
            ));
        }
        let shape = input.shape().to_vec();
        let out_shape = slice_window(&shape, config)?;
        let dtype = input.dtype();
        let kernel = kernel_dtype(dtype, op)?;
        let output = alloc_tensor(self.runtime(), dtype, &out_shape)?;
        let out_layout = self.layout_of(&output, op)?;
        let in_layout = self.layout_of(input, op)?;
        let out_len = count_i64(&out_shape, op)?;
        let in_len = count_i64(&shape, op)?;
        let starts: Vec<i64> = config.starts.iter().map(|&value| value as i64).collect();
        let steps: Vec<i64> = config.strides.iter().map(|&value| value as i64).collect();
        let runtime = self.runtime().clone();
        let mut dispatcher = Dispatcher::for_entries(&runtime, &[Entry::Slice])?;
        with_scalar!(dtype, |T| {
            dispatcher.slice(
                kernel,
                &out_layout,
                out_len,
                &in_layout,
                in_len,
                &starts,
                &steps,
                self.allocation_of::<T>(&output, op)?,
                self.allocation_of::<T>(input, op)?,
            )
        })?;
        self.runtime()
            .synchronize()
            .map_err(|error| crate::Error::runtime_state_source(op, error))?;
        Ok(output)
    }

    fn dynamic_slice(
        &mut self,
        input: &Tensor,
        starts: &Tensor,
        slice_sizes: &[usize],
    ) -> crate::Result<Tensor> {
        let op = "dynamic_slice";
        if is_host(input) || is_host(starts) {
            return Err(crate::Error::unsupported(
                op,
                "a host operand would need an implicit transfer",
            ));
        }
        let shape = input.shape().to_vec();
        if slice_sizes.len() != shape.len() {
            return Err(crate::Error::rank_mismatch(
                op,
                shape.len(),
                slice_sizes.len(),
            ));
        }
        if let Some(axis) = (0..shape.len()).find(|&axis| slice_sizes[axis] > shape[axis]) {
            return Err(crate::Error::invalid_argument(
                op,
                "slice_sizes",
                format!(
                    "extent {} on axis {axis} exceeds the dimension {}",
                    slice_sizes[axis], shape[axis]
                ),
            ));
        }
        if element_count(starts.shape())? != shape.len() {
            return Err(crate::Error::invalid_argument(
                op,
                "starts",
                format!(
                    "the starts tensor holds {} values for a rank-{} operand",
                    element_count(starts.shape())?,
                    shape.len()
                ),
            ));
        }
        let index_dtype = index_dtype_of(starts.dtype(), op)?;
        let dtype = input.dtype();
        let kernel = kernel_dtype(dtype, op)?;
        let index_kernel = kernel_dtype(index_dtype, op)?;
        let out_shape = slice_sizes.to_vec();
        let output = alloc_tensor(self.runtime(), dtype, &out_shape)?;
        let out_layout = self.layout_of(&output, op)?;
        let in_layout = self.layout_of(input, op)?;
        let out_len = count_i64(&out_shape, op)?;
        let in_len = count_i64(&shape, op)?;
        let starts_rank = u32::try_from(out_shape.len())
            .map_err(|_| crate::Error::unsupported(op, "the rank exceeds the ABI's int"))?;
        let runtime = self.runtime().clone();
        let mut dispatcher = Dispatcher::for_entries(&runtime, &[Entry::DynamicSlice])?;
        with_scalar!(index_dtype, |I| {
            let start_buffer = self.allocation_of::<I>(starts, op)?;
            with_scalar!(dtype, |T| {
                dispatcher.dynamic_slice(
                    kernel,
                    index_kernel,
                    &out_layout,
                    out_len,
                    &in_layout,
                    in_len,
                    starts_rank,
                    self.allocation_of::<T>(&output, op)?,
                    self.allocation_of::<T>(input, op)?,
                    start_buffer,
                )
            })
        })?;
        self.runtime()
            .synchronize()
            .map_err(|error| crate::Error::runtime_state_source(op, error))?;
        Ok(output)
    }

    fn dynamic_update_slice(
        &mut self,
        _operand: &Tensor,
        _update: &Tensor,
        _starts: &Tensor,
    ) -> crate::Result<Tensor> {
        Err(crate::Error::unsupported(
            "dynamic_update_slice",
            "the CubeCL path does not serve this operation either; it needs a windowed write \
             kernel the catalogue does not have",
        ))
    }

    fn concatenate(&mut self, inputs: &[&Tensor], axis: usize) -> crate::Result<Tensor> {
        let op = "concatenate";
        let Some(first) = inputs.first().copied() else {
            return Err(crate::Error::invalid_argument(
                op,
                "inputs",
                "concatenate requires at least one input",
            ));
        };
        let dtype = first.dtype();
        let shape = first.shape().to_vec();
        let rank = shape.len();
        if axis >= rank {
            return Err(crate::Error::axis_out_of_bounds(op, axis, rank));
        }
        let mut out_shape = shape.clone();
        out_shape[axis] = 0;
        for input in inputs {
            if is_host(input) {
                return Err(crate::Error::unsupported(
                    op,
                    "a host operand would need an implicit transfer",
                ));
            }
            if input.dtype() != dtype {
                return Err(crate::Error::dtype_mismatch(op, dtype, input.dtype()));
            }
            if input.shape().len() != rank {
                return Err(crate::Error::rank_mismatch(op, rank, input.shape().len()));
            }
            for (other, (&extent, &expected)) in input.shape().iter().zip(&shape).enumerate() {
                if other != axis && extent != expected {
                    return Err(crate::Error::shape_mismatch(
                        op,
                        input.shape().to_vec(),
                        shape.clone(),
                    ));
                }
            }
            let extent = out_shape[axis];
            out_shape[axis] = extent.checked_add(input.shape()[axis]).ok_or_else(|| {
                crate::Error::invalid_argument(
                    op,
                    "inputs",
                    format!("the concatenated extent overflows on axis {axis}"),
                )
            })?;
        }
        let kernel = kernel_dtype(dtype, op)?;
        let output = alloc_tensor(self.runtime(), dtype, &out_shape)?;
        let out_layout = self.layout_of(&output, op)?;
        let out_len = count_i64(&out_shape, op)?;
        let runtime = self.runtime().clone();
        let mut dispatcher = Dispatcher::for_entries(&runtime, &[Entry::ConcatenateCopy])?;
        let mut offset = 0usize;
        for input in inputs {
            // The kernel reads a compact source, so a strided input is compacted first.
            let compact = if compact_strides(input.shape()) == self.strides_of(input, op)? {
                None
            } else {
                Some(self.copy_to_fresh(input, input.shape().to_vec(), op)?)
            };
            let source = compact.as_ref().unwrap_or(input);
            let in_layout = self.layout_of(source, op)?;
            let in_len = count_i64(source.shape(), op)?;
            let axis_u32 = u32::try_from(axis)
                .map_err(|_| crate::Error::unsupported(op, "the rank exceeds the ABI's int"))?;
            let axis_offset = i64::try_from(offset).map_err(|_| {
                crate::Error::invalid_argument(
                    op,
                    "inputs",
                    "the concatenation offset is too large",
                )
            })?;
            with_scalar!(dtype, |T| {
                dispatcher.concatenate_copy(
                    kernel,
                    &out_layout,
                    out_len,
                    &in_layout,
                    in_len,
                    axis_u32,
                    axis_offset,
                    self.allocation_of::<T>(&output, op)?,
                    self.allocation_of::<T>(source, op)?,
                )
            })?;
            offset = offset.checked_add(source.shape()[axis]).ok_or_else(|| {
                crate::Error::invalid_argument(op, "inputs", "the concatenation offset overflows")
            })?;
        }
        self.runtime()
            .synchronize()
            .map_err(|error| crate::Error::runtime_state_source(op, error))?;
        Ok(output)
    }

    fn reverse(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
        let op = "reverse";
        if is_host(input) {
            return Err(crate::Error::unsupported(
                op,
                "a host operand would need an implicit transfer",
            ));
        }
        let shape = input.shape().to_vec();
        let strides = self.strides_of(input, op)?;
        let mut view_strides: Vec<i64> = strides.iter().map(|&stride| stride as i64).collect();
        let mut offset = 0i64;
        for (index, &axis) in axes.iter().enumerate() {
            if axis >= shape.len() {
                return Err(crate::Error::axis_out_of_bounds(op, axis, shape.len()));
            }
            if axes[..index].contains(&axis) {
                return Err(crate::Error::invalid_argument(
                    op,
                    "axes",
                    format!("axis {axis} is listed twice"),
                ));
            }
            let stride = view_strides[axis];
            offset += (shape[axis] as i64 - 1) * stride;
            view_strides[axis] = -stride;
        }
        let extents: Vec<i64> = shape.iter().map(|&extent| extent as i64).collect();
        let view = Layout::new(&extents, &view_strides, offset)
            .map_err(|error| crate::Error::runtime_state_source(op, error))?;
        let dtype = input.dtype();
        let kernel = kernel_dtype(dtype, op)?;
        let source_len = count_i64(&shape, op)?;
        let plan = plan_materialize_strided(kernel, &view, source_len)
            .map_err(|error| crate::Error::runtime_state_source(op, error))?;
        let output = alloc_tensor(self.runtime(), dtype, &shape)?;
        let layout = CLayout::new(&view);
        let (destination, source) = self.two_allocations(&output, input, op)?;
        self.launch_planned(
            Entry::MaterializeStrided,
            plan,
            &[
                Arg::DevicePtr(destination),
                Arg::DevicePtr(source),
                Arg::U32(kernel.tag() as u32),
                layout_arg(&layout),
                Arg::U64(source_len as u64),
            ],
            op,
        )?;
        Ok(output)
    }

    fn pad(
        &mut self,
        input: &Tensor,
        config: &tenferro_tensor::PadConfig,
    ) -> crate::Result<Tensor> {
        let op = "pad";
        if is_host(input) {
            return Err(crate::Error::unsupported(
                op,
                "a host operand would need an implicit transfer",
            ));
        }
        let shape = input.shape().to_vec();
        let out_shape = padded_shape(&shape, config)?;
        let dtype = input.dtype();
        let kernel = kernel_dtype(dtype, op)?;
        // An empty operand has no element to place, so every output element is padding and
        // the zeroed allocation is already the answer — and an empty allocation has no
        // device address to hand the kernel.
        let in_len = count_i64(&shape, op)?;
        let output = alloc_tensor(self.runtime(), dtype, &out_shape)?;
        if in_len == 0 {
            return Ok(output);
        }
        let out_layout = self.layout_of(&output, op)?;
        let in_layout = self.layout_of(input, op)?;
        let out_len = count_i64(&out_shape, op)?;
        let runtime = self.runtime().clone();
        let mut dispatcher = Dispatcher::for_entries(&runtime, &[Entry::Pad])?;
        with_scalar!(dtype, |T| {
            dispatcher.pad(
                kernel,
                &out_layout,
                out_len,
                &in_layout,
                in_len,
                &config.edge_padding_low,
                &config.edge_padding_high,
                &config.interior_padding,
                self.allocation_of::<T>(&output, op)?,
                self.allocation_of::<T>(input, op)?,
            )
        })?;
        self.runtime()
            .synchronize()
            .map_err(|error| crate::Error::runtime_state_source(op, error))?;
        Ok(output)
    }
}

/// The cuTENSOR mode assignment of one dot-general.
struct DotModes {
    lhs_modes: Vec<i32>,
    rhs_modes: Vec<i32>,
    out_modes: Vec<i32>,
    out_shape: Vec<usize>,
}

/// Assign one mode per axis for both operands and the destination: batch axes first, then
/// the contracted axes shared by both operands, then each operand's free axes.
fn dot_modes(
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    config: &tenferro_tensor::DotGeneralConfig,
    op: &'static str,
) -> crate::Result<DotModes> {
    let (lhs_rank, rhs_rank) = (lhs_shape.len(), rhs_shape.len());
    if config.lhs_batch_dims.len() != config.rhs_batch_dims.len()
        || config.lhs_contracting_dims.len() != config.rhs_contracting_dims.len()
    {
        return Err(crate::Error::invalid_argument(
            op,
            "configuration",
            "the batch and contracting dimension lists must pair up between the operands",
        ));
    }
    require_axes(op, "lhs_batch_dims", &config.lhs_batch_dims, lhs_rank)?;
    require_axes(op, "rhs_batch_dims", &config.rhs_batch_dims, rhs_rank)?;
    require_axes(
        op,
        "lhs_contracting_dims",
        &config.lhs_contracting_dims,
        lhs_rank,
    )?;
    require_axes(
        op,
        "rhs_contracting_dims",
        &config.rhs_contracting_dims,
        rhs_rank,
    )?;
    for (&lhs_axis, &rhs_axis) in config.lhs_batch_dims.iter().zip(&config.rhs_batch_dims) {
        if lhs_shape[lhs_axis] != rhs_shape[rhs_axis] {
            return Err(crate::Error::shape_mismatch(
                op,
                vec![lhs_shape[lhs_axis]],
                vec![rhs_shape[rhs_axis]],
            ));
        }
    }
    for (&lhs_axis, &rhs_axis) in config
        .lhs_contracting_dims
        .iter()
        .zip(&config.rhs_contracting_dims)
    {
        if lhs_shape[lhs_axis] != rhs_shape[rhs_axis] {
            return Err(crate::Error::shape_mismatch(
                op,
                vec![lhs_shape[lhs_axis]],
                vec![rhs_shape[rhs_axis]],
            ));
        }
    }
    let batches = config.lhs_batch_dims.len();
    let contracted = config.lhs_contracting_dims.len();
    let mut next = (batches + contracted) as i32;
    let assign = |named: [&[usize]; 2], modes: &mut [i32], next: &mut i32| -> Vec<usize> {
        for (index, &axis) in named[0].iter().enumerate() {
            modes[axis] = index as i32;
        }
        for (index, &axis) in named[1].iter().enumerate() {
            modes[axis] = (batches + index) as i32;
        }
        let mut free = Vec::new();
        for (axis, mode) in modes.iter_mut().enumerate() {
            if *mode < 0 {
                *mode = *next;
                *next += 1;
                free.push(axis);
            }
        }
        free
    };
    let mut lhs_modes = vec![-1i32; lhs_rank];
    let mut rhs_modes = vec![-1i32; rhs_rank];
    let lhs_free = assign(
        [&config.lhs_batch_dims, &config.lhs_contracting_dims],
        &mut lhs_modes,
        &mut next,
    );
    let rhs_free = assign(
        [&config.rhs_batch_dims, &config.rhs_contracting_dims],
        &mut rhs_modes,
        &mut next,
    );
    let mut out_modes = Vec::with_capacity(batches + lhs_free.len() + rhs_free.len());
    let mut out_shape = Vec::with_capacity(out_modes.capacity());
    for (index, &axis) in config.lhs_batch_dims.iter().enumerate() {
        out_modes.push(index as i32);
        out_shape.push(lhs_shape[axis]);
    }
    for &axis in &lhs_free {
        out_modes.push(lhs_modes[axis]);
        out_shape.push(lhs_shape[axis]);
    }
    for &axis in &rhs_free {
        out_modes.push(rhs_modes[axis]);
        out_shape.push(rhs_shape[axis]);
    }
    Ok(DotModes {
        lhs_modes,
        rhs_modes,
        out_modes,
        out_shape,
    })
}

/// One contraction scalar as the `(re, im)` pair the library reads.
fn scalar_pair(value: tenferro_tensor::ContractionScalar) -> (f64, f64) {
    match value {
        tenferro_tensor::ContractionScalar::F32(value) => (value as f64, 0.0),
        tenferro_tensor::ContractionScalar::F64(value) => (value, 0.0),
        tenferro_tensor::ContractionScalar::C32(value) => (value.re as f64, value.im as f64),
        tenferro_tensor::ContractionScalar::C64(value) => (value.re, value.im),
    }
}

impl CudaEngine {
    /// Submit one dot-general into an already allocated destination and wait for it.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Validation`] for a configuration the operands' shapes do not support
    /// and [`crate::Error::RuntimeState`] for a cuTENSOR failure.
    #[allow(clippy::too_many_arguments)]
    fn submit_dot(
        &mut self,
        lhs: &Tensor,
        rhs: &Tensor,
        config: &tenferro_tensor::DotGeneralConfig,
        conj_a: bool,
        conj_b: bool,
        alpha: (f64, f64),
        beta: (f64, f64),
        addend: Option<&Tensor>,
        output: &Tensor,
        op: &'static str,
    ) -> crate::Result<()> {
        let dtype = lhs.dtype();
        let kernel = contraction_dtype(dtype, op)?;
        if rhs.dtype() != dtype {
            return Err(crate::Error::dtype_mismatch(op, dtype, rhs.dtype()));
        }
        if output.dtype() != dtype {
            return Err(crate::Error::dtype_mismatch(op, dtype, output.dtype()));
        }
        let modes = dot_modes(lhs.shape(), rhs.shape(), config, op)?;
        if output.shape() != modes.out_shape.as_slice() {
            return Err(crate::Error::shape_mismatch(
                op,
                modes.out_shape.clone(),
                output.shape().to_vec(),
            ));
        }
        let strides = |tensor: &Tensor, op: &'static str| -> crate::Result<Vec<i64>> {
            Ok(self
                .strides_of(tensor, op)?
                .iter()
                .map(|&stride| stride as i64)
                .collect())
        };
        let lhs_strides = strides(lhs, op)?;
        let rhs_strides = strides(rhs, op)?;
        let out_strides = strides(output, op)?;
        let lhs_extents: Vec<i64> = lhs.shape().iter().map(|&extent| extent as i64).collect();
        let rhs_extents: Vec<i64> = rhs.shape().iter().map(|&extent| extent as i64).collect();
        let out_extents: Vec<i64> = output.shape().iter().map(|&extent| extent as i64).collect();
        let (a, b, d) = with_scalar!(dtype, |T| Ok((
            self.allocation_of::<T>(lhs, op)?,
            self.allocation_of::<T>(rhs, op)?,
            self.allocation_of::<T>(output, op)?,
        )))?;
        let addend_parts = match addend {
            Some(c) => {
                if c.dtype() != dtype {
                    return Err(crate::Error::dtype_mismatch(op, dtype, c.dtype()));
                }
                if c.shape() != output.shape() {
                    return Err(crate::Error::shape_mismatch(
                        op,
                        output.shape().to_vec(),
                        c.shape().to_vec(),
                    ));
                }
                let buffer = with_scalar!(dtype, |T| self.allocation_of::<T>(c, op))?;
                Some((buffer, strides(c, op)?))
            }
            None => None,
        };
        let request = Contraction {
            dtype: kernel,
            a: Operand {
                buffer: a,
                modes: &modes.lhs_modes,
                extents: &lhs_extents,
                strides: &lhs_strides,
            },
            b: Operand {
                buffer: b,
                modes: &modes.rhs_modes,
                extents: &rhs_extents,
                strides: &rhs_strides,
            },
            c: addend_parts.as_ref().map(|(buffer, strides)| Operand {
                buffer,
                modes: &modes.out_modes,
                extents: &out_extents,
                strides,
            }),
            d: Operand {
                buffer: d,
                modes: &modes.out_modes,
                extents: &out_extents,
                strides: &out_strides,
            },
            alpha,
            beta,
            conj_a,
            conj_b,
        };
        self.cutensor()
            .map_err(|error| crate::Error::runtime_state(op, error.to_string()))?
            .contract(&request)
            .map_err(|error| crate::Error::runtime_state(op, error.to_string()))?;
        self.runtime()
            .synchronize()
            .map_err(|error| crate::Error::runtime_state_source(op, error))
    }

    /// Allocate the destination of one dot-general, submit it and return it.
    fn dot_general_allocating(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &tenferro_tensor::DotGeneralConfig,
        conj_a: bool,
        conj_b: bool,
        op: &'static str,
    ) -> crate::Result<Tensor> {
        let left = self.device_read(lhs, op)?;
        let right = self.device_read(rhs, op)?;
        let dtype = left.dtype();
        let modes = dot_modes(left.shape(), right.shape(), config, op)?;
        let output = alloc_tensor(self.runtime(), dtype, &modes.out_shape)?;
        self.submit_dot(
            left,
            right,
            config,
            conj_a,
            conj_b,
            (1.0, 0.0),
            (0.0, 0.0),
            None,
            &output,
            op,
        )?;
        Ok(output)
    }
}

impl tenferro_tensor::TensorDot for CudaEngine {
    fn dot_general_read(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &tenferro_tensor::DotGeneralConfig,
    ) -> crate::Result<Tensor> {
        self.dot_general_allocating(lhs, rhs, config, false, false, "dot_general_read")
    }

    fn dot_general_read_into(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &tenferro_tensor::DotGeneralConfig,
        out: TensorWrite<'_>,
    ) -> crate::Result<()> {
        let op = "dot_general_read_into";
        let (TensorRead::Tensor(left), TensorWrite::Tensor(output)) = (lhs, out) else {
            return Err(crate::Error::unsupported(
                op,
                "a borrowed view is not served by the structural slice",
            ));
        };
        self.submit_dot(
            left,
            rhs_tensor(rhs, op)?,
            config,
            false,
            false,
            (1.0, 0.0),
            (0.0, 0.0),
            None,
            output,
            op,
        )
    }

    fn dot_general_with_conj(
        &mut self,
        lhs: &Tensor,
        rhs: &Tensor,
        config: &tenferro_tensor::DotGeneralConfig,
        lhs_conj: bool,
        rhs_conj: bool,
    ) -> crate::Result<Tensor> {
        self.dot_general_allocating(
            TensorRead::from_tensor(lhs),
            TensorRead::from_tensor(rhs),
            config,
            lhs_conj,
            rhs_conj,
            "dot_general_with_conj",
        )
    }

    fn dot_general_with_conj_read(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &tenferro_tensor::DotGeneralConfig,
        lhs_conj: bool,
        rhs_conj: bool,
    ) -> crate::Result<Tensor> {
        self.dot_general_allocating(
            lhs,
            rhs,
            config,
            lhs_conj,
            rhs_conj,
            "dot_general_with_conj_read",
        )
    }

    fn dot_general_read_into_accum(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &tenferro_tensor::DotGeneralConfig,
        accumulation: tenferro_tensor::DotGeneralAccumulation,
        out: TensorWrite<'_>,
    ) -> crate::Result<()> {
        let op = "dot_general_read_into_accum";
        let TensorWrite::Tensor(output) = out else {
            return Err(crate::Error::unsupported(
                op,
                "a borrowed view is not served by the structural slice",
            ));
        };
        let left = self.device_read(lhs, op)?;
        let right = rhs_tensor(rhs, op)?;
        self.submit_dot(
            left,
            right,
            config,
            accumulation.lhs_conj,
            accumulation.rhs_conj,
            scalar_pair(accumulation.alpha),
            scalar_pair(accumulation.beta),
            Some(output),
            output,
            op,
        )
    }
}

/// The device tensor behind a read operand, or the trait's refusal.
fn rhs_tensor<'a>(input: TensorRead<'a>, op: &'static str) -> crate::Result<&'a Tensor> {
    match input {
        TensorRead::Tensor(tensor) => Ok(tensor),
        TensorRead::View(_) => Err(crate::Error::unsupported(
            op,
            "a borrowed view is not served by the structural slice",
        )),
    }
}
