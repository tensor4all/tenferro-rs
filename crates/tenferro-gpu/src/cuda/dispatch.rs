//! Per-family dispatch: plan through the kernel crate, marshal the ABI arguments, launch.
//!
//! Every plan comes from the kernel crate's `plan_*` function — this module adds no
//! validation of its own — and every argument order follows the documented ABI.

use t4a_cuda_kernels::{
    plan_binary, plan_clamp, plan_compare, plan_concatenate_copy, plan_convert, plan_copy_exact,
    plan_dynamic_slice, plan_elementwise, plan_gather, plan_pad, plan_reduce, plan_reduction,
    plan_scatter_add, plan_select, plan_slice, plan_unary, plan_validate_float_indices,
    plan_validate_real_cast, BinaryOp, DType, Entry, GatherPlan, Layout, ReduceOp, ReductionPlan,
    ScalarSide, ScatterPlan, UnaryOp,
};
use t4a_cuda_runtime::{Arg, LaunchConfig};

use super::engine::CudaEngine;
use super::launch::{layout_arg, KernelModules};

/// One runtime's dispatcher over a prepared module set.
#[doc(hidden)]
pub struct Dispatcher {
    modules: KernelModules,
}

/// Dispatchers whose translation units have been prepared, reused across operations.
///
/// A [`KernelModules`] prepares its artifacts through the runtime, which is a round trip of about
/// a millisecond per module set; an engine that rebuilt one per operation paid it on every call.
#[derive(Default)]
pub(crate) struct PreparedDispatchers {
    entries: Vec<Entry>,
    dispatcher: Option<Dispatcher>,
}

impl PreparedDispatchers {
    /// A dispatcher that covers `entries`, extending the prepared set when it does not yet.
    fn ensure(
        &mut self,
        runtime: &super::runtime::CudaRuntime,
        entries: &[Entry],
    ) -> crate::Result<&mut Dispatcher> {
        let covered = entries.iter().all(|entry| self.entries.contains(entry));
        if !covered {
            let mut wanted = self.entries.clone();
            for entry in entries {
                if !wanted.contains(entry) {
                    wanted.push(*entry);
                }
            }
            self.dispatcher = Some(Dispatcher::for_entries(runtime, &wanted)?);
            self.entries = wanted;
        }
        self.dispatcher
            .as_mut()
            .ok_or_else(|| crate::Error::runtime_state("cuda prepare", "no dispatcher was kept"))
    }
}

impl CudaEngine {
    /// The engine's shared dispatcher, prepared for `entries`.
    ///
    /// # Errors
    ///
    /// The runtime's failure to prepare a translation unit.
    pub(crate) fn dispatcher(
        &self,
        entries: &[Entry],
    ) -> crate::Result<std::cell::RefMut<'_, Dispatcher>> {
        let mut cell = self.prepared.borrow_mut();
        cell.ensure(self.runtime(), entries)?;
        Ok(std::cell::RefMut::map(cell, |prepared| {
            prepared
                .dispatcher
                .as_mut()
                .expect("the dispatcher was prepared above")
        }))
    }
}

impl Dispatcher {
    /// Prepare exactly the modules `entries` need.
    ///
    /// # Errors
    ///
    /// [`crate::Error::RuntimeStateSource`] when NVRTC rejects a module or the runtime
    /// cannot prepare its artifact.
    pub(crate) fn for_entries(
        runtime: &super::runtime::CudaRuntime,
        entries: &[Entry],
    ) -> crate::Result<Self> {
        Ok(Dispatcher {
            modules: KernelModules::for_entries(runtime, entries)?,
        })
    }

    /// Enqueue one launch with the plan's own geometry.
    fn run(
        &mut self,
        entry: Entry,
        plan: t4a_cuda_kernels::Launch,
        args: &[Arg<'_>],
    ) -> crate::Result<()> {
        if plan.grid[0] == 0 {
            // An empty domain has no element to visit: an element-driven family leaves the
            // destination as its allocation found it. A reduction is the exception — its
            // identity fill is the kernel's, so an empty reduction keeps whatever the
            // planner and the runtime decide.
            match entry {
                Entry::Reduce | Entry::ReducePartials => {}
                _ => return Ok(()),
            }
        }
        // SAFETY: every caller below builds the argument list in the documented ABI order
        // after the kernel crate's planner accepted the invocation.
        unsafe {
            self.modules
                .launch(entry, LaunchConfig::new(plan.grid[0], plan.block[0]), args)
        }
    }

    /// `t4a_cuda_copy_exact`: a compact-to-compact byte copy of `len` elements.
    ///
    /// # Errors
    ///
    /// [`crate::Error::RuntimeState`] when the kernel crate's planner refuses the
    /// invocation, and [`crate::Error::RuntimeStateSource`] when binding or admitting the
    /// launch fails.
    pub(crate) fn copy_exact(
        &mut self,
        dtype: DType,
        len: i64,
        out: &t4a_cuda_runtime::Buffer,
        input: &t4a_cuda_runtime::Buffer,
    ) -> crate::Result<()> {
        let plan = plan_copy_exact(dtype, len).map_err(plan_error)?;
        self.run(
            Entry::CopyExact,
            plan,
            &[
                Arg::DevicePtr(out),
                Arg::DevicePtr(input),
                Arg::U32(dtype.tag() as u32),
                Arg::U64(len as u64),
            ],
        )
    }

    /// One arithmetic elementwise entry (`add`, `sub`, `mul`, `div`) over `len` elements.
    ///
    /// # Errors
    ///
    /// [`crate::Error::RuntimeState`] when the kernel crate's planner refuses the
    /// invocation, and [`crate::Error::RuntimeStateSource`] when binding or admitting the
    /// launch fails.
    pub(crate) fn arithmetic(
        &mut self,
        entry: Entry,
        dtype: DType,
        len: i64,
        out: &t4a_cuda_runtime::Buffer,
        lhs: &t4a_cuda_runtime::Buffer,
        rhs: &t4a_cuda_runtime::Buffer,
    ) -> crate::Result<()> {
        let plan = plan_elementwise(entry, dtype, len).map_err(plan_error)?;
        self.run(
            entry,
            plan,
            &[
                Arg::DevicePtr(out),
                Arg::DevicePtr(lhs),
                Arg::DevicePtr(rhs),
                Arg::U32(dtype.tag() as u32),
                Arg::U64(len as u64),
            ],
        )
    }

    /// `t4a_cuda_clamp`: per-element lower and upper bounds over `len` elements.
    ///
    /// # Errors
    ///
    /// [`crate::Error::RuntimeState`] when the kernel crate's planner refuses the
    /// invocation, and [`crate::Error::RuntimeStateSource`] when binding or admitting the
    /// launch fails.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn clamp_buffers(
        &mut self,
        dtype: DType,
        len: i64,
        out: &t4a_cuda_runtime::Buffer,
        input: &t4a_cuda_runtime::Buffer,
        lower: &t4a_cuda_runtime::Buffer,
        upper: &t4a_cuda_runtime::Buffer,
    ) -> crate::Result<()> {
        let plan = plan_clamp(dtype, len).map_err(plan_error)?;
        self.run(
            Entry::Clamp,
            plan,
            &[
                Arg::DevicePtr(out),
                Arg::DevicePtr(input),
                Arg::DevicePtr(lower),
                Arg::DevicePtr(upper),
                Arg::U32(dtype.tag() as u32),
                Arg::U64(len as u64),
            ],
        )
    }

    /// `t4a_cuda_unary`: one unary operation over `len` elements.
    ///
    /// # Errors
    ///
    /// [`crate::Error::RuntimeState`] when the kernel crate's planner refuses the
    /// invocation, and [`crate::Error::RuntimeStateSource`] when binding or admitting the
    /// launch fails.
    pub(crate) fn unary(
        &mut self,
        op: UnaryOp,
        dst: DType,
        src: DType,
        len: i64,
        out: &t4a_cuda_runtime::Buffer,
        input: &t4a_cuda_runtime::Buffer,
    ) -> crate::Result<()> {
        let plan = plan_unary(op, dst, src, len).map_err(plan_error)?;
        self.run(
            Entry::Unary,
            plan,
            &[
                Arg::DevicePtr(out),
                Arg::DevicePtr(input),
                Arg::U32(dst.tag() as u32),
                Arg::U32(src.tag() as u32),
                Arg::I32(op.tag()),
                Arg::U64(len as u64),
            ],
        )
    }

    /// `t4a_cuda_binary`: one binary operation, optionally with a scalar side.
    ///
    /// The trailing status word is the entry's checked-integer error flag; the caller
    /// supplies it zeroed and inspects it afterwards.
    ///
    /// # Errors
    ///
    /// [`crate::Error::RuntimeState`] when the planner refuses the invocation, and
    /// [`crate::Error::RuntimeStateSource`] when binding or admitting the launch fails.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn binary(
        &mut self,
        op: BinaryOp,
        dtype: DType,
        side: ScalarSide,
        len: i64,
        out: &t4a_cuda_runtime::Buffer,
        lhs: &t4a_cuda_runtime::Buffer,
        rhs: &t4a_cuda_runtime::Buffer,
        status: &t4a_cuda_runtime::Buffer,
    ) -> crate::Result<()> {
        let plan = plan_binary(op, dtype, side, len).map_err(plan_error)?;
        self.run(
            Entry::Binary,
            plan,
            &[
                Arg::DevicePtr(out),
                Arg::DevicePtr(lhs),
                Arg::DevicePtr(rhs),
                Arg::U32(dtype.tag() as u32),
                Arg::I32(op.tag()),
                Arg::I32(side.tag()),
                Arg::U64(len as u64),
                Arg::DevicePtr(status),
            ],
        )
    }

    /// `t4a_cuda_compare`: ordered comparison of two operands into `len` Bool bytes.
    ///
    /// # Errors
    ///
    /// [`crate::Error::RuntimeState`] when the planner refuses the invocation, and
    /// [`crate::Error::RuntimeStateSource`] when binding or admitting the launch fails.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn compare(
        &mut self,
        op: t4a_cuda_kernels::CompareOp,
        dtype: DType,
        len: i64,
        out: &t4a_cuda_runtime::Buffer,
        lhs: &t4a_cuda_runtime::Buffer,
        rhs: &t4a_cuda_runtime::Buffer,
    ) -> crate::Result<()> {
        let plan = plan_compare(dtype, len).map_err(plan_error)?;
        self.run(
            Entry::Compare,
            plan,
            &[
                Arg::DevicePtr(out),
                Arg::DevicePtr(lhs),
                Arg::DevicePtr(rhs),
                Arg::U32(dtype.tag() as u32),
                Arg::I32(op.tag()),
                Arg::U64(len as u64),
            ],
        )
    }

    /// `t4a_cuda_select`: `len` predicate bytes choose between two operands.
    ///
    /// # Errors
    ///
    /// [`crate::Error::RuntimeState`] when the planner refuses the invocation, and
    /// [`crate::Error::RuntimeStateSource`] when binding or admitting the launch fails.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn select(
        &mut self,
        dtype: DType,
        len: i64,
        out: &t4a_cuda_runtime::Buffer,
        predicate: &t4a_cuda_runtime::Buffer,
        on_true: &t4a_cuda_runtime::Buffer,
        on_false: &t4a_cuda_runtime::Buffer,
    ) -> crate::Result<()> {
        let plan = plan_select(dtype, len).map_err(plan_error)?;
        self.run(
            Entry::Select,
            plan,
            &[
                Arg::DevicePtr(out),
                Arg::DevicePtr(predicate),
                Arg::DevicePtr(on_true),
                Arg::DevicePtr(on_false),
                Arg::U32(dtype.tag() as u32),
                Arg::U64(len as u64),
            ],
        )
    }

    /// `t4a_cuda_clamp`: clamp `len` elements between the two bounds.
    ///
    /// # Errors
    ///
    /// [`crate::Error::RuntimeState`] when the planner refuses the invocation, and
    /// [`crate::Error::RuntimeStateSource`] when binding or admitting the launch fails.
    #[allow(clippy::too_many_arguments)]
    // Only the dispatch tests exercise this form; the engine clamps through the buffer entry.
    #[allow(dead_code)]
    pub(crate) fn clamp(
        &mut self,
        dtype: DType,
        len: i64,
        out: &t4a_cuda_runtime::Buffer,
        input: &t4a_cuda_runtime::Buffer,
        lower: &t4a_cuda_runtime::Buffer,
        upper: &t4a_cuda_runtime::Buffer,
    ) -> crate::Result<()> {
        let plan = plan_clamp(dtype, len).map_err(plan_error)?;
        self.run(
            Entry::Clamp,
            plan,
            &[
                Arg::DevicePtr(out),
                Arg::DevicePtr(input),
                Arg::DevicePtr(lower),
                Arg::DevicePtr(upper),
                Arg::U32(dtype.tag() as u32),
                Arg::U64(len as u64),
            ],
        )
    }

    /// `t4a_cuda_convert`: convert `len` elements between two dtypes.
    ///
    /// # Errors
    ///
    /// [`crate::Error::RuntimeState`] when the planner refuses the invocation, and
    /// [`crate::Error::RuntimeStateSource`] when binding or admitting the launch fails.
    pub(crate) fn convert(
        &mut self,
        dst: DType,
        src: DType,
        len: i64,
        out: &t4a_cuda_runtime::Buffer,
        input: &t4a_cuda_runtime::Buffer,
    ) -> crate::Result<()> {
        let plan = plan_convert(dst, src, len).map_err(plan_error)?;
        self.run(
            Entry::Convert,
            plan,
            &[
                Arg::DevicePtr(out),
                Arg::DevicePtr(input),
                Arg::U32(dst.tag() as u32),
                Arg::U32(src.tag() as u32),
                Arg::U64(len as u64),
            ],
        )
    }

    /// `t4a_cuda_gather`: the XLA-style gather described by `plan`.
    ///
    /// # Errors
    ///
    /// [`crate::Error::RuntimeState`] when the planner refuses the invocation or the
    /// gather/scatter plan, and [`crate::Error::RuntimeStateSource`] when binding or
    /// admitting the launch fails.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gather(
        &mut self,
        dtype: DType,
        index_dtype: DType,
        out: &t4a_cuda_kernels::Layout,
        out_len: i64,
        operand: &t4a_cuda_kernels::Layout,
        operand_len: i64,
        indices: &t4a_cuda_kernels::Layout,
        indices_len: i64,
        plan: &GatherPlan,
        out_buffer: &t4a_cuda_runtime::Buffer,
        operand_buffer: &t4a_cuda_runtime::Buffer,
        indices_buffer: &t4a_cuda_runtime::Buffer,
    ) -> crate::Result<()> {
        let (launch, output_len) = plan_gather(
            dtype,
            index_dtype,
            out,
            out_len,
            operand,
            operand_len,
            indices,
            indices_len,
            plan,
        )
        .map_err(plan_error)?;
        let out_layout = t4a_cuda_kernels::CLayout::new(out);
        let operand_layout = t4a_cuda_kernels::CLayout::new(operand);
        let indices_layout = t4a_cuda_kernels::CLayout::new(indices);
        let mirror = t4a_cuda_kernels::CGatherPlan::new(plan, out.rank()).map_err(plan_error)?;
        self.run(
            Entry::Gather,
            launch,
            &[
                Arg::DevicePtr(out_buffer),
                Arg::DevicePtr(operand_buffer),
                Arg::DevicePtr(indices_buffer),
                Arg::U32(dtype.tag() as u32),
                Arg::U32(index_dtype.tag() as u32),
                layout_arg(&out_layout),
                layout_arg(&operand_layout),
                layout_arg(&indices_layout),
                pod_arg(&mirror),
                Arg::U64(output_len as u64),
            ],
        )
    }

    /// `t4a_cuda_scatter_add`: the additive scatter described by `plan`.
    ///
    /// # Errors
    ///
    /// [`crate::Error::RuntimeState`] when the planner refuses the invocation or the
    /// gather/scatter plan, and [`crate::Error::RuntimeStateSource`] when binding or
    /// admitting the launch fails.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn scatter_add(
        &mut self,
        dtype: DType,
        index_dtype: DType,
        out: &t4a_cuda_kernels::Layout,
        out_len: i64,
        indices: &t4a_cuda_kernels::Layout,
        indices_len: i64,
        updates: &t4a_cuda_kernels::Layout,
        updates_len: i64,
        plan: &ScatterPlan,
        out_buffer: &t4a_cuda_runtime::Buffer,
        indices_buffer: &t4a_cuda_runtime::Buffer,
        updates_buffer: &t4a_cuda_runtime::Buffer,
    ) -> crate::Result<()> {
        let launch = plan_scatter_add(
            dtype,
            index_dtype,
            out,
            out_len,
            indices,
            indices_len,
            updates,
            updates_len,
            plan,
        )
        .map_err(plan_error)?;
        let out_layout = t4a_cuda_kernels::CLayout::new(out);
        let indices_layout = t4a_cuda_kernels::CLayout::new(indices);
        let updates_layout = t4a_cuda_kernels::CLayout::new(updates);
        let mirror = t4a_cuda_kernels::CScatterPlan::new(out.rank(), updates.rank(), plan)
            .map_err(plan_error)?;
        let update_len = updates.element_count().map_err(plan_error)?;
        self.run(
            Entry::ScatterAdd,
            launch,
            &[
                Arg::DevicePtr(out_buffer),
                Arg::DevicePtr(indices_buffer),
                Arg::DevicePtr(updates_buffer),
                Arg::U32(dtype.tag() as u32),
                Arg::U32(index_dtype.tag() as u32),
                layout_arg(&out_layout),
                layout_arg(&indices_layout),
                layout_arg(&updates_layout),
                pod_arg(&mirror),
                Arg::U64(update_len as u64),
            ],
        )
    }

    /// `t4a_cuda_slice`: the window `starts`/`steps` describes, into a compact destination.
    ///
    /// # Errors
    ///
    /// [`crate::Error::RuntimeState`] when the planner refuses the window, and
    /// [`crate::Error::RuntimeStateSource`] when binding or admitting the launch fails.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn slice(
        &mut self,
        dtype: DType,
        out: &t4a_cuda_kernels::Layout,
        out_len: i64,
        operand: &t4a_cuda_kernels::Layout,
        operand_len: i64,
        starts: &[i64],
        steps: &[i64],
        out_buffer: &t4a_cuda_runtime::Buffer,
        operand_buffer: &t4a_cuda_runtime::Buffer,
    ) -> crate::Result<()> {
        let launch = plan_slice(dtype, out, out_len, operand, operand_len, starts, steps)
            .map_err(plan_error)?;
        let out_layout = t4a_cuda_kernels::CLayout::new(out);
        let operand_layout = t4a_cuda_kernels::CLayout::new(operand);
        let starts = t4a_cuda_kernels::CAxisValues::new(starts).map_err(plan_error)?;
        let steps = t4a_cuda_kernels::CAxisValues::new(steps).map_err(plan_error)?;
        self.run(
            Entry::Slice,
            launch,
            &[
                Arg::DevicePtr(out_buffer),
                Arg::DevicePtr(operand_buffer),
                Arg::U32(dtype.tag() as u32),
                layout_arg(&out_layout),
                layout_arg(&operand_layout),
                pod_arg(&starts),
                pod_arg(&steps),
                Arg::U64(out_len as u64),
            ],
        )
    }

    /// `t4a_cuda_dynamic_slice`: the window whose starts are device-resident.
    ///
    /// # Errors
    ///
    /// [`crate::Error::RuntimeState`] when the planner refuses the window, and
    /// [`crate::Error::RuntimeStateSource`] when binding or admitting the launch fails.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dynamic_slice(
        &mut self,
        dtype: DType,
        index_dtype: DType,
        out: &t4a_cuda_kernels::Layout,
        out_len: i64,
        operand: &t4a_cuda_kernels::Layout,
        operand_len: i64,
        starts_rank: u32,
        out_buffer: &t4a_cuda_runtime::Buffer,
        operand_buffer: &t4a_cuda_runtime::Buffer,
        starts_buffer: &t4a_cuda_runtime::Buffer,
    ) -> crate::Result<()> {
        let launch = plan_dynamic_slice(
            dtype,
            index_dtype,
            out,
            out_len,
            operand,
            operand_len,
            starts_rank,
        )
        .map_err(plan_error)?;
        let out_layout = t4a_cuda_kernels::CLayout::new(out);
        let operand_layout = t4a_cuda_kernels::CLayout::new(operand);
        self.run(
            Entry::DynamicSlice,
            launch,
            &[
                Arg::DevicePtr(out_buffer),
                Arg::DevicePtr(operand_buffer),
                Arg::DevicePtr(starts_buffer),
                Arg::U32(dtype.tag() as u32),
                Arg::U32(index_dtype.tag() as u32),
                layout_arg(&out_layout),
                layout_arg(&operand_layout),
                Arg::U64(out_len as u64),
            ],
        )
    }

    /// `t4a_cuda_pad`: edge and interior padding into a compact destination.
    ///
    /// # Errors
    ///
    /// [`crate::Error::RuntimeState`] when the planner refuses the padding, and
    /// [`crate::Error::RuntimeStateSource`] when binding or admitting the launch fails.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn pad(
        &mut self,
        dtype: DType,
        out: &t4a_cuda_kernels::Layout,
        out_len: i64,
        operand: &t4a_cuda_kernels::Layout,
        operand_len: i64,
        edge_low: &[i64],
        edge_high: &[i64],
        interior: &[i64],
        out_buffer: &t4a_cuda_runtime::Buffer,
        operand_buffer: &t4a_cuda_runtime::Buffer,
    ) -> crate::Result<()> {
        let launch = plan_pad(
            dtype,
            out,
            out_len,
            operand,
            operand_len,
            edge_low,
            edge_high,
            interior,
        )
        .map_err(plan_error)?;
        let out_layout = t4a_cuda_kernels::CLayout::new(out);
        let operand_layout = t4a_cuda_kernels::CLayout::new(operand);
        let edge_low = t4a_cuda_kernels::CAxisValues::new(edge_low).map_err(plan_error)?;
        let edge_high = t4a_cuda_kernels::CAxisValues::new(edge_high).map_err(plan_error)?;
        let interior = t4a_cuda_kernels::CAxisValues::new(interior).map_err(plan_error)?;
        self.run(
            Entry::Pad,
            launch,
            &[
                Arg::DevicePtr(out_buffer),
                Arg::DevicePtr(operand_buffer),
                Arg::U32(dtype.tag() as u32),
                layout_arg(&out_layout),
                layout_arg(&operand_layout),
                pod_arg(&edge_low),
                pod_arg(&edge_high),
                pod_arg(&interior),
                Arg::U64(out_len as u64),
            ],
        )
    }

    /// `t4a_cuda_validate_float_indices`: record the lowest index that is not an exact
    /// integer.
    ///
    /// # Errors
    ///
    /// [`crate::Error::RuntimeState`] when the planner refuses the invocation, and
    /// [`crate::Error::RuntimeStateSource`] when binding or admitting the launch fails.
    pub(crate) fn validate_float_indices(
        &mut self,
        dtype: DType,
        len: i64,
        max_exact_integer: f64,
        indices: &t4a_cuda_runtime::Buffer,
        flag: &t4a_cuda_runtime::Buffer,
    ) -> crate::Result<()> {
        let plan = plan_validate_float_indices(dtype, len).map_err(plan_error)?;
        self.run(
            Entry::ValidateFloatIndices,
            plan,
            &[
                Arg::DevicePtr(indices),
                Arg::DevicePtr(flag),
                Arg::U32(dtype.tag() as u32),
                Arg::F64(max_exact_integer),
                Arg::U64(len as u64),
            ],
        )
    }

    /// `t4a_cuda_validate_real_cast`: record the first sampled value a cast to a signed
    /// integer cannot represent.
    ///
    /// # Errors
    ///
    /// [`crate::Error::RuntimeState`] when the planner refuses the invocation, and
    /// [`crate::Error::RuntimeStateSource`] when binding or admitting the launch fails.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn validate_real_cast(
        &mut self,
        dtype: DType,
        len: i64,
        stride: i64,
        min: f64,
        max: f64,
        max_inclusive: bool,
        input: &t4a_cuda_runtime::Buffer,
        flag: &t4a_cuda_runtime::Buffer,
    ) -> crate::Result<()> {
        let plan = plan_validate_real_cast(dtype, len, stride).map_err(plan_error)?;
        self.run(
            Entry::ValidateRealCast,
            plan,
            &[
                Arg::DevicePtr(input),
                Arg::DevicePtr(flag),
                Arg::U32(dtype.tag() as u32),
                Arg::F64(min),
                Arg::F64(max),
                Arg::U64(stride as u64),
                Arg::I32(i32::from(max_inclusive)),
                Arg::U64(len as u64),
            ],
        )
    }

    /// `t4a_cuda_combined_strided`: one logical domain re-addressed through each operand layout.
    ///
    /// `AXPBY` computes `out = coefficients[0] * a + coefficients[1] * b`, and the ABI allows it in
    /// place on `b` when the two layouts are identical.
    ///
    /// # Errors
    ///
    /// [`crate::Error::RuntimeState`] when the planner refuses the layouts, and
    /// [`crate::Error::RuntimeStateSource`] when binding or admitting the launch fails.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn combined_strided(
        &mut self,
        op: t4a_cuda_kernels::StridedOp,
        dtype: DType,
        out: &t4a_cuda_kernels::Layout,
        out_len: i64,
        a: &t4a_cuda_kernels::Layout,
        a_len: i64,
        b: &t4a_cuda_kernels::Layout,
        b_len: i64,
        out_buffer: &t4a_cuda_runtime::Buffer,
        a_buffer: &t4a_cuda_runtime::Buffer,
        b_buffer: &t4a_cuda_runtime::Buffer,
        coefficients: &t4a_cuda_runtime::Buffer,
    ) -> crate::Result<()> {
        let launch =
            t4a_cuda_kernels::plan_combined_strided(op, dtype, out, out_len, a, a_len, b, b_len)
                .map_err(plan_error)?;
        let out_layout = t4a_cuda_kernels::CLayout::new(out);
        let a_layout = t4a_cuda_kernels::CLayout::new(a);
        let b_layout = t4a_cuda_kernels::CLayout::new(b);
        let out_count = out.element_count().map_err(plan_error)?;
        self.run(
            Entry::CombinedStrided,
            launch,
            &[
                Arg::DevicePtr(out_buffer),
                Arg::DevicePtr(a_buffer),
                Arg::DevicePtr(b_buffer),
                Arg::DevicePtr(coefficients),
                layout_arg(&out_layout),
                layout_arg(&a_layout),
                layout_arg(&b_layout),
                Arg::U32(dtype.tag() as u32),
                Arg::I32(op.tag()),
                Arg::U64(out_count as u64),
            ],
        )
    }

    /// `t4a_cuda_concatenate_copy`: one compact input into `out` at `axis_offset`.
    ///
    /// # Errors
    ///
    /// [`crate::Error::RuntimeState`] when the planner refuses the placement, and
    /// [`crate::Error::RuntimeStateSource`] when binding or admitting the launch fails.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn concatenate_copy(
        &mut self,
        dtype: DType,
        out: &t4a_cuda_kernels::Layout,
        out_len: i64,
        input: &t4a_cuda_kernels::Layout,
        input_len: i64,
        axis: u32,
        axis_offset: i64,
        out_buffer: &t4a_cuda_runtime::Buffer,
        input_buffer: &t4a_cuda_runtime::Buffer,
    ) -> crate::Result<()> {
        let launch =
            plan_concatenate_copy(dtype, out, out_len, input, input_len, axis, axis_offset)
                .map_err(plan_error)?;
        let out_layout = t4a_cuda_kernels::CLayout::new(out);
        let input_layout = t4a_cuda_kernels::CLayout::new(input);
        let input_len = input.element_count().map_err(plan_error)?;
        self.run(
            Entry::ConcatenateCopy,
            launch,
            &[
                Arg::DevicePtr(out_buffer),
                Arg::DevicePtr(input_buffer),
                Arg::U32(dtype.tag() as u32),
                layout_arg(&out_layout),
                layout_arg(&input_layout),
                Arg::U32(axis),
                Arg::U64(axis_offset as u64),
                Arg::U64(input_len as u64),
            ],
        )
    }
}

/// A planner refusal as the seam's typed cause.
fn plan_error(error: t4a_cuda_kernels::Error) -> crate::Error {
    match error {
        // A planner refusal about the operation or dtype is the backend not serving the
        // request, not a runtime failure; keep the two classes apart for the caller.
        t4a_cuda_kernels::Error::UnsupportedDType | t4a_cuda_kernels::Error::UnsupportedOp => {
            crate::Error::unsupported("cuda plan", error.to_string())
        }
        error => crate::Error::runtime_state("cuda plan", error.to_string()),
    }
}

/// A by-value parameter view of a plain-data ABI struct, borrowing it for the launch.
///
/// # Safety contract
///
/// `T` must be `repr(C)` plain data with no padding and entirely initialised; the four
/// ABI structs this module passes (`CLayout` is handled by
/// [`layout_arg`](super::layout_arg)) are asserted by the kernel crate.
pub(crate) fn pod_arg<T>(value: &T) -> Arg<'_> {
    // SAFETY: the callers pass the kernel crate's `repr(C)` parameter structs, which the
    // crate asserts are sized without padding; the slice borrows `value` for the launch.
    let bytes = unsafe {
        std::slice::from_raw_parts((value as *const T).cast::<u8>(), std::mem::size_of::<T>())
    };
    Arg::Bytes {
        bytes,
        align: std::mem::align_of::<T>(),
    }
}

impl Dispatcher {
    /// `t4a_cuda_reduce`: one reduction pass with the crate's strategy.
    ///
    /// # Errors
    ///
    /// [`crate::Error::RuntimeState`] when the planner refuses the invocation, and
    /// [`crate::Error::RuntimeStateSource`] when binding or admitting the launch fails.
    #[allow(clippy::too_many_arguments)]
    // Only the dispatch tests exercise the explicit-strategy form; the engine uses `reduce_auto`.
    #[allow(dead_code)]
    pub(crate) fn reduce(
        &mut self,
        dtype: DType,
        op: ReduceOp,
        strategy: t4a_cuda_kernels::ReduceStrategy,
        out: &Layout,
        out_len: i64,
        input: &Layout,
        in_len: i64,
        axis: u32,
        plane_width: u32,
        out_buffer: &t4a_cuda_runtime::Buffer,
        in_buffer: &t4a_cuda_runtime::Buffer,
    ) -> crate::Result<()> {
        let plan = plan_reduce(
            dtype,
            op,
            strategy,
            out,
            out_len,
            input,
            in_len,
            axis,
            plane_width,
        )
        .map_err(plan_error)?;
        let out_layout = t4a_cuda_kernels::CLayout::new(out);
        let in_layout = t4a_cuda_kernels::CLayout::new(input);
        let output_len = out.element_count().map_err(plan_error)?;
        self.run(
            Entry::Reduce,
            plan,
            &[
                Arg::DevicePtr(out_buffer),
                Arg::DevicePtr(in_buffer),
                Arg::U32(dtype.tag() as u32),
                Arg::I32(op.tag()),
                Arg::I32(strategy.tag()),
                layout_arg(&out_layout),
                layout_arg(&in_layout),
                Arg::I32(axis as i32),
                Arg::U64(output_len as u64),
            ],
        )
    }

    /// Plan a whole reduction with the crate's split policy and run both passes.
    ///
    /// `partials` must hold `splits * out_len` elements, and is only used when the
    /// policy splits; the returned [`ReductionPlan`] says which happened.
    ///
    /// # Errors
    ///
    /// [`crate::Error::RuntimeState`] when the planner refuses the invocation, and
    /// [`crate::Error::RuntimeStateSource`] when binding or admitting a launch fails.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn reduce_auto(
        &mut self,
        dtype: DType,
        op: ReduceOp,
        out: &Layout,
        out_len: i64,
        input: &Layout,
        in_len: i64,
        axis: u32,
        plane_width: u32,
        out_buffer: &t4a_cuda_runtime::Buffer,
        in_buffer: &t4a_cuda_runtime::Buffer,
        partials_buffer: &t4a_cuda_runtime::Buffer,
    ) -> crate::Result<ReductionPlan> {
        let plan = plan_reduction(dtype, op, out, out_len, input, in_len, axis, plane_width)
            .map_err(plan_error)?;
        match &plan {
            ReductionPlan::Direct { strategy, launch } => {
                let out_layout = t4a_cuda_kernels::CLayout::new(out);
                let in_layout = t4a_cuda_kernels::CLayout::new(input);
                let output_len = out.element_count().map_err(plan_error)?;
                self.run(
                    Entry::Reduce,
                    *launch,
                    &[
                        Arg::DevicePtr(out_buffer),
                        Arg::DevicePtr(in_buffer),
                        Arg::U32(dtype.tag() as u32),
                        Arg::I32(op.tag()),
                        Arg::I32(strategy.tag()),
                        layout_arg(&out_layout),
                        layout_arg(&in_layout),
                        Arg::I32(axis as i32),
                        Arg::U64(output_len as u64),
                    ],
                )?;
            }
            ReductionPlan::Split {
                split,
                pass1,
                combine_op,
                combine_strategy,
                pass2,
            } => {
                let out_layout = t4a_cuda_kernels::CLayout::new(out);
                let in_layout = t4a_cuda_kernels::CLayout::new(input);
                let partials =
                    t4a_cuda_kernels::reduction_partials_layout(input, axis, split.splits)
                        .map_err(plan_error)?;
                let partials_len = partials.element_count().map_err(plan_error)?;
                let partials_layout =
                    t4a_cuda_kernels::CLayout::from_wide(&partials).map_err(plan_error)?;
                let output_len = out.element_count().map_err(plan_error)?;
                self.run(
                    Entry::ReducePartials,
                    *pass1,
                    &[
                        Arg::DevicePtr(partials_buffer),
                        Arg::DevicePtr(in_buffer),
                        Arg::U32(dtype.tag() as u32),
                        Arg::I32(op.tag()),
                        layout_arg(&partials_layout),
                        layout_arg(&in_layout),
                        Arg::I32(axis as i32),
                        Arg::U64(output_len as u64),
                        // The kernel's `long long` parameters carry the same bits.
                        Arg::U64(split.splits as u64),
                        Arg::U64(split.chunk_len as u64),
                    ],
                )?;
                self.run(
                    Entry::Reduce,
                    *pass2,
                    &[
                        Arg::DevicePtr(out_buffer),
                        Arg::DevicePtr(partials_buffer),
                        Arg::U32(dtype.tag() as u32),
                        Arg::I32(combine_op.tag()),
                        Arg::I32(combine_strategy.tag()),
                        layout_arg(&out_layout),
                        layout_arg(&partials_layout),
                        Arg::I32(axis as i32),
                        Arg::U64(output_len as u64),
                    ],
                )?;
                let _ = partials_len;
            }
        }
        Ok(plan)
    }
}

#[cfg(test)]
mod reduce_tests;
#[cfg(test)]
mod tests;
