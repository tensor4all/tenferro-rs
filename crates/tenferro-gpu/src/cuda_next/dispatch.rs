//! Per-family dispatch: plan through the kernel crate, marshal the ABI arguments, launch.
//!
//! Every plan comes from the kernel crate's `plan_*` function — this module adds no
//! validation of its own — and every argument order follows the documented ABI.

use t4a_cuda_kernels::{
    plan_binary, plan_clamp, plan_compare, plan_convert, plan_gather, plan_scatter_add,
    plan_select, plan_unary, BinaryOp, DType, Entry, GatherPlan, ScalarSide, ScatterPlan, UnaryOp,
};
use t4a_cuda_runtime::{Arg, LaunchConfig};

use super::launch::{layout_arg, KernelModules};

/// One runtime's dispatcher over a prepared module set.
#[doc(hidden)]
pub struct Dispatcher {
    modules: KernelModules,
}

impl Dispatcher {
    /// Prepare exactly the modules `entries` need.
    ///
    /// # Errors
    ///
    /// [`crate::Error::RuntimeStateSource`] when NVRTC rejects a module or the runtime
    /// cannot prepare its artifact.
    pub fn for_entries(
        runtime: &super::runtime::CudaRuntime,
        entries: &[Entry],
    ) -> crate::Result<Self> {
        Ok(Dispatcher {
            modules: KernelModules::for_entries(runtime, entries)?,
        })
    }

    /// The prepared module set.
    pub fn modules(&self) -> &KernelModules {
        &self.modules
    }

    /// The prepared module set, mutably, for a launch this module does not wrap.
    pub fn modules_mut(&mut self) -> &mut KernelModules {
        &mut self.modules
    }

    /// Enqueue one launch with the plan's own geometry.
    fn run(
        &mut self,
        entry: Entry,
        plan: t4a_cuda_kernels::Launch,
        args: &[Arg<'_>],
    ) -> crate::Result<()> {
        // SAFETY: every caller below builds the argument list in the documented ABI order
        // after the kernel crate's planner accepted the invocation.
        unsafe {
            self.modules
                .launch(entry, LaunchConfig::new(plan.grid[0], plan.block[0]), args)
        }
    }

    /// `t4a_cuda_unary`: one unary operation over `len` elements.
    ///
    /// # Errors
    ///
    /// [`crate::Error::RuntimeState`] when the kernel crate's planner refuses the
    /// invocation, and [`crate::Error::RuntimeStateSource`] when binding or admitting the
    /// launch fails.
    pub fn unary(
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
    pub fn binary(
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
    pub fn compare(
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
    pub fn select(
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
    pub fn clamp(
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
    pub fn convert(
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
    pub fn gather(
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
    pub fn scatter_add(
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
}

/// A planner refusal as the seam's typed cause.
fn plan_error(error: t4a_cuda_kernels::Error) -> crate::Error {
    crate::Error::runtime_state("cuda plan", error.to_string())
}

/// A by-value parameter view of a plain-data ABI struct, borrowing it for the launch.
///
/// # Safety contract
///
/// `T` must be `repr(C)` plain data with no padding and entirely initialised; the four
/// ABI structs this module passes (`CLayout` is handled by
/// [`layout_arg`](super::layout_arg)) are asserted by the kernel crate.
fn pod_arg<T>(value: &T) -> Arg<'_> {
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
