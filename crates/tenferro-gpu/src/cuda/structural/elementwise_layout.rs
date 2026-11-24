//! Elementwise operands read where they lie (#2701).
//!
//! The flat elementwise entries address `len` consecutive elements from an allocation's base, so
//! every operand that is not one compact span at its base used to be staged into a compact copy
//! by `device_read` first: a permuted, offset, reversed or broadcast view, and also an eager
//! value, which reads as a view of its retained allocation. That copy ran inside the operation.
//!
//! Each operand is now resolved once into an [`ElementwiseOperand`]:
//!
//! - a **flat** region (an owned tensor, or a view that is one compact span at its allocation's
//!   base) goes to the flat entry at its own allocation;
//! - any other region of rank at most [`t4a_cuda_kernels::MAX_RANK`] goes to the layout-taking
//!   entry (`t4a_cuda_strided_{unary,binary,compare,select}`), which reads it through its own
//!   extents, strides and offset, a zero stride included;
//! - only a wider view, which the by-value layout cannot carry, is still staged once.
//!
//! The layout-taking entries compute the flat entries' per-element formulas with the same device
//! helpers, so the two routes agree bit for bit on every non-NaN result (the sign and payload of a
//! NaN result are unspecified, as in the catalogue's ABI); the staging counter
//! ([`CudaEngine::staged_reads`]) stays put on both. Reductions read only flat views in place.

use t4a_cuda_kernels::{Entry, Layout};
use tenferro_tensor::{DType, TensorRead};

use super::{
    allocation_len, catalogue_extents, catalogue_strides, compact_strides, kernel_dtype,
    region_is_flat, storage_of, CudaEngine, Region, Resolved,
};

/// One elementwise operand, resolved for its launch.
pub(super) enum ElementwiseOperand<'v> {
    /// One compact span at its allocation's base: the flat entries read it in place.
    Flat(Region<'v>),
    /// A strided, offset, reversed or broadcast region the layout-taking entries read in place.
    Strided(Region<'v>),
    /// A view wider than the by-value layout, staged once into a compact copy.
    Staged(Resolved<'v>),
}

impl ElementwiseOperand<'_> {
    pub(super) fn dtype(&self) -> DType {
        match self {
            Self::Flat(region) | Self::Strided(region) => region.dtype,
            Self::Staged(resolved) => resolved.dtype(),
        }
    }

    pub(super) fn shape(&self) -> &[usize] {
        match self {
            Self::Flat(region) | Self::Strided(region) => &region.shape,
            Self::Staged(resolved) => resolved.shape(),
        }
    }

    pub(super) fn is_strided(&self) -> bool {
        matches!(self, Self::Strided(_))
    }
}

/// Whether any operand needs the layout-taking entry.
pub(super) fn any_strided(operands: &[&ElementwiseOperand<'_>]) -> bool {
    operands.iter().any(|operand| operand.is_strided())
}

/// The compact, offset-zero layout of `shape`.
///
/// A zero extent multiplies into the following strides as one, so an empty destination keeps
/// nonzero strides: the planners refuse a zero stride on a destination as a broadcast, and an
/// empty launch is skipped anyway.
pub(super) fn compact_destination_layout(
    shape: &[usize],
    op: &'static str,
) -> crate::Result<Layout> {
    let extents = catalogue_extents(shape, op)?;
    let padded: Vec<usize> = shape.iter().map(|&extent| extent.max(1)).collect();
    let strides = catalogue_strides(&compact_strides(&padded, op)?, op)?;
    Layout::new(&extents, &strides, 0)
        .map_err(|error| crate::Error::runtime_state_source(op, error))
}

impl CudaEngine {
    /// Resolve one elementwise operand: in place when a flat or layout-taking entry can read it,
    /// staged only when its rank exceeds the by-value layout.
    ///
    /// The native read boundary's refusals (a broken engine, a host operand, another runtime's
    /// allocation) are applied first, by [`Self::read_region`].
    pub(super) fn elementwise_operand<'v>(
        &mut self,
        input: &'v TensorRead<'v>,
        op: &'static str,
    ) -> crate::Result<ElementwiseOperand<'v>> {
        let region = self.read_region(input, op)?;
        // A region with no element is read by no launch; the flat route serves it (it skips the
        // zero-length launch), so it never needs a layout the planners would refuse.
        if region_is_flat(&region) || region.shape.contains(&0) {
            return Ok(ElementwiseOperand::Flat(region));
        }
        if region.extents.len() <= t4a_cuda_kernels::MAX_RANK {
            return Ok(ElementwiseOperand::Strided(region));
        }
        self.device_read(input.clone(), op)
            .map(ElementwiseOperand::Staged)
    }

    /// The device allocation an operand's launch pointer addresses.
    pub(super) fn elementwise_operand_buffer<'o>(
        &self,
        operand: &'o ElementwiseOperand<'_>,
        op: &'static str,
    ) -> crate::Result<&'o t4a_cuda_runtime::Buffer> {
        match operand {
            ElementwiseOperand::Flat(region) | ElementwiseOperand::Strided(region) => {
                Ok(region.storage.allocation())
            }
            ElementwiseOperand::Staged(resolved) => {
                with_scalar!(resolved.dtype(), |T| self.allocation_of::<T>(resolved, op))
            }
        }
    }

    /// An operand's layout over `shape` and its allocation's element capacity.
    ///
    /// `shape` is the launch domain. An operand of that shape keeps its own layout; a rank-0
    /// operand (the scalar side of a power) is read through stride zero on every axis, so the same
    /// element repeats over the domain.
    pub(super) fn elementwise_operand_layout(
        &self,
        operand: &ElementwiseOperand<'_>,
        shape: &[usize],
        op: &'static str,
    ) -> crate::Result<(Layout, i64)> {
        let (layout, len, offset) = match operand {
            ElementwiseOperand::Flat(region) | ElementwiseOperand::Strided(region) => {
                (region.layout(op)?, region.len(op)?, region.offset)
            }
            ElementwiseOperand::Staged(resolved) => {
                let storage = with_scalar!(resolved.dtype(), |T| storage_of::<T>(resolved, op))?;
                (
                    compact_destination_layout(resolved.shape(), op)?,
                    allocation_len(storage, resolved.dtype(), op)?,
                    0,
                )
            }
        };
        if operand.shape() == shape {
            return Ok((layout, len));
        }
        if operand.shape().is_empty() {
            let extents = catalogue_extents(shape, op)?;
            let zero = vec![0i64; extents.len()];
            let broadcast = Layout::new(&extents, &zero, offset)
                .map_err(|error| crate::Error::runtime_state_source(op, error))?;
            return Ok((broadcast, len));
        }
        Err(crate::Error::shape_mismatch(
            op,
            operand.shape().to_vec(),
            shape.to_vec(),
        ))
    }

    /// `out = op(input)` through the layout-taking unary entry, into the destination `out`
    /// addressed by `out_layout` (a fresh compact output, or a caller's writable region).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn strided_unary_launch(
        &mut self,
        op: t4a_cuda_kernels::UnaryOp,
        out_dtype: DType,
        out: super::super::dispatch::Dest<'_>,
        out_layout: &Layout,
        out_len: i64,
        input: &ElementwiseOperand<'_>,
        shape: &[usize],
        name: &'static str,
    ) -> crate::Result<()> {
        let (in_layout, in_len) = self.elementwise_operand_layout(input, shape, name)?;
        let in_buffer = self.elementwise_operand_buffer(input, name)?;
        let dst = kernel_dtype(out_dtype, name)?;
        let src = kernel_dtype(input.dtype(), name)?;
        let mut dispatcher = self.dispatcher(&[Entry::StridedUnary])?;
        dispatcher.strided_unary(
            op, dst, src, out_layout, out_len, &in_layout, in_len, out, in_buffer,
        )
    }

    /// `out = lhs op rhs` through the layout-taking binary entry.
    ///
    /// `status` is the zeroed word a checked integer operation reports through.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn strided_binary_launch(
        &mut self,
        op: t4a_cuda_kernels::StridedBinaryOp,
        dtype: DType,
        out: super::super::dispatch::Dest<'_>,
        out_layout: &Layout,
        out_len: i64,
        operands: [&ElementwiseOperand<'_>; 2],
        shape: &[usize],
        status: Option<&t4a_cuda_runtime::Buffer>,
        name: &'static str,
    ) -> crate::Result<()> {
        let (lhs_layout, lhs_len) = self.elementwise_operand_layout(operands[0], shape, name)?;
        let (rhs_layout, rhs_len) = self.elementwise_operand_layout(operands[1], shape, name)?;
        let lhs = self.elementwise_operand_buffer(operands[0], name)?;
        let rhs = self.elementwise_operand_buffer(operands[1], name)?;
        let kernel = kernel_dtype(dtype, name)?;
        let mut dispatcher = self.dispatcher(&[Entry::StridedBinary])?;
        dispatcher.strided_binary(
            op,
            kernel,
            out_layout,
            out_len,
            &lhs_layout,
            lhs_len,
            &rhs_layout,
            rhs_len,
            out,
            lhs,
            rhs,
            status,
        )
    }

    /// One Bool byte per element through the layout-taking comparison entry.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn strided_compare_launch(
        &mut self,
        op: t4a_cuda_kernels::CompareOp,
        dtype: DType,
        out: super::super::dispatch::Dest<'_>,
        out_layout: &Layout,
        out_len: i64,
        operands: [&ElementwiseOperand<'_>; 2],
        shape: &[usize],
        name: &'static str,
    ) -> crate::Result<()> {
        let (lhs_layout, lhs_len) = self.elementwise_operand_layout(operands[0], shape, name)?;
        let (rhs_layout, rhs_len) = self.elementwise_operand_layout(operands[1], shape, name)?;
        let lhs = self.elementwise_operand_buffer(operands[0], name)?;
        let rhs = self.elementwise_operand_buffer(operands[1], name)?;
        let kernel = kernel_dtype(dtype, name)?;
        let mut dispatcher = self.dispatcher(&[Entry::StridedCompare])?;
        dispatcher.strided_compare(
            op,
            kernel,
            out_layout,
            out_len,
            &lhs_layout,
            lhs_len,
            &rhs_layout,
            rhs_len,
            out,
            lhs,
            rhs,
        )
    }

    /// `out = predicate ? on_true : on_false` through the layout-taking selection entry.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn strided_select_launch(
        &mut self,
        dtype: DType,
        out: super::super::dispatch::Dest<'_>,
        out_layout: &Layout,
        out_len: i64,
        operands: [&ElementwiseOperand<'_>; 3],
        shape: &[usize],
        name: &'static str,
    ) -> crate::Result<()> {
        let (predicate_layout, predicate_len) =
            self.elementwise_operand_layout(operands[0], shape, name)?;
        let (true_layout, true_len) = self.elementwise_operand_layout(operands[1], shape, name)?;
        let (false_layout, false_len) =
            self.elementwise_operand_layout(operands[2], shape, name)?;
        let predicate = self.elementwise_operand_buffer(operands[0], name)?;
        let on_true = self.elementwise_operand_buffer(operands[1], name)?;
        let on_false = self.elementwise_operand_buffer(operands[2], name)?;
        let kernel = kernel_dtype(dtype, name)?;
        let mut dispatcher = self.dispatcher(&[Entry::StridedSelect])?;
        dispatcher.strided_select(
            kernel,
            out_layout,
            out_len,
            (&predicate_layout, predicate_len, predicate),
            (&true_layout, true_len, on_true),
            (&false_layout, false_len, on_false),
            out,
        )
    }
}

impl CudaEngine {
    /// Stage every strided operand of a launch whose domain rank exceeds the by-value layout.
    ///
    /// An operand is classified by its own rank, but a rank-0 scalar side (a power's exponent) is
    /// read over the result's domain, which may be wider than [`t4a_cuda_kernels::MAX_RANK`]; such
    /// a launch keeps the flat route, so a strided scalar is staged into a compact copy.
    pub(super) fn stage_for_domain<'v>(
        &mut self,
        operand: ElementwiseOperand<'v>,
        input: &'v TensorRead<'v>,
        domain_rank: usize,
        op: &'static str,
    ) -> crate::Result<ElementwiseOperand<'v>> {
        if operand.is_strided() && domain_rank > t4a_cuda_kernels::MAX_RANK {
            return self
                .device_read(input.clone(), op)
                .map(ElementwiseOperand::Staged);
        }
        Ok(operand)
    }
}
