use std::collections::HashSet;
use std::marker::PhantomData;

use tenferro_algebra::{Scalar, Standard};
use tenferro_device::{Error, Result};

use crate::{
    mode_position, validate_rank, validate_shape_count, validate_shape_eq, SemiringBinaryOp,
    SemiringCoreDescriptor, SemiringFastPathDescriptor, TensorSemiringCore, TensorSemiringFastPath,
};

use super::context::{CpuBackend, CpuContext};
use super::execution::execute_semiring_plan;
use super::plan::{build_contract_gemm_spec, compute_paired_components, CpuPlan};

impl CpuBackend {
    pub(super) fn build_semiring_core_plan<T: Scalar>(
        desc: &SemiringCoreDescriptor,
        shapes: &[&[usize]],
    ) -> Result<CpuPlan<T>> {
        match desc {
            SemiringCoreDescriptor::BatchedGemm {
                batch_dims,
                m,
                n,
                k,
            } => {
                validate_shape_count(shapes, 3, "BatchedGemm")?;
                if !Self::supports_batched_gemm_type::<T>() {
                    return Err(Error::InvalidArgument(format!(
                        "BatchedGemm supports only f32, f64, Complex32, and Complex64 (got {})",
                        std::any::type_name::<T>()
                    )));
                }
                let expected_a: Vec<usize> = [*m, *k]
                    .iter()
                    .copied()
                    .chain(batch_dims.iter().copied())
                    .collect();
                let expected_b: Vec<usize> = [*k, *n]
                    .iter()
                    .copied()
                    .chain(batch_dims.iter().copied())
                    .collect();
                let expected_c: Vec<usize> = [*m, *n]
                    .iter()
                    .copied()
                    .chain(batch_dims.iter().copied())
                    .collect();
                validate_shape_eq(shapes[0], &expected_a, "BatchedGemm input A")?;
                validate_shape_eq(shapes[1], &expected_b, "BatchedGemm input B")?;
                validate_shape_eq(shapes[2], &expected_c, "BatchedGemm output C")?;
                Ok(CpuPlan::BatchedGemm {
                    batch_dims: batch_dims.clone(),
                    m: *m,
                    n: *n,
                    k: *k,
                    _marker: PhantomData,
                })
            }
            SemiringCoreDescriptor::ReduceAdd { modes_a, modes_c } => {
                validate_shape_count(shapes, 2, "ReduceAdd")?;
                validate_rank(shapes[0], modes_a.len(), "ReduceAdd input A")?;
                validate_rank(shapes[1], modes_c.len(), "ReduceAdd output C")?;
                let reduced_axes: Vec<usize> = modes_a
                    .iter()
                    .enumerate()
                    .filter(|(_, mode)| !modes_c.contains(mode))
                    .map(|(idx, _)| idx)
                    .collect();
                for window in reduced_axes.windows(2) {
                    if window[0] >= window[1] {
                        return Err(Error::InvalidArgument(format!(
                            "ReduceAdd: reduced_axes must be sorted and unique, got {reduced_axes:?}"
                        )));
                    }
                }
                if let Some(&last) = reduced_axes.last() {
                    if last >= modes_a.len() {
                        return Err(Error::InvalidArgument(format!(
                            "ReduceAdd: reduced axis {last} out of range for rank {}",
                            modes_a.len()
                        )));
                    }
                }
                Ok(CpuPlan::ReduceAdd {
                    reduced_axes,
                    _marker: PhantomData,
                })
            }
            SemiringCoreDescriptor::Trace {
                modes_a,
                modes_c,
                paired,
            } => {
                validate_shape_count(shapes, 2, "Trace")?;
                validate_rank(shapes[0], modes_a.len(), "Trace input A")?;
                validate_rank(shapes[1], modes_c.len(), "Trace output C")?;
                let paired_axes: Vec<(usize, usize)> = paired
                    .iter()
                    .map(|(m1, m2)| {
                        Ok((mode_position(modes_a, *m1)?, mode_position(modes_a, *m2)?))
                    })
                    .collect::<Result<_>>()?;
                for &(ax1, ax2) in &paired_axes {
                    if shapes[0][ax1] != shapes[0][ax2] {
                        return Err(Error::InvalidArgument(format!(
                            "Trace paired axes ({ax1}, {ax2}) have mismatched dimensions: {} vs {}",
                            shapes[0][ax1], shapes[0][ax2]
                        )));
                    }
                }
                let free_axes: Vec<usize> = modes_c
                    .iter()
                    .map(|mode| mode_position(modes_a, *mode))
                    .collect::<Result<_>>()?;
                let (components, comp_dims) = compute_paired_components(&paired_axes, shapes[0]);
                Ok(CpuPlan::Trace {
                    free_axes,
                    components,
                    comp_dims,
                    _marker: PhantomData,
                })
            }
            SemiringCoreDescriptor::AntiTrace {
                modes_a,
                modes_c,
                paired,
            } => {
                validate_shape_count(shapes, 2, "AntiTrace")?;
                validate_rank(shapes[0], modes_a.len(), "AntiTrace input A")?;
                validate_rank(shapes[1], modes_c.len(), "AntiTrace output C")?;
                let paired_axes: Vec<(usize, usize)> = paired
                    .iter()
                    .map(|(m1, m2)| {
                        Ok((mode_position(modes_c, *m1)?, mode_position(modes_c, *m2)?))
                    })
                    .collect::<Result<_>>()?;
                for &(ax1, ax2) in &paired_axes {
                    if shapes[1][ax1] != shapes[1][ax2] {
                        return Err(Error::InvalidArgument(format!(
                            "AntiTrace paired axes ({ax1}, {ax2}) have mismatched dimensions: {} vs {}",
                            shapes[1][ax1], shapes[1][ax2]
                        )));
                    }
                }
                let free_axes: Vec<usize> = modes_a
                    .iter()
                    .map(|mode| mode_position(modes_c, *mode))
                    .collect::<Result<_>>()?;
                let (components, comp_dims) = compute_paired_components(&paired_axes, shapes[1]);
                Ok(CpuPlan::AntiTrace {
                    paired_axes,
                    free_axes,
                    components,
                    comp_dims,
                    _marker: PhantomData,
                })
            }
            SemiringCoreDescriptor::AntiDiag {
                modes_a,
                modes_c,
                paired,
            } => {
                validate_shape_count(shapes, 2, "AntiDiag")?;
                validate_rank(shapes[0], modes_a.len(), "AntiDiag input A")?;
                validate_rank(shapes[1], modes_c.len(), "AntiDiag output C")?;
                let paired_axes: Vec<(usize, usize)> = paired
                    .iter()
                    .map(|(m1, m2)| {
                        Ok((mode_position(modes_c, *m1)?, mode_position(modes_c, *m2)?))
                    })
                    .collect::<Result<_>>()?;
                let free_axes: Vec<usize> = modes_a
                    .iter()
                    .map(|mode| mode_position(modes_c, *mode))
                    .collect::<Result<_>>()?;
                let (components, comp_dims) = compute_paired_components(&paired_axes, shapes[1]);
                let free_ax_set: HashSet<usize> = free_axes.iter().copied().collect();
                let generative_comps: Vec<usize> = components
                    .iter()
                    .enumerate()
                    .filter(|(_, comp)| comp.iter().all(|ax| !free_ax_set.contains(ax)))
                    .map(|(idx, _)| idx)
                    .collect();
                Ok(CpuPlan::AntiDiag {
                    paired_axes,
                    free_axes,
                    components,
                    comp_dims,
                    generative_comps,
                    _marker: PhantomData,
                })
            }
            SemiringCoreDescriptor::MakeContiguous => {
                validate_shape_count(shapes, 2, "MakeContiguous")?;
                validate_shape_eq(shapes[1], shapes[0], "MakeContiguous output")?;
                Ok(CpuPlan::MakeContiguous {
                    _marker: PhantomData,
                })
            }
        }
    }

    pub(super) fn build_semiring_fast_path_plan<T: Scalar>(
        desc: &SemiringFastPathDescriptor,
        shapes: &[&[usize]],
    ) -> Result<CpuPlan<T>> {
        match desc {
            SemiringFastPathDescriptor::Contract {
                modes_a,
                modes_b,
                modes_c,
            } => {
                validate_shape_count(shapes, 3, "Contract")?;
                validate_rank(shapes[0], modes_a.len(), "Contract input A")?;
                validate_rank(shapes[1], modes_b.len(), "Contract input B")?;
                validate_rank(shapes[2], modes_c.len(), "Contract output C")?;
                for (a_pos, &mode) in modes_a.iter().enumerate() {
                    if let Some(b_pos) = modes_b.iter().position(|&m| m == mode) {
                        if shapes[0][a_pos] != shapes[1][b_pos] {
                            return Err(Error::InvalidArgument(format!(
                                "Contract mode {mode} has mismatched dimensions: A={} vs B={}",
                                shapes[0][a_pos], shapes[1][b_pos]
                            )));
                        }
                    }
                }
                let gemm_spec = build_contract_gemm_spec(modes_a, modes_b, modes_c);
                Ok(CpuPlan::Contract {
                    modes_a: modes_a.clone(),
                    modes_b: modes_b.clone(),
                    modes_c: modes_c.clone(),
                    gemm_spec,
                    _marker: PhantomData,
                })
            }
            SemiringFastPathDescriptor::ElementwiseBinary { op } => {
                validate_shape_count(shapes, 3, "ElementwiseBinary")?;
                validate_shape_eq(shapes[1], shapes[0], "ElementwiseBinary input B")?;
                validate_shape_eq(shapes[2], shapes[0], "ElementwiseBinary output C")?;
                Ok(CpuPlan::ElementwiseBinary {
                    op: *op,
                    _marker: PhantomData,
                })
            }
        }
    }
}

impl<S: Scalar> TensorSemiringCore<Standard<S>> for CpuBackend {
    type Plan = CpuPlan<S>;
    type Context = CpuContext;

    fn plan(
        ctx: &mut CpuContext,
        desc: &SemiringCoreDescriptor,
        shapes: &[&[usize]],
    ) -> Result<CpuPlan<S>> {
        if let Some(cached) = ctx
            .plan_cache
            .get::<CpuPlan<S>, SemiringCoreDescriptor>(desc, shapes)
        {
            return Ok(cached);
        }

        let plan = Self::build_semiring_core_plan::<S>(desc, shapes)?;
        ctx.plan_cache.insert(desc, shapes, plan.clone());
        Ok(plan)
    }

    fn execute(
        ctx: &mut CpuContext,
        plan: &CpuPlan<S>,
        alpha: S,
        inputs: &[&tenferro_tensor::Tensor<S>],
        beta: S,
        output: &mut tenferro_tensor::Tensor<S>,
    ) -> Result<()> {
        execute_semiring_plan(ctx, plan, alpha, inputs, beta, output)
    }
}

impl<S: Scalar> TensorSemiringFastPath<Standard<S>> for CpuBackend {
    type Plan = CpuPlan<S>;
    type Context = CpuContext;

    fn plan(
        ctx: &mut CpuContext,
        desc: &SemiringFastPathDescriptor,
        shapes: &[&[usize]],
    ) -> Result<CpuPlan<S>> {
        if let Some(cached) = ctx
            .plan_cache
            .get::<CpuPlan<S>, SemiringFastPathDescriptor>(desc, shapes)
        {
            return Ok(cached);
        }

        let plan = Self::build_semiring_fast_path_plan::<S>(desc, shapes)?;
        ctx.plan_cache.insert(desc, shapes, plan.clone());
        Ok(plan)
    }

    fn execute(
        ctx: &mut CpuContext,
        plan: &CpuPlan<S>,
        alpha: S,
        inputs: &[&tenferro_tensor::Tensor<S>],
        beta: S,
        output: &mut tenferro_tensor::Tensor<S>,
    ) -> Result<()> {
        execute_semiring_plan(ctx, plan, alpha, inputs, beta, output)
    }

    fn has_fast_path(desc: SemiringFastPathDescriptor) -> bool {
        matches!(
            desc,
            SemiringFastPathDescriptor::Contract { .. }
                | SemiringFastPathDescriptor::ElementwiseBinary {
                    op: SemiringBinaryOp::Add | SemiringBinaryOp::Mul,
                }
        )
    }
}
