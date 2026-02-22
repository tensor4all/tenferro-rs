//! [`TensorPrims`] implementations for tropical algebras on [`CpuBackend`].
//!
//! Each tropical algebra gets its own `impl TensorPrims<XxxAlgebra> for CpuBackend`.
//! The orphan rule is satisfied because `XxxAlgebra` is defined in this crate.
//!
//! Extended operations (Contract, ElementwiseMul) are not supported for
//! tropical algebras — `has_extension_for` always returns `false`.
//!
//! Because tropical scalar types redefine `Add` (= max/min) and `Mul` (= +/×),
//! the standard ScalarBase-based helper functions work correctly: the expression
//! `sum = sum + a * b` becomes `sum = max(sum, a_val + b_val)` for MaxPlus,
//! which is exactly tropical GEMM.

use std::marker::PhantomData;

use strided_traits::ScalarBase;
use strided_view::{StridedView, StridedViewMut};
use tenferro_device::{Error, Result};
use tenferro_prims::{CpuBackend, CpuContext, Extension, PrimDescriptor, ReduceOp, TensorPrims};

use crate::algebra::{MaxMulAlgebra, MaxPlusAlgebra, MinPlusAlgebra};

/// Execution plan for tropical primitive operations on CPU.
///
/// Analogous to [`CpuPlan`](tenferro_prims::CpuPlan) but for tropical
/// algebras. The plan captures pre-computed kernel selection information.
///
/// # Examples
///
/// ```ignore
/// use tenferro_prims::{CpuBackend, CpuContext, TensorPrims, PrimDescriptor, ReduceOp};
/// use tenferro_tropical::{MaxPlusAlgebra, TropicalPlan};
///
/// let mut ctx = CpuContext::new(1);
/// let desc = PrimDescriptor::Reduce {
///     modes_a: vec![0, 1],
///     modes_c: vec![0],
///     op: ReduceOp::Sum,
/// };
/// let plan = <CpuBackend as TensorPrims<MaxPlusAlgebra>>::plan::<f64>(
///     &mut ctx, &desc, &[&[3, 4], &[3]],
/// ).unwrap();
/// ```
pub enum TropicalPlan<T: ScalarBase> {
    /// Plan for batched GEMM under tropical algebra.
    BatchedGemm {
        /// Batch dimension sizes.
        batch_dims: Vec<usize>,
        /// Number of rows.
        m: usize,
        /// Number of columns.
        n: usize,
        /// Contraction dimension.
        k: usize,
        _marker: PhantomData<T>,
    },
    /// Plan for reduction under tropical algebra.
    Reduce {
        /// Axes to reduce over (positions in input).
        reduced_axes: Vec<usize>,
        /// Reduction operation.
        op: ReduceOp,
        _marker: PhantomData<T>,
    },
    /// Plan for trace under tropical algebra.
    Trace {
        /// Paired axis positions in input.
        paired_axes: Vec<(usize, usize)>,
        /// Free axis positions in input (corresponding to output modes).
        free_axes: Vec<usize>,
        _marker: PhantomData<T>,
    },
    /// Plan for permutation.
    Permute {
        /// Permutation mapping: perm[out_axis] = in_axis.
        perm: Vec<usize>,
        _marker: PhantomData<T>,
    },
    /// Plan for anti-trace (AD backward).
    AntiTrace {
        /// Paired axis positions in output.
        paired_axes: Vec<(usize, usize)>,
        /// Free axis positions in output (corresponding to input modes).
        free_axes: Vec<usize>,
        _marker: PhantomData<T>,
    },
    /// Plan for anti-diag (AD backward).
    AntiDiag {
        /// Paired axis positions in output.
        paired_axes: Vec<(usize, usize)>,
        /// Free axis positions in output (corresponding to input modes).
        free_axes: Vec<usize>,
        _marker: PhantomData<T>,
    },
}

// ===========================================================================
// Helpers for multi-index iteration (same pattern as tenferro-prims CPU)
// ===========================================================================

/// Iterate over all index combinations for the given dimensions (column-major order).
fn for_each_index(dims: &[usize], mut f: impl FnMut(&[usize])) {
    let ndim = dims.len();
    if ndim == 0 {
        f(&[]);
        return;
    }
    let total: usize = dims.iter().product();
    if total == 0 {
        return;
    }
    let mut index = vec![0usize; ndim];
    for _ in 0..total {
        f(&index);
        for d in 0..ndim {
            index[d] += 1;
            if index[d] < dims[d] {
                break;
            }
            index[d] = 0;
        }
    }
}

/// Unflatten a linear index to multi-dimensional indices (column-major).
fn unflatten_index(flat: usize, dims: &[usize]) -> Vec<usize> {
    let mut indices = vec![0; dims.len()];
    let mut remainder = flat;
    for d in 0..dims.len() {
        indices[d] = remainder % dims[d];
        remainder /= dims[d];
    }
    indices
}

/// Find the position of a mode label in a mode list.
fn mode_position(modes: &[u32], label: u32) -> Result<usize> {
    modes
        .iter()
        .position(|&m| m == label)
        .ok_or_else(|| Error::InvalidArgument(format!("mode label {label} not found")))
}

/// Scale all elements of the output by `beta`, or zero them if `beta == 0`.
fn scale_output<T: ScalarBase>(output: &mut StridedViewMut<T>, beta: T) {
    let dims = output.dims().to_vec();
    if beta == T::zero() {
        for_each_index(&dims, |idx| {
            output.set(idx, T::zero());
        });
    } else if beta != T::one() {
        for_each_index(&dims, |idx| {
            let old = output.get(idx);
            output.set(idx, beta * old);
        });
    }
}

// ===========================================================================
// Execute helpers (reused across all three tropical algebras)
// ===========================================================================

fn execute_batched_gemm<T: ScalarBase>(
    alpha: T,
    inputs: &[&StridedView<T>],
    beta: T,
    output: &mut StridedViewMut<T>,
    batch_dims: &[usize],
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let a = inputs[0];
    let b = inputs[1];
    let batch_size: usize = if batch_dims.is_empty() {
        1
    } else {
        batch_dims.iter().product()
    };

    for batch_flat in 0..batch_size {
        let batch_idx = unflatten_index(batch_flat, batch_dims);
        for i in 0..m {
            for j in 0..n {
                // Initialize sum to the semiring zero (additive identity).
                // For MaxPlus: -inf, for MinPlus: +inf, for MaxMul: 0.
                let mut sum = T::zero();
                for kk in 0..k {
                    let mut a_idx = batch_idx.clone();
                    a_idx.push(i);
                    a_idx.push(kk);
                    let mut b_idx = batch_idx.clone();
                    b_idx.push(kk);
                    b_idx.push(j);
                    // sum = sum ⊕ (a[i,kk] ⊗ b[kk,j])
                    // For MaxPlus: sum = max(sum, a + b)
                    sum = sum + a.get(&a_idx) * b.get(&b_idx);
                }
                let mut c_idx = batch_idx.clone();
                c_idx.push(i);
                c_idx.push(j);
                let old = if beta == T::zero() {
                    T::zero()
                } else {
                    beta * output.get(&c_idx)
                };
                output.set(&c_idx, alpha * sum + old);
            }
        }
    }
    Ok(())
}

fn execute_reduce_sum<T: ScalarBase>(
    alpha: T,
    input: &StridedView<T>,
    beta: T,
    output: &mut StridedViewMut<T>,
    reduced_axes: &[usize],
) -> Result<()> {
    let in_dims = input.dims().to_vec();
    let out_dims = output.dims().to_vec();
    let reduced_dims: Vec<usize> = reduced_axes.iter().map(|&ax| in_dims[ax]).collect();
    let reduced_total: usize = reduced_dims.iter().product();

    for_each_index(&out_dims, |out_idx| {
        // For tropical: sum starts at zero (additive identity = -inf/+inf/0)
        let mut sum = T::zero();
        for red_flat in 0..reduced_total {
            let red_idx = unflatten_index(red_flat, &reduced_dims);
            let mut in_idx = Vec::with_capacity(in_dims.len());
            let mut out_pos = 0;
            let mut red_pos = 0;
            for ax in 0..in_dims.len() {
                if red_pos < reduced_axes.len() && reduced_axes[red_pos] == ax {
                    in_idx.push(red_idx[red_pos]);
                    red_pos += 1;
                } else {
                    in_idx.push(out_idx[out_pos]);
                    out_pos += 1;
                }
            }
            // sum = sum ⊕ input[in_idx]
            sum = sum + input.get(&in_idx);
        }
        let old = if beta == T::zero() {
            T::zero()
        } else {
            beta * output.get(out_idx)
        };
        output.set(out_idx, alpha * sum + old);
    });
    Ok(())
}

fn execute_trace<T: ScalarBase>(
    alpha: T,
    input: &StridedView<T>,
    beta: T,
    output: &mut StridedViewMut<T>,
    paired_axes: &[(usize, usize)],
    free_axes: &[usize],
) -> Result<()> {
    let in_dims = input.dims().to_vec();
    let out_dims = output.dims().to_vec();
    let diag_dim = in_dims[paired_axes[0].0];

    for_each_index(&out_dims, |out_idx| {
        let mut sum = T::zero();
        for d in 0..diag_dim {
            let mut in_idx = vec![0; in_dims.len()];
            for (out_pos, &in_ax) in free_axes.iter().enumerate() {
                in_idx[in_ax] = out_idx[out_pos];
            }
            for &(ax1, ax2) in paired_axes {
                in_idx[ax1] = d;
                in_idx[ax2] = d;
            }
            sum = sum + input.get(&in_idx);
        }
        let old = if beta == T::zero() {
            T::zero()
        } else {
            beta * output.get(out_idx)
        };
        output.set(out_idx, alpha * sum + old);
    });
    Ok(())
}

fn execute_permute<T: ScalarBase>(
    alpha: T,
    input: &StridedView<T>,
    beta: T,
    output: &mut StridedViewMut<T>,
    perm: &[usize],
) -> Result<()> {
    let permuted = input
        .permute(perm)
        .map_err(|e| Error::StrideError(e.to_string()))?;

    if alpha == T::one() && beta == T::zero() {
        strided_perm::copy_into(output, &permuted)
            .map_err(|e| Error::StrideError(e.to_string()))?;
    } else {
        let dims = output.dims().to_vec();
        for_each_index(&dims, |idx| {
            let val = alpha * permuted.get(idx);
            if beta == T::zero() {
                output.set(idx, val);
            } else {
                output.set(idx, val + beta * output.get(idx));
            }
        });
    }
    Ok(())
}

fn execute_anti_trace<T: ScalarBase>(
    alpha: T,
    input: &StridedView<T>,
    beta: T,
    output: &mut StridedViewMut<T>,
    paired_axes: &[(usize, usize)],
    free_axes: &[usize],
) -> Result<()> {
    scale_output(output, beta);

    let in_dims = input.dims().to_vec();
    let out_dims = output.dims().to_vec();
    let diag_dim = out_dims[paired_axes[0].0];

    for_each_index(&in_dims, |in_idx| {
        let val = alpha * input.get(in_idx);
        for d in 0..diag_dim {
            let mut out_idx = vec![0; out_dims.len()];
            for (in_pos, &out_ax) in free_axes.iter().enumerate() {
                out_idx[out_ax] = in_idx[in_pos];
            }
            for &(ax1, ax2) in paired_axes {
                out_idx[ax1] = d;
                out_idx[ax2] = d;
            }
            let old = output.get(&out_idx);
            output.set(&out_idx, old + val);
        }
    });
    Ok(())
}

fn execute_anti_diag<T: ScalarBase>(
    alpha: T,
    input: &StridedView<T>,
    beta: T,
    output: &mut StridedViewMut<T>,
    paired_axes: &[(usize, usize)],
    free_axes: &[usize],
) -> Result<()> {
    scale_output(output, beta);

    let in_dims = input.dims().to_vec();
    let out_dims = output.dims().to_vec();

    for_each_index(&in_dims, |in_idx| {
        let val = alpha * input.get(in_idx);
        let mut out_idx = vec![0; out_dims.len()];
        for (in_pos, &out_ax) in free_axes.iter().enumerate() {
            out_idx[out_ax] = in_idx[in_pos];
        }
        for &(ax1, ax2) in paired_axes {
            out_idx[ax2] = out_idx[ax1];
        }
        let old = output.get(&out_idx);
        output.set(&out_idx, old + val);
    });
    Ok(())
}

// ===========================================================================
// Plan construction (shared logic for all three tropical algebras)
// ===========================================================================

fn tropical_plan<T: ScalarBase>(desc: &PrimDescriptor) -> Result<TropicalPlan<T>> {
    match desc {
        PrimDescriptor::BatchedGemm {
            batch_dims,
            m,
            n,
            k,
        } => Ok(TropicalPlan::BatchedGemm {
            batch_dims: batch_dims.clone(),
            m: *m,
            n: *n,
            k: *k,
            _marker: PhantomData,
        }),

        PrimDescriptor::Reduce {
            modes_a,
            modes_c,
            op,
        } => {
            let reduced_axes: Vec<usize> = modes_a
                .iter()
                .enumerate()
                .filter(|(_, m)| !modes_c.contains(m))
                .map(|(i, _)| i)
                .collect();
            Ok(TropicalPlan::Reduce {
                reduced_axes,
                op: *op,
                _marker: PhantomData,
            })
        }

        PrimDescriptor::Trace {
            modes_a,
            modes_c,
            paired,
        } => {
            let paired_axes: Vec<(usize, usize)> = paired
                .iter()
                .map(|(m1, m2)| Ok((mode_position(modes_a, *m1)?, mode_position(modes_a, *m2)?)))
                .collect::<Result<_>>()?;
            let free_axes: Vec<usize> = modes_c
                .iter()
                .map(|m| mode_position(modes_a, *m))
                .collect::<Result<_>>()?;
            Ok(TropicalPlan::Trace {
                paired_axes,
                free_axes,
                _marker: PhantomData,
            })
        }

        PrimDescriptor::Permute { modes_a, modes_b } => {
            let perm: Vec<usize> = modes_b
                .iter()
                .map(|m| mode_position(modes_a, *m))
                .collect::<Result<_>>()?;
            Ok(TropicalPlan::Permute {
                perm,
                _marker: PhantomData,
            })
        }

        PrimDescriptor::AntiTrace {
            modes_a,
            modes_c,
            paired,
        } => {
            let paired_axes: Vec<(usize, usize)> = paired
                .iter()
                .map(|(m1, m2)| Ok((mode_position(modes_c, *m1)?, mode_position(modes_c, *m2)?)))
                .collect::<Result<_>>()?;
            let free_axes: Vec<usize> = modes_a
                .iter()
                .map(|m| mode_position(modes_c, *m))
                .collect::<Result<_>>()?;
            Ok(TropicalPlan::AntiTrace {
                paired_axes,
                free_axes,
                _marker: PhantomData,
            })
        }

        PrimDescriptor::AntiDiag {
            modes_a,
            modes_c,
            paired,
        } => {
            let paired_axes: Vec<(usize, usize)> = paired
                .iter()
                .map(|(m1, m2)| Ok((mode_position(modes_c, *m1)?, mode_position(modes_c, *m2)?)))
                .collect::<Result<_>>()?;
            let free_axes: Vec<usize> = modes_a
                .iter()
                .map(|m| mode_position(modes_c, *m))
                .collect::<Result<_>>()?;
            Ok(TropicalPlan::AntiDiag {
                paired_axes,
                free_axes,
                _marker: PhantomData,
            })
        }

        PrimDescriptor::ElementwiseUnary { .. }
        | PrimDescriptor::Contract { .. }
        | PrimDescriptor::ElementwiseMul
        | PrimDescriptor::MakeContiguous => Err(Error::InvalidArgument(
            "tropical algebras do not support this operation".into(),
        )),
    }
}

fn tropical_execute<T: ScalarBase>(
    plan: &TropicalPlan<T>,
    alpha: T,
    inputs: &[&StridedView<T>],
    beta: T,
    output: &mut StridedViewMut<T>,
) -> Result<()> {
    match plan {
        TropicalPlan::BatchedGemm {
            batch_dims,
            m,
            n,
            k,
            ..
        } => execute_batched_gemm(alpha, inputs, beta, output, batch_dims, *m, *n, *k),

        TropicalPlan::Reduce {
            reduced_axes, op, ..
        } => match op {
            ReduceOp::Sum => execute_reduce_sum(alpha, inputs[0], beta, output, reduced_axes),
            ReduceOp::Max | ReduceOp::Min => {
                // For tropical types, "Sum" reduction already uses tropical addition
                // (max or min), so ReduceOp::Sum is the correct choice for callers.
                // Max/Min are not meaningful as separate ops for tropical scalars.
                Err(Error::InvalidArgument(
                    "use ReduceOp::Sum for tropical reduction (+ is already max/min)".into(),
                ))
            }
        },

        TropicalPlan::Trace {
            paired_axes,
            free_axes,
            ..
        } => execute_trace(alpha, inputs[0], beta, output, paired_axes, free_axes),

        TropicalPlan::Permute { perm, .. } => execute_permute(alpha, inputs[0], beta, output, perm),

        TropicalPlan::AntiTrace {
            paired_axes,
            free_axes,
            ..
        } => execute_anti_trace(alpha, inputs[0], beta, output, paired_axes, free_axes),

        TropicalPlan::AntiDiag {
            paired_axes,
            free_axes,
            ..
        } => execute_anti_diag(alpha, inputs[0], beta, output, paired_axes, free_axes),
    }
}

// ===========================================================================
// impl TensorPrims<MaxPlusAlgebra> for CpuBackend
// ===========================================================================

impl TensorPrims<MaxPlusAlgebra> for CpuBackend {
    type Plan<T: ScalarBase> = TropicalPlan<T>;
    type Context = CpuContext;

    fn plan<T: ScalarBase>(
        _ctx: &mut CpuContext,
        desc: &PrimDescriptor,
        _shapes: &[&[usize]],
    ) -> Result<TropicalPlan<T>> {
        tropical_plan(desc)
    }

    fn execute<T: ScalarBase>(
        _ctx: &mut CpuContext,
        plan: &TropicalPlan<T>,
        alpha: T,
        inputs: &[&StridedView<T>],
        beta: T,
        output: &mut StridedViewMut<T>,
    ) -> Result<()> {
        tropical_execute(plan, alpha, inputs, beta, output)
    }

    /// Tropical backends do not support extended operations.
    fn has_extension_for<T: ScalarBase>(_ext: Extension) -> bool {
        false
    }
}

// ===========================================================================
// impl TensorPrims<MinPlusAlgebra> for CpuBackend
// ===========================================================================

impl TensorPrims<MinPlusAlgebra> for CpuBackend {
    type Plan<T: ScalarBase> = TropicalPlan<T>;
    type Context = CpuContext;

    fn plan<T: ScalarBase>(
        _ctx: &mut CpuContext,
        desc: &PrimDescriptor,
        _shapes: &[&[usize]],
    ) -> Result<TropicalPlan<T>> {
        tropical_plan(desc)
    }

    fn execute<T: ScalarBase>(
        _ctx: &mut CpuContext,
        plan: &TropicalPlan<T>,
        alpha: T,
        inputs: &[&StridedView<T>],
        beta: T,
        output: &mut StridedViewMut<T>,
    ) -> Result<()> {
        tropical_execute(plan, alpha, inputs, beta, output)
    }

    /// Tropical backends do not support extended operations.
    fn has_extension_for<T: ScalarBase>(_ext: Extension) -> bool {
        false
    }
}

// ===========================================================================
// impl TensorPrims<MaxMulAlgebra> for CpuBackend
// ===========================================================================

impl TensorPrims<MaxMulAlgebra> for CpuBackend {
    type Plan<T: ScalarBase> = TropicalPlan<T>;
    type Context = CpuContext;

    fn plan<T: ScalarBase>(
        _ctx: &mut CpuContext,
        desc: &PrimDescriptor,
        _shapes: &[&[usize]],
    ) -> Result<TropicalPlan<T>> {
        tropical_plan(desc)
    }

    fn execute<T: ScalarBase>(
        _ctx: &mut CpuContext,
        plan: &TropicalPlan<T>,
        alpha: T,
        inputs: &[&StridedView<T>],
        beta: T,
        output: &mut StridedViewMut<T>,
    ) -> Result<()> {
        tropical_execute(plan, alpha, inputs, beta, output)
    }

    /// Tropical backends do not support extended operations.
    fn has_extension_for<T: ScalarBase>(_ext: Extension) -> bool {
        false
    }
}
