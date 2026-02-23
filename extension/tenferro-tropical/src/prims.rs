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

use std::collections::HashSet;
use std::marker::PhantomData;

use strided_view::{StridedView, StridedViewMut};
use tenferro_algebra::Scalar;
use tenferro_device::{Error, Result};
use tenferro_prims::{CpuBackend, CpuContext, Extension, PrimDescriptor, ReduceOp, TensorPrims};
use tenferro_tensor::Tensor;
use tropical_gemm::{TropicalMaxMul, TropicalMaxPlus, TropicalMinPlus, TropicalSemiring};

use crate::algebra::{MaxMulAlgebra, MaxPlusAlgebra, MinPlusAlgebra};
use crate::scalar::{MaxMul, MaxPlus, MinPlus};

/// Convert a CPU tensor to an immutable strided view.
pub(crate) fn tensor_to_view<T: Scalar>(t: &Tensor<T>) -> Result<StridedView<'_, T>> {
    let data = t
        .buffer()
        .as_slice()
        .ok_or_else(|| Error::DeviceError("GPU tensor passed to CPU backend".into()))?;
    StridedView::new(data, t.dims(), t.strides(), t.offset())
        .map_err(|e| Error::StrideError(format!("{e}")))
}

/// Convert a CPU tensor to a mutable strided view.
fn tensor_to_view_mut<T: Scalar>(t: &mut Tensor<T>) -> Result<StridedViewMut<'_, T>> {
    let dims = t.dims().to_vec();
    let strides = t.strides().to_vec();
    let offset = t.offset();
    let data = t
        .buffer_mut()
        .as_mut_slice()
        .ok_or_else(|| Error::DeviceError("GPU tensor passed to CPU backend".into()))?;
    StridedViewMut::new(data, &dims, &strides, offset)
        .map_err(|e| Error::StrideError(format!("{e}")))
}

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
#[derive(Debug)]
pub enum TropicalPlan<T: Scalar> {
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
    /// Plan for making a tensor contiguous.
    MakeContiguous { _marker: PhantomData<T> },
}

// ===========================================================================
// Helpers for multi-index iteration (same pattern as tenferro-prims CPU)
// ===========================================================================

/// Iterate over all index combinations for the given dimensions (column-major order).
pub(crate) fn for_each_index(dims: &[usize], mut f: impl FnMut(&[usize])) {
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
pub(crate) fn unflatten_index(flat: usize, dims: &[usize]) -> Vec<usize> {
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
fn scale_output<T: Scalar>(output: &mut StridedViewMut<T>, beta: T) {
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

/// Trait for dispatching to SIMD-optimized tropical GEMM via tropical-gemm crate.
///
/// Maps tenferro's tropical scalar types to tropical-gemm's types and calls
/// `tropical_matmul_strided_batched` for SIMD-accelerated computation.
/// Generic over the inner scalar type (f32 or f64).
trait TropicalGemmDispatch: Scalar {
    /// The inner floating-point type (f32 or f64).
    type Inner: Copy + Default;
    /// Extract the inner scalar value.
    fn inner_value(&self) -> Self::Inner;
    /// Wrap an inner scalar value back into the tropical type.
    fn from_inner(v: Self::Inner) -> Self;
    /// Execute SIMD-optimized batched GEMM.
    /// Input: row-major buffers packed per batch element.
    /// Returns: row-major result buffer.
    fn dispatch_gemm(
        a: &[Self::Inner],
        b: &[Self::Inner],
        batch_size: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<Self::Inner>;
}

/// Implements `TropicalGemmDispatch` for a tropical scalar type.
macro_rules! impl_tropical_gemm_dispatch {
    ($tropical_ty:ident, $inner:ty, $gemm_ty:ty) => {
        impl TropicalGemmDispatch for $tropical_ty<$inner> {
            type Inner = $inner;
            #[inline]
            fn inner_value(&self) -> $inner {
                self.0
            }
            #[inline]
            fn from_inner(v: $inner) -> Self {
                $tropical_ty(v)
            }
            fn dispatch_gemm(
                a: &[$inner],
                b: &[$inner],
                batch_size: usize,
                m: usize,
                k: usize,
                n: usize,
            ) -> Vec<$inner> {
                if batch_size <= 1 {
                    let result = tropical_gemm::tropical_matmul::<$gemm_ty>(a, m, k, b, n);
                    result.iter().map(|v| v.value()).collect()
                } else {
                    let result = tropical_gemm::tropical_matmul_strided_batched::<$gemm_ty>(
                        a, b, batch_size, m, k, n,
                    );
                    result.iter().map(|v| v.value()).collect()
                }
            }
        }
    };
}

// f64 impls
impl_tropical_gemm_dispatch!(MaxPlus, f64, TropicalMaxPlus<f64>);
impl_tropical_gemm_dispatch!(MinPlus, f64, TropicalMinPlus<f64>);
impl_tropical_gemm_dispatch!(MaxMul, f64, TropicalMaxMul<f64>);

// f32 impls
impl_tropical_gemm_dispatch!(MaxPlus, f32, TropicalMaxPlus<f32>);
impl_tropical_gemm_dispatch!(MinPlus, f32, TropicalMinPlus<f32>);
impl_tropical_gemm_dispatch!(MaxMul, f32, TropicalMaxMul<f32>);

/// SIMD-optimized tropical GEMM using the tropical-gemm crate.
///
/// Extracts data into row-major contiguous buffers, calls tropical-gemm,
/// and writes results back with alpha/beta scaling.
fn execute_batched_gemm_optimized<T: Scalar + TropicalGemmDispatch>(
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
    let batch_size: usize = batch_dims.iter().product::<usize>().max(1);

    // Extract A into contiguous row-major buffer: [batch, m, k] row-major
    let a_total = batch_size * m * k;
    let b_total = batch_size * k * n;
    let mut a_buf: Vec<T::Inner> = Vec::with_capacity(a_total);
    let mut b_buf: Vec<T::Inner> = Vec::with_capacity(b_total);

    for batch_flat in 0..batch_size {
        let batch_idx = unflatten_index(batch_flat, batch_dims);
        for i in 0..m {
            for j in 0..k {
                let mut idx = batch_idx.clone();
                idx.push(i);
                idx.push(j);
                a_buf.push(a.get(&idx).inner_value());
            }
        }
    }
    for batch_flat in 0..batch_size {
        let batch_idx = unflatten_index(batch_flat, batch_dims);
        for i in 0..k {
            for j in 0..n {
                let mut idx = batch_idx.clone();
                idx.push(i);
                idx.push(j);
                b_buf.push(b.get(&idx).inner_value());
            }
        }
    }

    // Call SIMD-optimized tropical GEMM
    let result = T::dispatch_gemm(&a_buf, &b_buf, batch_size, m, k, n);

    // Write results back with alpha/beta scaling
    for batch_flat in 0..batch_size {
        let batch_idx = unflatten_index(batch_flat, batch_dims);
        for i in 0..m {
            for j in 0..n {
                let mut c_idx = batch_idx.clone();
                c_idx.push(i);
                c_idx.push(j);
                let flat = batch_flat * m * n + i * n + j;
                let val = T::from_inner(result[flat]);
                let old = if beta == T::zero() {
                    T::zero()
                } else {
                    beta * output.get(&c_idx)
                };
                output.set(&c_idx, alpha * val + old);
            }
        }
    }
    Ok(())
}

/// Fallback loop-based tropical GEMM for types without SIMD dispatch.
fn execute_batched_gemm_fallback<T: Scalar>(
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
                let mut sum = T::zero();
                for kk in 0..k {
                    let mut a_idx = batch_idx.clone();
                    a_idx.push(i);
                    a_idx.push(kk);
                    let mut b_idx = batch_idx.clone();
                    b_idx.push(kk);
                    b_idx.push(j);
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

fn execute_reduce_sum<T: Scalar>(
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

fn execute_trace<T: Scalar>(
    alpha: T,
    input: &StridedView<T>,
    beta: T,
    output: &mut StridedViewMut<T>,
    paired_axes: &[(usize, usize)],
    free_axes: &[usize],
) -> Result<()> {
    if paired_axes.is_empty() {
        return Err(Error::InvalidArgument(
            "trace requires at least one paired axis".into(),
        ));
    }

    let in_dims = input.dims().to_vec();
    let out_dims = output.dims().to_vec();
    for &(ax1, ax2) in paired_axes {
        if ax1 >= in_dims.len() || ax2 >= in_dims.len() {
            return Err(Error::InvalidArgument(
                "trace paired axis out of bounds".into(),
            ));
        }
    }
    for &ax in free_axes {
        if ax >= in_dims.len() {
            return Err(Error::InvalidArgument(
                "trace free axis out of bounds".into(),
            ));
        }
    }
    if out_dims.len() != free_axes.len() {
        return Err(Error::InvalidArgument(
            "trace output rank does not match free axes".into(),
        ));
    }

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

fn execute_permute<T: Scalar>(
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

fn execute_anti_trace<T: Scalar>(
    alpha: T,
    input: &StridedView<T>,
    beta: T,
    output: &mut StridedViewMut<T>,
    paired_axes: &[(usize, usize)],
    free_axes: &[usize],
) -> Result<()> {
    if paired_axes.is_empty() {
        return Err(Error::InvalidArgument(
            "anti-trace requires at least one paired axis".into(),
        ));
    }

    scale_output(output, beta);

    let in_dims = input.dims().to_vec();
    let out_dims = output.dims().to_vec();
    for &(ax1, ax2) in paired_axes {
        if ax1 >= out_dims.len() || ax2 >= out_dims.len() {
            return Err(Error::InvalidArgument(
                "anti-trace paired axis out of bounds".into(),
            ));
        }
    }
    for &ax in free_axes {
        if ax >= out_dims.len() {
            return Err(Error::InvalidArgument(
                "anti-trace free axis out of bounds".into(),
            ));
        }
    }
    if in_dims.len() != free_axes.len() {
        return Err(Error::InvalidArgument(
            "anti-trace input rank does not match free axes".into(),
        ));
    }
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

fn execute_anti_diag<T: Scalar>(
    alpha: T,
    input: &StridedView<T>,
    beta: T,
    output: &mut StridedViewMut<T>,
    paired_axes: &[(usize, usize)],
    free_axes: &[usize],
) -> Result<()> {
    if paired_axes.is_empty() {
        return Err(Error::InvalidArgument(
            "anti-diag requires at least one paired axis".into(),
        ));
    }

    scale_output(output, beta);

    let in_dims = input.dims().to_vec();
    let out_dims = output.dims().to_vec();
    for &(ax1, ax2) in paired_axes {
        if ax1 >= out_dims.len() || ax2 >= out_dims.len() {
            return Err(Error::InvalidArgument(
                "anti-diag paired axis out of bounds".into(),
            ));
        }
    }
    for &ax in free_axes {
        if ax >= out_dims.len() {
            return Err(Error::InvalidArgument(
                "anti-diag free axis out of bounds".into(),
            ));
        }
    }
    if in_dims.len() != free_axes.len() {
        return Err(Error::InvalidArgument(
            "anti-diag input rank does not match free axes".into(),
        ));
    }

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

fn ensure_shape_count(shapes: &[&[usize]], expected: usize, op: &str) -> Result<()> {
    if shapes.len() != expected {
        return Err(Error::InvalidArgument(format!(
            "{op} expects {expected} shapes, got {}",
            shapes.len()
        )));
    }
    Ok(())
}

fn ensure_unique_modes(modes: &[u32], name: &str) -> Result<()> {
    let mut seen = HashSet::new();
    for &m in modes {
        if !seen.insert(m) {
            return Err(Error::InvalidArgument(format!(
                "{name} contains duplicate mode label {m}"
            )));
        }
    }
    Ok(())
}

fn ensure_pair_labels_unique(paired: &[(u32, u32)], name: &str) -> Result<()> {
    let mut seen = HashSet::new();
    for &(m1, m2) in paired {
        if m1 == m2 {
            return Err(Error::InvalidArgument(format!(
                "{name} contains invalid pair ({m1},{m2})"
            )));
        }
        if !seen.insert(m1) || !seen.insert(m2) {
            return Err(Error::InvalidArgument(format!(
                "{name} contains duplicated paired label"
            )));
        }
    }
    Ok(())
}

fn tropical_plan<T: Scalar>(desc: &PrimDescriptor, shapes: &[&[usize]]) -> Result<TropicalPlan<T>> {
    match desc {
        PrimDescriptor::BatchedGemm {
            batch_dims,
            m,
            n,
            k,
        } => {
            ensure_shape_count(shapes, 3, "BatchedGemm")?;
            let a_shape = shapes[0];
            let b_shape = shapes[1];
            let c_shape = shapes[2];
            let expected_rank = batch_dims.len() + 2;
            if a_shape.len() != expected_rank
                || b_shape.len() != expected_rank
                || c_shape.len() != expected_rank
            {
                return Err(Error::InvalidArgument(
                    "BatchedGemm rank mismatch between descriptor and shapes".into(),
                ));
            }
            for (i, &bd) in batch_dims.iter().enumerate() {
                if a_shape[i] != bd || b_shape[i] != bd || c_shape[i] != bd {
                    return Err(Error::InvalidArgument(
                        "BatchedGemm batch dimensions do not match shapes".into(),
                    ));
                }
            }
            let off = batch_dims.len();
            if a_shape[off] != *m || a_shape[off + 1] != *k {
                return Err(Error::InvalidArgument(
                    "BatchedGemm A shape mismatch".into(),
                ));
            }
            if b_shape[off] != *k || b_shape[off + 1] != *n {
                return Err(Error::InvalidArgument(
                    "BatchedGemm B shape mismatch".into(),
                ));
            }
            if c_shape[off] != *m || c_shape[off + 1] != *n {
                return Err(Error::InvalidArgument(
                    "BatchedGemm C shape mismatch".into(),
                ));
            }

            Ok(TropicalPlan::BatchedGemm {
                batch_dims: batch_dims.clone(),
                m: *m,
                n: *n,
                k: *k,
                _marker: PhantomData,
            })
        }

        PrimDescriptor::Reduce {
            modes_a,
            modes_c,
            op,
        } => {
            ensure_shape_count(shapes, 2, "Reduce")?;
            ensure_unique_modes(modes_a, "modes_a")?;
            ensure_unique_modes(modes_c, "modes_c")?;
            let a_shape = shapes[0];
            let c_shape = shapes[1];
            if modes_a.len() != a_shape.len() || modes_c.len() != c_shape.len() {
                return Err(Error::InvalidArgument(
                    "Reduce mode rank does not match shape rank".into(),
                ));
            }
            for &m in modes_c {
                if !modes_a.contains(&m) {
                    return Err(Error::InvalidArgument(
                        "Reduce modes_c must be a subset of modes_a".into(),
                    ));
                }
            }
            for (out_ax, &m) in modes_c.iter().enumerate() {
                let in_ax = mode_position(modes_a, m)?;
                if a_shape[in_ax] != c_shape[out_ax] {
                    return Err(Error::InvalidArgument(
                        "Reduce output shape does not match input modes".into(),
                    ));
                }
            }

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
            ensure_shape_count(shapes, 2, "Trace")?;
            ensure_unique_modes(modes_a, "modes_a")?;
            ensure_unique_modes(modes_c, "modes_c")?;
            if paired.is_empty() {
                return Err(Error::InvalidArgument(
                    "Trace requires non-empty paired axes".into(),
                ));
            }
            ensure_pair_labels_unique(paired, "Trace paired")?;
            let a_shape = shapes[0];
            let c_shape = shapes[1];
            if modes_a.len() != a_shape.len() || modes_c.len() != c_shape.len() {
                return Err(Error::InvalidArgument(
                    "Trace mode rank does not match shape rank".into(),
                ));
            }

            let paired_labels: HashSet<u32> =
                paired.iter().flat_map(|(m1, m2)| [*m1, *m2]).collect();
            for &(m1, m2) in paired {
                if !modes_a.contains(&m1) || !modes_a.contains(&m2) {
                    return Err(Error::InvalidArgument(
                        "Trace paired labels must exist in modes_a".into(),
                    ));
                }
                if modes_c.contains(&m1) || modes_c.contains(&m2) {
                    return Err(Error::InvalidArgument(
                        "Trace paired labels must be reduced (not present in modes_c)".into(),
                    ));
                }
                let ax1 = mode_position(modes_a, m1)?;
                let ax2 = mode_position(modes_a, m2)?;
                if a_shape[ax1] != a_shape[ax2] {
                    return Err(Error::InvalidArgument(
                        "Trace paired dimensions must be equal".into(),
                    ));
                }
            }
            for &m in modes_a {
                if !modes_c.contains(&m) && !paired_labels.contains(&m) {
                    return Err(Error::InvalidArgument(
                        "Trace modes_a contains labels neither free nor paired".into(),
                    ));
                }
            }
            for (out_ax, &m) in modes_c.iter().enumerate() {
                if paired_labels.contains(&m) {
                    return Err(Error::InvalidArgument(
                        "Trace free labels must not be in paired set".into(),
                    ));
                }
                let in_ax = mode_position(modes_a, m)?;
                if a_shape[in_ax] != c_shape[out_ax] {
                    return Err(Error::InvalidArgument(
                        "Trace output shape does not match free modes".into(),
                    ));
                }
            }

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
            ensure_shape_count(shapes, 2, "Permute")?;
            ensure_unique_modes(modes_a, "modes_a")?;
            ensure_unique_modes(modes_b, "modes_b")?;
            let a_shape = shapes[0];
            let b_shape = shapes[1];
            if modes_a.len() != a_shape.len()
                || modes_b.len() != b_shape.len()
                || modes_a.len() != modes_b.len()
            {
                return Err(Error::InvalidArgument(
                    "Permute mode rank does not match shape rank".into(),
                ));
            }
            for &m in modes_b {
                if !modes_a.contains(&m) {
                    return Err(Error::InvalidArgument(
                        "Permute modes_b must be a permutation of modes_a".into(),
                    ));
                }
            }
            for (out_ax, &m) in modes_b.iter().enumerate() {
                let in_ax = mode_position(modes_a, m)?;
                if a_shape[in_ax] != b_shape[out_ax] {
                    return Err(Error::InvalidArgument(
                        "Permute output shape does not match permutation".into(),
                    ));
                }
            }

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
            ensure_shape_count(shapes, 2, "AntiTrace")?;
            ensure_unique_modes(modes_a, "modes_a")?;
            ensure_unique_modes(modes_c, "modes_c")?;
            if paired.is_empty() {
                return Err(Error::InvalidArgument(
                    "AntiTrace requires non-empty paired axes".into(),
                ));
            }
            ensure_pair_labels_unique(paired, "AntiTrace paired")?;
            let a_shape = shapes[0];
            let c_shape = shapes[1];
            if modes_a.len() != a_shape.len() || modes_c.len() != c_shape.len() {
                return Err(Error::InvalidArgument(
                    "AntiTrace mode rank does not match shape rank".into(),
                ));
            }

            let paired_labels: HashSet<u32> =
                paired.iter().flat_map(|(m1, m2)| [*m1, *m2]).collect();
            for &(m1, m2) in paired {
                if !modes_c.contains(&m1) || !modes_c.contains(&m2) {
                    return Err(Error::InvalidArgument(
                        "AntiTrace paired labels must exist in modes_c".into(),
                    ));
                }
                if modes_a.contains(&m1) || modes_a.contains(&m2) {
                    return Err(Error::InvalidArgument(
                        "AntiTrace paired labels must not be in modes_a".into(),
                    ));
                }
                let ax1 = mode_position(modes_c, m1)?;
                let ax2 = mode_position(modes_c, m2)?;
                if c_shape[ax1] != c_shape[ax2] {
                    return Err(Error::InvalidArgument(
                        "AntiTrace paired dimensions must be equal".into(),
                    ));
                }
            }
            for &m in modes_c {
                if !modes_a.contains(&m) && !paired_labels.contains(&m) {
                    return Err(Error::InvalidArgument(
                        "AntiTrace modes_c contains labels neither free nor paired".into(),
                    ));
                }
            }
            for (in_ax, &m) in modes_a.iter().enumerate() {
                if paired_labels.contains(&m) {
                    return Err(Error::InvalidArgument(
                        "AntiTrace free labels must not be in paired set".into(),
                    ));
                }
                let out_ax = mode_position(modes_c, m)?;
                if a_shape[in_ax] != c_shape[out_ax] {
                    return Err(Error::InvalidArgument(
                        "AntiTrace input shape does not match output free modes".into(),
                    ));
                }
            }

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
            ensure_shape_count(shapes, 2, "AntiDiag")?;
            ensure_unique_modes(modes_a, "modes_a")?;
            ensure_unique_modes(modes_c, "modes_c")?;
            if paired.is_empty() {
                return Err(Error::InvalidArgument(
                    "AntiDiag requires non-empty paired axes".into(),
                ));
            }
            ensure_pair_labels_unique(paired, "AntiDiag paired")?;
            let a_shape = shapes[0];
            let c_shape = shapes[1];
            if modes_a.len() != a_shape.len() || modes_c.len() != c_shape.len() {
                return Err(Error::InvalidArgument(
                    "AntiDiag mode rank does not match shape rank".into(),
                ));
            }

            let paired_labels: HashSet<u32> =
                paired.iter().flat_map(|(m1, m2)| [*m1, *m2]).collect();
            let free_labels: HashSet<u32> = modes_a.iter().copied().collect();
            for &(m1, m2) in paired {
                if !modes_c.contains(&m1) || !modes_c.contains(&m2) {
                    return Err(Error::InvalidArgument(
                        "AntiDiag paired labels must exist in modes_c".into(),
                    ));
                }
                if !free_labels.contains(&m1) {
                    return Err(Error::InvalidArgument(
                        "AntiDiag first paired label must exist in modes_a".into(),
                    ));
                }
                if free_labels.contains(&m2) {
                    return Err(Error::InvalidArgument(
                        "AntiDiag second paired label must not exist in modes_a".into(),
                    ));
                }
                let ax1 = mode_position(modes_c, m1)?;
                let ax2 = mode_position(modes_c, m2)?;
                if c_shape[ax1] != c_shape[ax2] {
                    return Err(Error::InvalidArgument(
                        "AntiDiag paired dimensions must be equal".into(),
                    ));
                }
            }
            for &m in modes_c {
                if !free_labels.contains(&m) && !paired_labels.contains(&m) {
                    return Err(Error::InvalidArgument(
                        "AntiDiag modes_c contains labels neither free nor paired".into(),
                    ));
                }
            }
            for (in_ax, &m) in modes_a.iter().enumerate() {
                let out_ax = mode_position(modes_c, m)?;
                if a_shape[in_ax] != c_shape[out_ax] {
                    return Err(Error::InvalidArgument(
                        "AntiDiag input shape does not match output free modes".into(),
                    ));
                }
            }

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

        PrimDescriptor::MakeContiguous => {
            ensure_shape_count(shapes, 2, "MakeContiguous")?;
            if shapes[0] != shapes[1] {
                return Err(Error::InvalidArgument(
                    "MakeContiguous input and output shapes must match".into(),
                ));
            }
            Ok(TropicalPlan::MakeContiguous {
                _marker: PhantomData,
            })
        }

        PrimDescriptor::ElementwiseUnary { .. }
        | PrimDescriptor::Contract { .. }
        | PrimDescriptor::ElementwiseMul => Err(Error::InvalidArgument(
            "tropical algebras do not support this operation".into(),
        )),
    }
}

/// Execute tropical operations using the fallback loop-based GEMM.
fn tropical_execute<T: Scalar>(
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
        } => {
            if inputs.len() != 2 {
                return Err(Error::InvalidArgument(
                    "BatchedGemm execute requires 2 input tensors".into(),
                ));
            }
            execute_batched_gemm_fallback(alpha, inputs, beta, output, batch_dims, *m, *n, *k)
        }

        TropicalPlan::Reduce {
            reduced_axes, op, ..
        } => match op {
            ReduceOp::Sum => {
                if inputs.len() != 1 {
                    return Err(Error::InvalidArgument(
                        "Reduce execute requires 1 input tensor".into(),
                    ));
                }
                execute_reduce_sum(alpha, inputs[0], beta, output, reduced_axes)
            }
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
        } => {
            if inputs.len() != 1 {
                return Err(Error::InvalidArgument(
                    "Trace execute requires 1 input tensor".into(),
                ));
            }
            execute_trace(alpha, inputs[0], beta, output, paired_axes, free_axes)
        }

        TropicalPlan::Permute { perm, .. } => {
            if inputs.len() != 1 {
                return Err(Error::InvalidArgument(
                    "Permute execute requires 1 input tensor".into(),
                ));
            }
            execute_permute(alpha, inputs[0], beta, output, perm)
        }

        TropicalPlan::AntiTrace {
            paired_axes,
            free_axes,
            ..
        } => {
            if inputs.len() != 1 {
                return Err(Error::InvalidArgument(
                    "AntiTrace execute requires 1 input tensor".into(),
                ));
            }
            execute_anti_trace(alpha, inputs[0], beta, output, paired_axes, free_axes)
        }

        TropicalPlan::AntiDiag {
            paired_axes,
            free_axes,
            ..
        } => {
            if inputs.len() != 1 {
                return Err(Error::InvalidArgument(
                    "AntiDiag execute requires 1 input tensor".into(),
                ));
            }
            execute_anti_diag(alpha, inputs[0], beta, output, paired_axes, free_axes)
        }

        TropicalPlan::MakeContiguous { .. } => {
            if inputs.len() != 1 {
                return Err(Error::InvalidArgument(
                    "MakeContiguous execute requires 1 input tensor".into(),
                ));
            }
            execute_make_contiguous(alpha, inputs[0], beta, output)
        }
    }
}

fn execute_make_contiguous<T: Scalar>(
    alpha: T,
    input: &StridedView<T>,
    beta: T,
    output: &mut StridedViewMut<T>,
) -> Result<()> {
    if alpha == T::one() && beta == T::zero() {
        strided_perm::copy_into(output, input).map_err(|e| Error::StrideError(e.to_string()))?;
    } else {
        let dims = output.dims().to_vec();
        for_each_index(&dims, |idx| {
            let val = alpha * input.get(idx);
            if beta == T::zero() {
                output.set(idx, val);
            } else {
                output.set(idx, val + beta * output.get(idx));
            }
        });
    }
    Ok(())
}

// ===========================================================================
// impl TensorPrims<MaxPlusAlgebra> for CpuBackend
// ===========================================================================

/// Try to dispatch BatchedGemm to the SIMD-optimized path for a concrete type.
///
/// SAFETY: The TypeId check guarantees T == $concrete_ty before transmuting.
/// All tropical scalar types are #[repr(transparent)] over their inner type,
/// so the transmute is sound.
macro_rules! try_simd_dispatch {
    ($T:ty, $concrete_ty:ty, $inputs:expr, $alpha:expr, $beta:expr,
     $output:expr, $batch_dims:expr, $m:expr, $n:expr, $k:expr) => {
        if std::any::TypeId::of::<$T>() == std::any::TypeId::of::<$concrete_ty>() {
            let a = unsafe {
                &*($inputs[0] as *const StridedView<$T> as *const StridedView<$concrete_ty>)
            };
            let b = unsafe {
                &*($inputs[1] as *const StridedView<$T> as *const StridedView<$concrete_ty>)
            };
            let out = unsafe {
                &mut *($output as *mut StridedViewMut<$T> as *mut StridedViewMut<$concrete_ty>)
            };
            let alpha = unsafe { *(&$alpha as *const $T as *const $concrete_ty) };
            let beta = unsafe { *(&$beta as *const $T as *const $concrete_ty) };
            return execute_batched_gemm_optimized(
                alpha,
                &[a, b],
                beta,
                out,
                $batch_dims,
                $m,
                $n,
                $k,
            );
        }
    };
}

/// Execute tropical operations with SIMD-optimized GEMM for MaxPlus<f64/f32>.
fn tropical_execute_maxplus<T: Scalar>(
    plan: &TropicalPlan<T>,
    alpha: T,
    inputs: &[&StridedView<T>],
    beta: T,
    output: &mut StridedViewMut<T>,
) -> Result<()> {
    if let TropicalPlan::BatchedGemm {
        batch_dims,
        m,
        n,
        k,
        ..
    } = plan
    {
        if inputs.len() != 2 {
            return Err(Error::InvalidArgument(
                "BatchedGemm execute requires 2 input tensors".into(),
            ));
        }
        try_simd_dispatch!(
            T,
            MaxPlus<f64>,
            inputs,
            alpha,
            beta,
            output,
            batch_dims,
            *m,
            *n,
            *k
        );
        try_simd_dispatch!(
            T,
            MaxPlus<f32>,
            inputs,
            alpha,
            beta,
            output,
            batch_dims,
            *m,
            *n,
            *k
        );
    }
    tropical_execute(plan, alpha, inputs, beta, output)
}

/// Execute tropical operations with SIMD-optimized GEMM for MinPlus<f64/f32>.
fn tropical_execute_minplus<T: Scalar>(
    plan: &TropicalPlan<T>,
    alpha: T,
    inputs: &[&StridedView<T>],
    beta: T,
    output: &mut StridedViewMut<T>,
) -> Result<()> {
    if let TropicalPlan::BatchedGemm {
        batch_dims,
        m,
        n,
        k,
        ..
    } = plan
    {
        if inputs.len() != 2 {
            return Err(Error::InvalidArgument(
                "BatchedGemm execute requires 2 input tensors".into(),
            ));
        }
        try_simd_dispatch!(
            T,
            MinPlus<f64>,
            inputs,
            alpha,
            beta,
            output,
            batch_dims,
            *m,
            *n,
            *k
        );
        try_simd_dispatch!(
            T,
            MinPlus<f32>,
            inputs,
            alpha,
            beta,
            output,
            batch_dims,
            *m,
            *n,
            *k
        );
    }
    tropical_execute(plan, alpha, inputs, beta, output)
}

/// Execute tropical operations with SIMD-optimized GEMM for MaxMul<f64/f32>.
fn tropical_execute_maxmul<T: Scalar>(
    plan: &TropicalPlan<T>,
    alpha: T,
    inputs: &[&StridedView<T>],
    beta: T,
    output: &mut StridedViewMut<T>,
) -> Result<()> {
    if let TropicalPlan::BatchedGemm {
        batch_dims,
        m,
        n,
        k,
        ..
    } = plan
    {
        if inputs.len() != 2 {
            return Err(Error::InvalidArgument(
                "BatchedGemm execute requires 2 input tensors".into(),
            ));
        }
        try_simd_dispatch!(
            T,
            MaxMul<f64>,
            inputs,
            alpha,
            beta,
            output,
            batch_dims,
            *m,
            *n,
            *k
        );
        try_simd_dispatch!(
            T,
            MaxMul<f32>,
            inputs,
            alpha,
            beta,
            output,
            batch_dims,
            *m,
            *n,
            *k
        );
    }
    tropical_execute(plan, alpha, inputs, beta, output)
}

impl TensorPrims<MaxPlusAlgebra> for CpuBackend {
    type Plan<T: Scalar> = TropicalPlan<T>;
    type Context = CpuContext;

    fn plan<T: Scalar>(
        _ctx: &mut CpuContext,
        desc: &PrimDescriptor,
        shapes: &[&[usize]],
    ) -> Result<TropicalPlan<T>> {
        tropical_plan(desc, shapes)
    }

    fn execute<T: Scalar>(
        _ctx: &mut CpuContext,
        plan: &TropicalPlan<T>,
        alpha: T,
        inputs: &[&Tensor<T>],
        beta: T,
        output: &mut Tensor<T>,
    ) -> Result<()> {
        let views: Vec<StridedView<T>> = inputs
            .iter()
            .map(|t| tensor_to_view(t))
            .collect::<Result<Vec<_>>>()?;
        let view_refs: Vec<&StridedView<T>> = views.iter().collect();
        let mut out_view = tensor_to_view_mut(output)?;
        tropical_execute_maxplus(plan, alpha, &view_refs, beta, &mut out_view)
    }

    fn has_extension_for<T: Scalar>(_ext: Extension) -> bool {
        false
    }
}

// ===========================================================================
// impl TensorPrims<MinPlusAlgebra> for CpuBackend
// ===========================================================================

impl TensorPrims<MinPlusAlgebra> for CpuBackend {
    type Plan<T: Scalar> = TropicalPlan<T>;
    type Context = CpuContext;

    fn plan<T: Scalar>(
        _ctx: &mut CpuContext,
        desc: &PrimDescriptor,
        shapes: &[&[usize]],
    ) -> Result<TropicalPlan<T>> {
        tropical_plan(desc, shapes)
    }

    fn execute<T: Scalar>(
        _ctx: &mut CpuContext,
        plan: &TropicalPlan<T>,
        alpha: T,
        inputs: &[&Tensor<T>],
        beta: T,
        output: &mut Tensor<T>,
    ) -> Result<()> {
        let views: Vec<StridedView<T>> = inputs
            .iter()
            .map(|t| tensor_to_view(t))
            .collect::<Result<Vec<_>>>()?;
        let view_refs: Vec<&StridedView<T>> = views.iter().collect();
        let mut out_view = tensor_to_view_mut(output)?;
        tropical_execute_minplus(plan, alpha, &view_refs, beta, &mut out_view)
    }

    fn has_extension_for<T: Scalar>(_ext: Extension) -> bool {
        false
    }
}

// ===========================================================================
// impl TensorPrims<MaxMulAlgebra> for CpuBackend
// ===========================================================================

impl TensorPrims<MaxMulAlgebra> for CpuBackend {
    type Plan<T: Scalar> = TropicalPlan<T>;
    type Context = CpuContext;

    fn plan<T: Scalar>(
        _ctx: &mut CpuContext,
        desc: &PrimDescriptor,
        shapes: &[&[usize]],
    ) -> Result<TropicalPlan<T>> {
        tropical_plan(desc, shapes)
    }

    fn execute<T: Scalar>(
        _ctx: &mut CpuContext,
        plan: &TropicalPlan<T>,
        alpha: T,
        inputs: &[&Tensor<T>],
        beta: T,
        output: &mut Tensor<T>,
    ) -> Result<()> {
        let views: Vec<StridedView<T>> = inputs
            .iter()
            .map(|t| tensor_to_view(t))
            .collect::<Result<Vec<_>>>()?;
        let view_refs: Vec<&StridedView<T>> = views.iter().collect();
        let mut out_view = tensor_to_view_mut(output)?;
        tropical_execute_maxmul(plan, alpha, &view_refs, beta, &mut out_view)
    }

    fn has_extension_for<T: Scalar>(_ext: Extension) -> bool {
        false
    }
}
