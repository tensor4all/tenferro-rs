use std::any::TypeId;
use std::marker::PhantomData;

use num_complex::{Complex32, Complex64};
use strided_view::{StridedView, StridedViewMut};
use tenferro_algebra::{Conjugate, Scalar, Standard};
use tenferro_device::{Error, Result};
use tenferro_tensor::Tensor;

use crate::{
    for_each_index, mode_position, unflatten_index, validate_execute_inputs, validate_rank,
    validate_shape_count, validate_shape_eq, Extension, PlanCache, PrimDescriptor, ReduceOp,
    TensorPrims, UnaryOp,
};

/// Convert a CPU tensor to an immutable strided view.
fn tensor_to_view<T: Scalar>(t: &Tensor<T>) -> Result<StridedView<'_, T>> {
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

/// Compute connected components from a list of paired axis positions using union-find.
///
/// Returns `(components, comp_dims)` where:
/// - `components[i]` = sorted list of all axis positions in the i-th component
/// - `comp_dims[i]` = the shared dimension of the i-th component (looked up from `shape`)
fn compute_paired_components(
    paired_axes: &[(usize, usize)],
    shape: &[usize],
) -> (Vec<Vec<usize>>, Vec<usize>) {
    use std::collections::HashMap;

    if paired_axes.is_empty() {
        return (vec![], vec![]);
    }

    // Collect all axes that appear in paired_axes
    let mut all_axes: Vec<usize> = Vec::new();
    for &(ax1, ax2) in paired_axes {
        all_axes.push(ax1);
        all_axes.push(ax2);
    }
    all_axes.sort();
    all_axes.dedup();

    // Union-find: parent map
    let mut parent: HashMap<usize, usize> = all_axes.iter().map(|&ax| (ax, ax)).collect();

    fn find(parent: &mut HashMap<usize, usize>, x: usize) -> usize {
        let p = parent[&x];
        if p != x {
            let root = find(parent, p);
            parent.insert(x, root);
            root
        } else {
            x
        }
    }

    // Union each pair
    for &(ax1, ax2) in paired_axes {
        let r1 = find(&mut parent, ax1);
        let r2 = find(&mut parent, ax2);
        if r1 != r2 {
            // Union: make smaller root the parent (deterministic)
            let (lo, hi) = if r1 < r2 { (r1, r2) } else { (r2, r1) };
            parent.insert(hi, lo);
        }
    }

    // Group by root
    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for &ax in &all_axes {
        let root = find(&mut parent, ax);
        groups.entry(root).or_default().push(ax);
    }

    // Sort components by their minimum axis for determinism
    let mut components: Vec<Vec<usize>> = groups.into_values().collect();
    components.sort_by_key(|c| c[0]);

    let comp_dims: Vec<usize> = components.iter().map(|c| shape[c[0]]).collect();

    (components, comp_dims)
}

/// CPU plan — concrete enum, no type erasure.
///
/// Created by [`CpuBackend::plan`](TensorPrims::plan) and consumed by
/// [`CpuBackend::execute`](TensorPrims::execute).
#[derive(Debug, Clone)]
pub enum CpuPlan<T: Scalar> {
    /// Plan for batched GEMM.
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
    /// Plan for reduction.
    Reduce {
        /// Axes to reduce over (positions in input tensor).
        reduced_axes: Vec<usize>,
        /// Reduction operation.
        op: ReduceOp,
        _marker: PhantomData<T>,
    },
    /// Plan for trace.
    Trace {
        /// Output axis positions mapping.
        free_axes: Vec<usize>,
        /// Connected components of paired axes (union-find groups).
        /// Each inner Vec contains all axis positions in one component.
        components: Vec<Vec<usize>>,
        /// Dimension of each component (all axes in a component share the same dim).
        comp_dims: Vec<usize>,
        _marker: PhantomData<T>,
    },
    /// Plan for permutation.
    Permute {
        /// Permutation mapping (perm[out_axis] = in_axis).
        perm: Vec<usize>,
        _marker: PhantomData<T>,
    },
    /// Plan for anti-trace (AD backward).
    AntiTrace {
        /// Paired axis positions in output tensor.
        paired_axes: Vec<(usize, usize)>,
        /// Input axis positions mapping.
        free_axes: Vec<usize>,
        /// Connected components of paired axes (union-find groups).
        components: Vec<Vec<usize>>,
        /// Dimension of each component.
        comp_dims: Vec<usize>,
        _marker: PhantomData<T>,
    },
    /// Plan for anti-diag (AD backward).
    AntiDiag {
        /// Paired axis positions in output tensor.
        paired_axes: Vec<(usize, usize)>,
        /// Input axis positions mapping.
        free_axes: Vec<usize>,
        /// Connected components of paired axes (union-find groups).
        components: Vec<Vec<usize>>,
        /// Dimension of each component.
        comp_dims: Vec<usize>,
        /// Indices of generative components (no overlap with free axes).
        generative_comps: Vec<usize>,
        _marker: PhantomData<T>,
    },
    /// Plan for element-wise unary operation.
    ElementwiseUnary {
        /// Unary operation.
        op: UnaryOp,
        _marker: PhantomData<T>,
    },
    /// Plan for fused contraction.
    Contract {
        /// Mode labels for input A.
        modes_a: Vec<u32>,
        /// Mode labels for input B.
        modes_b: Vec<u32>,
        /// Mode labels for output C.
        modes_c: Vec<u32>,
        _marker: PhantomData<T>,
    },
    /// Plan for element-wise multiplication (extended op).
    ElementwiseMul { _marker: PhantomData<T> },
    /// Plan for making a tensor contiguous.
    MakeContiguous { _marker: PhantomData<T> },
}

/// CPU execution context.
///
/// Encapsulates CPU-side execution resources, analogous to cuTENSOR's
/// `cutensorHandle_t`. Holds a rayon thread pool and a [`PlanCache`]
/// for plan reuse. Intermediate buffer allocation relies on the global
/// allocator (e.g., mimalloc/jemalloc) rather than a custom buffer pool.
///
/// # Examples
///
/// ```
/// use tenferro_prims::CpuContext;
///
/// let mut ctx = CpuContext::new(4); // 4-thread pool
/// assert_eq!(ctx.num_threads(), 4);
/// ```
pub struct CpuContext {
    pool: rayon::ThreadPool,
    plan_cache: PlanCache,
}

impl CpuContext {
    /// Create a new CPU context with the given number of threads.
    pub fn new(num_threads: usize) -> Self {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .expect("failed to build rayon thread pool");
        Self {
            pool,
            plan_cache: PlanCache::new(),
        }
    }

    /// Returns the number of threads in the pool.
    pub fn num_threads(&self) -> usize {
        self.pool.current_num_threads()
    }

    /// Returns a reference to the underlying rayon thread pool.
    pub fn thread_pool(&self) -> &rayon::ThreadPool {
        &self.pool
    }

    /// Returns a mutable reference to the plan cache.
    pub fn plan_cache_mut(&mut self) -> &mut PlanCache {
        &mut self.plan_cache
    }
}

/// CPU backend using strided-kernel and GEMM.
///
/// Dispatched automatically when tensors reside on
/// [`LogicalMemorySpace::MainMemory`](tenferro_device::LogicalMemorySpace::MainMemory).
/// Implements [`TensorPrims<Standard<T>>`](TensorPrims) for standard arithmetic.
///
/// # Examples
///
/// ```ignore
/// use tenferro_prims::{CpuBackend, CpuContext, TensorPrims, PrimDescriptor};
/// use strided_view::StridedArray;
///
/// let mut ctx = CpuContext::new(4);
/// let desc = PrimDescriptor::Permute {
///     modes_a: vec![0, 1],
///     modes_b: vec![1, 0],
/// };
/// let plan = CpuBackend::plan::<f64>(&mut ctx, &desc, &[&[3, 4], &[4, 3]]).unwrap();
/// let a = StridedArray::<f64>::col_major(&[3, 4]);
/// let mut b = StridedArray::<f64>::col_major(&[4, 3]);
/// CpuBackend::execute(&mut ctx, &plan, 1.0, &[&a.view()], 0.0, &mut b.view_mut()).unwrap();
/// ```
pub struct CpuBackend;

impl CpuBackend {
    /// Materialize a lazily-conjugated tensor.
    ///
    /// If `src.is_conjugated()` is `false`, returns a shallow clone.
    /// If `true`, applies element-wise conjugation via
    /// `ElementwiseUnary(Conj)` and returns a new tensor with
    /// `conjugated = false`.
    ///
    /// This is the equivalent of PyTorch's `torch.resolve_conj()`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_prims::{CpuBackend, CpuContext};
    ///
    /// let a_conj = a.into_conj(); // lazy
    /// let a_resolved = CpuBackend::resolve_conj(&mut ctx, &a_conj);
    /// assert!(!a_resolved.is_conjugated());
    /// ```
    pub fn resolve_conj<T: Scalar + Conjugate>(
        _ctx: &mut CpuContext,
        src: &tenferro_tensor::Tensor<T>,
    ) -> tenferro_tensor::Tensor<T> {
        if !src.is_conjugated() {
            return src.clone();
        }
        // Create a fresh non-conjugated copy with element-wise conjugation applied.
        // For real types (f64, f32), Conjugate::conj() is identity so this is a plain copy.
        // For complex types (Complex64, Complex32), conj() negates the imaginary part.
        let contiguous = src.contiguous(tenferro_tensor::MemoryOrder::ColumnMajor);
        let data = contiguous
            .buffer()
            .as_slice()
            .expect("CPU tensor must have CPU-accessible data");
        let conjugated_data: Vec<T> = data.iter().map(|&v| v.conj()).collect();
        tenferro_tensor::Tensor::from_slice(
            &conjugated_data,
            src.dims(),
            tenferro_tensor::MemoryOrder::ColumnMajor,
        )
        .expect("from_slice should succeed with valid data and dims")
    }

    /// Build a CPU plan from a descriptor and shapes (without cache lookup).
    ///
    /// This is the internal plan construction logic, factored out of
    /// [`TensorPrims::plan`] so that the trait method can wrap it with
    /// cache lookup/insert.
    fn build_plan<T: Scalar>(desc: &PrimDescriptor, shapes: &[&[usize]]) -> Result<CpuPlan<T>> {
        match desc {
            PrimDescriptor::BatchedGemm {
                batch_dims,
                m,
                n,
                k,
            } => {
                // BatchedGemm expects 3 shapes: A, B, C
                validate_shape_count(shapes, 3, "BatchedGemm")?;
                let expected_a: Vec<usize> = batch_dims.iter().copied().chain([*m, *k]).collect();
                let expected_b: Vec<usize> = batch_dims.iter().copied().chain([*k, *n]).collect();
                let expected_c: Vec<usize> = batch_dims.iter().copied().chain([*m, *n]).collect();
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

            PrimDescriptor::Reduce {
                modes_a,
                modes_c,
                op,
            } => {
                // Reduce expects 2 shapes: A (input), C (output)
                validate_shape_count(shapes, 2, "Reduce")?;
                validate_rank(shapes[0], modes_a.len(), "Reduce input A")?;
                validate_rank(shapes[1], modes_c.len(), "Reduce output C")?;
                // reduced_axes = positions in modes_a not present in modes_c
                let reduced_axes: Vec<usize> = modes_a
                    .iter()
                    .enumerate()
                    .filter(|(_, m)| !modes_c.contains(m))
                    .map(|(i, _)| i)
                    .collect();
                Ok(CpuPlan::Reduce {
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
                // Trace expects 2 shapes: A (input), C (output)
                validate_shape_count(shapes, 2, "Trace")?;
                validate_rank(shapes[0], modes_a.len(), "Trace input A")?;
                validate_rank(shapes[1], modes_c.len(), "Trace output C")?;
                let paired_axes: Vec<(usize, usize)> = paired
                    .iter()
                    .map(|(m1, m2)| {
                        Ok((mode_position(modes_a, *m1)?, mode_position(modes_a, *m2)?))
                    })
                    .collect::<Result<_>>()?;
                // Validate that paired axes have equal dimensions
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
                    .map(|m| mode_position(modes_a, *m))
                    .collect::<Result<_>>()?;
                let (components, comp_dims) = compute_paired_components(&paired_axes, shapes[0]);
                Ok(CpuPlan::Trace {
                    free_axes,
                    components,
                    comp_dims,
                    _marker: PhantomData,
                })
            }

            PrimDescriptor::Permute { modes_a, modes_b } => {
                // Permute expects 2 shapes: A (input), B (output)
                validate_shape_count(shapes, 2, "Permute")?;
                validate_rank(shapes[0], modes_a.len(), "Permute input A")?;
                validate_rank(shapes[1], modes_b.len(), "Permute output B")?;
                // perm[out_axis] = in_axis
                let perm: Vec<usize> = modes_b
                    .iter()
                    .map(|m| mode_position(modes_a, *m))
                    .collect::<Result<_>>()?;
                Ok(CpuPlan::Permute {
                    perm,
                    _marker: PhantomData,
                })
            }

            PrimDescriptor::AntiTrace {
                modes_a,
                modes_c,
                paired,
            } => {
                // AntiTrace expects 2 shapes: A (input), C (output)
                validate_shape_count(shapes, 2, "AntiTrace")?;
                validate_rank(shapes[0], modes_a.len(), "AntiTrace input A")?;
                validate_rank(shapes[1], modes_c.len(), "AntiTrace output C")?;
                let paired_axes: Vec<(usize, usize)> = paired
                    .iter()
                    .map(|(m1, m2)| {
                        Ok((mode_position(modes_c, *m1)?, mode_position(modes_c, *m2)?))
                    })
                    .collect::<Result<_>>()?;
                // Validate that paired axes in output have equal dimensions
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
                    .map(|m| mode_position(modes_c, *m))
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

            PrimDescriptor::AntiDiag {
                modes_a,
                modes_c,
                paired,
            } => {
                // AntiDiag expects 2 shapes: A (input), C (output)
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
                    .map(|m| mode_position(modes_c, *m))
                    .collect::<Result<_>>()?;
                let (components, comp_dims) = compute_paired_components(&paired_axes, shapes[1]);
                // Generative components: those whose axes have no overlap with free_axes
                let free_ax_set: std::collections::HashSet<usize> =
                    free_axes.iter().copied().collect();
                let generative_comps: Vec<usize> = components
                    .iter()
                    .enumerate()
                    .filter(|(_, comp)| comp.iter().all(|ax| !free_ax_set.contains(ax)))
                    .map(|(i, _)| i)
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

            PrimDescriptor::ElementwiseUnary { op } => {
                // ElementwiseUnary expects 2 shapes: A (input), C (output)
                validate_shape_count(shapes, 2, "ElementwiseUnary")?;
                validate_shape_eq(shapes[1], shapes[0], "ElementwiseUnary output")?;
                Ok(CpuPlan::ElementwiseUnary {
                    op: *op,
                    _marker: PhantomData,
                })
            }

            PrimDescriptor::Contract {
                modes_a,
                modes_b,
                modes_c,
            } => {
                // Contract expects 3 shapes: A, B, C
                validate_shape_count(shapes, 3, "Contract")?;
                validate_rank(shapes[0], modes_a.len(), "Contract input A")?;
                validate_rank(shapes[1], modes_b.len(), "Contract input B")?;
                validate_rank(shapes[2], modes_c.len(), "Contract output C")?;
                // Validate contracted dimensions match between A and B
                for &mode in modes_a.iter() {
                    if let Some(b_pos) = modes_b.iter().position(|&m| m == mode) {
                        let a_pos = modes_a.iter().position(|&m| m == mode).unwrap();
                        if shapes[0][a_pos] != shapes[1][b_pos] {
                            return Err(Error::InvalidArgument(format!(
                                "Contract mode {mode} has mismatched dimensions: A={} vs B={}",
                                shapes[0][a_pos], shapes[1][b_pos]
                            )));
                        }
                    }
                }
                Ok(CpuPlan::Contract {
                    modes_a: modes_a.clone(),
                    modes_b: modes_b.clone(),
                    modes_c: modes_c.clone(),
                    _marker: PhantomData,
                })
            }

            PrimDescriptor::ElementwiseMul => {
                // ElementwiseMul expects 3 shapes: A, B, C
                validate_shape_count(shapes, 3, "ElementwiseMul")?;
                validate_shape_eq(shapes[1], shapes[0], "ElementwiseMul input B")?;
                validate_shape_eq(shapes[2], shapes[0], "ElementwiseMul output C")?;
                Ok(CpuPlan::ElementwiseMul {
                    _marker: PhantomData,
                })
            }

            PrimDescriptor::MakeContiguous => {
                // MakeContiguous expects 2 shapes: A (input), C (output)
                validate_shape_count(shapes, 2, "MakeContiguous")?;
                validate_shape_eq(shapes[1], shapes[0], "MakeContiguous output")?;
                Ok(CpuPlan::MakeContiguous {
                    _marker: PhantomData,
                })
            }
        }
    }
}

// ===========================================================================
// CPU execute helpers for each operation
// ===========================================================================

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
        // Fast path: use strided-perm HPTT-based copy
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

fn execute_batched_gemm_naive<T: Scalar>(
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

fn execute_batched_gemm_f64(
    alpha: f64,
    inputs: &[&StridedView<f64>],
    beta: f64,
    output: &mut StridedViewMut<f64>,
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

    let mut a_mat = vec![0.0_f64; m * k];
    let mut b_mat = vec![0.0_f64; k * n];
    let mut c_mat = vec![0.0_f64; m * n];
    let mut idx = Vec::with_capacity(batch_dims.len() + 2);

    for batch_flat in 0..batch_size {
        let batch_idx = unflatten_index(batch_flat, batch_dims);

        for kk in 0..k {
            for i in 0..m {
                idx.clear();
                idx.extend_from_slice(&batch_idx);
                idx.push(i);
                idx.push(kk);
                a_mat[i + kk * m] = a.get(&idx);
            }
        }
        for j in 0..n {
            for kk in 0..k {
                idx.clear();
                idx.extend_from_slice(&batch_idx);
                idx.push(kk);
                idx.push(j);
                b_mat[kk + j * k] = b.get(&idx);
            }
        }
        for j in 0..n {
            for i in 0..m {
                idx.clear();
                idx.extend_from_slice(&batch_idx);
                idx.push(i);
                idx.push(j);
                c_mat[i + j * m] = output.get(&idx);
            }
        }

        gemm_f64(alpha, &a_mat, &b_mat, beta, &mut c_mat, m, n, k)?;

        for j in 0..n {
            for i in 0..m {
                idx.clear();
                idx.extend_from_slice(&batch_idx);
                idx.push(i);
                idx.push(j);
                output.set(&idx, c_mat[i + j * m]);
            }
        }
    }

    Ok(())
}

fn execute_batched_gemm_f32(
    alpha: f32,
    inputs: &[&StridedView<f32>],
    beta: f32,
    output: &mut StridedViewMut<f32>,
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

    let mut a_mat = vec![0.0_f32; m * k];
    let mut b_mat = vec![0.0_f32; k * n];
    let mut c_mat = vec![0.0_f32; m * n];
    let mut idx = Vec::with_capacity(batch_dims.len() + 2);

    for batch_flat in 0..batch_size {
        let batch_idx = unflatten_index(batch_flat, batch_dims);

        for kk in 0..k {
            for i in 0..m {
                idx.clear();
                idx.extend_from_slice(&batch_idx);
                idx.push(i);
                idx.push(kk);
                a_mat[i + kk * m] = a.get(&idx);
            }
        }
        for j in 0..n {
            for kk in 0..k {
                idx.clear();
                idx.extend_from_slice(&batch_idx);
                idx.push(kk);
                idx.push(j);
                b_mat[kk + j * k] = b.get(&idx);
            }
        }
        for j in 0..n {
            for i in 0..m {
                idx.clear();
                idx.extend_from_slice(&batch_idx);
                idx.push(i);
                idx.push(j);
                c_mat[i + j * m] = output.get(&idx);
            }
        }

        gemm_f32(alpha, &a_mat, &b_mat, beta, &mut c_mat, m, n, k)?;

        for j in 0..n {
            for i in 0..m {
                idx.clear();
                idx.extend_from_slice(&batch_idx);
                idx.push(i);
                idx.push(j);
                output.set(&idx, c_mat[i + j * m]);
            }
        }
    }

    Ok(())
}

#[cfg(feature = "gemm-faer")]
fn gemm_f64(
    alpha: f64,
    a: &[f64],
    b: &[f64],
    beta: f64,
    c: &mut [f64],
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let a_mat = faer::mat::from_column_major_slice(a, m, k);
    let b_mat = faer::mat::from_column_major_slice(b, k, n);
    let prod = &a_mat * &b_mat;
    for j in 0..n {
        for i in 0..m {
            let p = prod[(i, j)];
            c[i + j * m] = alpha * p + beta * c[i + j * m];
        }
    }
    Ok(())
}

#[cfg(all(not(feature = "gemm-faer"), feature = "gemm-openblas"))]
fn gemm_f64(
    alpha: f64,
    a: &[f64],
    b: &[f64],
    beta: f64,
    c: &mut [f64],
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let m_i32 = i32::try_from(m).map_err(|_| Error::InvalidArgument("m too large".into()))?;
    let n_i32 = i32::try_from(n).map_err(|_| Error::InvalidArgument("n too large".into()))?;
    let k_i32 = i32::try_from(k).map_err(|_| Error::InvalidArgument("k too large".into()))?;
    unsafe {
        cblas_sys::cblas_dgemm(
            cblas_sys::CBLAS_LAYOUT::CblasColMajor,
            cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
            cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
            m_i32,
            n_i32,
            k_i32,
            alpha,
            a.as_ptr(),
            m_i32,
            b.as_ptr(),
            k_i32,
            beta,
            c.as_mut_ptr(),
            m_i32,
        );
    }
    Ok(())
}

#[cfg(all(not(feature = "gemm-faer"), not(feature = "gemm-openblas")))]
fn gemm_f64(
    alpha: f64,
    a: &[f64],
    b: &[f64],
    beta: f64,
    c: &mut [f64],
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    for j in 0..n {
        for i in 0..m {
            let mut sum = 0.0_f64;
            for p in 0..k {
                sum += a[i + p * m] * b[p + j * k];
            }
            c[i + j * m] = alpha * sum + beta * c[i + j * m];
        }
    }
    Ok(())
}

#[cfg(feature = "gemm-faer")]
fn gemm_f32(
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let a_mat = faer::mat::from_column_major_slice(a, m, k);
    let b_mat = faer::mat::from_column_major_slice(b, k, n);
    let prod = &a_mat * &b_mat;
    for j in 0..n {
        for i in 0..m {
            let p = prod[(i, j)];
            c[i + j * m] = alpha * p + beta * c[i + j * m];
        }
    }
    Ok(())
}

#[cfg(all(not(feature = "gemm-faer"), feature = "gemm-openblas"))]
fn gemm_f32(
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let m_i32 = i32::try_from(m).map_err(|_| Error::InvalidArgument("m too large".into()))?;
    let n_i32 = i32::try_from(n).map_err(|_| Error::InvalidArgument("n too large".into()))?;
    let k_i32 = i32::try_from(k).map_err(|_| Error::InvalidArgument("k too large".into()))?;
    unsafe {
        cblas_sys::cblas_sgemm(
            cblas_sys::CBLAS_LAYOUT::CblasColMajor,
            cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
            cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
            m_i32,
            n_i32,
            k_i32,
            alpha,
            a.as_ptr(),
            m_i32,
            b.as_ptr(),
            k_i32,
            beta,
            c.as_mut_ptr(),
            m_i32,
        );
    }
    Ok(())
}

#[cfg(all(not(feature = "gemm-faer"), not(feature = "gemm-openblas")))]
fn gemm_f32(
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    for j in 0..n {
        for i in 0..m {
            let mut sum = 0.0_f32;
            for p in 0..k {
                sum += a[i + p * m] * b[p + j * k];
            }
            c[i + j * m] = alpha * sum + beta * c[i + j * m];
        }
    }
    Ok(())
}

fn execute_batched_gemm<T: Scalar + 'static>(
    alpha: T,
    inputs: &[&StridedView<T>],
    beta: T,
    output: &mut StridedViewMut<T>,
    batch_dims: &[usize],
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let tid = TypeId::of::<T>();

    if tid == TypeId::of::<f64>() {
        let a = unsafe { &*(inputs[0] as *const StridedView<T> as *const StridedView<f64>) };
        let b = unsafe { &*(inputs[1] as *const StridedView<T> as *const StridedView<f64>) };
        let out = unsafe { &mut *(output as *mut StridedViewMut<T> as *mut StridedViewMut<f64>) };
        let alpha = unsafe { *(&alpha as *const T as *const f64) };
        let beta = unsafe { *(&beta as *const T as *const f64) };
        return execute_batched_gemm_f64(alpha, &[a, b], beta, out, batch_dims, m, n, k);
    }

    if tid == TypeId::of::<f32>() {
        let a = unsafe { &*(inputs[0] as *const StridedView<T> as *const StridedView<f32>) };
        let b = unsafe { &*(inputs[1] as *const StridedView<T> as *const StridedView<f32>) };
        let out = unsafe { &mut *(output as *mut StridedViewMut<T> as *mut StridedViewMut<f32>) };
        let alpha = unsafe { *(&alpha as *const T as *const f32) };
        let beta = unsafe { *(&beta as *const T as *const f32) };
        return execute_batched_gemm_f32(alpha, &[a, b], beta, out, batch_dims, m, n, k);
    }

    execute_batched_gemm_naive(alpha, inputs, beta, output, batch_dims, m, n, k)
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
        let mut sum = T::zero();
        for red_flat in 0..reduced_total {
            let red_idx = unflatten_index(red_flat, &reduced_dims);
            // Build full input index by interleaving free and reduced
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
    components: &[Vec<usize>],
    comp_dims: &[usize],
    free_axes: &[usize],
) -> Result<()> {
    let in_dims = input.dims().to_vec();
    let out_dims = output.dims().to_vec();
    let n_comps = comp_dims.len();

    for_each_index(&out_dims, |out_idx| {
        let mut sum = T::zero();
        // Odometer over component dimensions (Cartesian product)
        let mut comp_idx = vec![0usize; n_comps];
        loop {
            let mut in_idx = vec![0; in_dims.len()];
            for (out_pos, &in_ax) in free_axes.iter().enumerate() {
                in_idx[in_ax] = out_idx[out_pos];
            }
            for (t, comp) in components.iter().enumerate() {
                for &ax in comp {
                    in_idx[ax] = comp_idx[t];
                }
            }
            sum = sum + input.get(&in_idx);

            // Increment odometer
            let mut carry = true;
            for t in 0..n_comps {
                if carry {
                    comp_idx[t] += 1;
                    if comp_idx[t] < comp_dims[t] {
                        carry = false;
                    } else {
                        comp_idx[t] = 0;
                    }
                }
            }
            if carry {
                break;
            }
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

fn execute_anti_trace<T: Scalar>(
    alpha: T,
    input: &StridedView<T>,
    beta: T,
    output: &mut StridedViewMut<T>,
    components: &[Vec<usize>],
    comp_dims: &[usize],
    free_axes: &[usize],
) -> Result<()> {
    // AntiTrace: C = alpha * antitrace(A) + beta * C
    // First scale output by beta (since diagonal positions may be written multiple times)
    scale_output(output, beta);

    let in_dims = input.dims().to_vec();
    let out_dims = output.dims().to_vec();
    let n_comps = comp_dims.len();

    // For each input element, scatter to all diagonal positions in output
    // using Cartesian product over component dimensions.
    for_each_index(&in_dims, |in_idx| {
        let val = alpha * input.get(in_idx);
        // Odometer over component dimensions (Cartesian product)
        let mut comp_idx = vec![0usize; n_comps];
        loop {
            let mut out_idx = vec![0; out_dims.len()];
            for (in_pos, &out_ax) in free_axes.iter().enumerate() {
                out_idx[out_ax] = in_idx[in_pos];
            }
            for (t, comp) in components.iter().enumerate() {
                for &ax in comp {
                    out_idx[ax] = comp_idx[t];
                }
            }
            let old = output.get(&out_idx);
            output.set(&out_idx, old + val);

            // Increment odometer
            let mut carry = true;
            for t in 0..n_comps {
                if carry {
                    comp_idx[t] += 1;
                    if comp_idx[t] < comp_dims[t] {
                        carry = false;
                    } else {
                        comp_idx[t] = 0;
                    }
                }
            }
            if carry {
                break;
            }
        }
    });
    Ok(())
}

fn execute_anti_diag<T: Scalar>(
    alpha: T,
    input: &StridedView<T>,
    beta: T,
    output: &mut StridedViewMut<T>,
    components: &[Vec<usize>],
    comp_dims: &[usize],
    free_axes: &[usize],
    generative_comps: &[usize],
) -> Result<()> {
    // AntiDiag: write input values to diagonal positions in output.
    // Anchored components: at least one axis overlaps with free_axes, constraint propagated.
    // Generative components: no axis overlaps with free_axes, need own loop.
    scale_output(output, beta);

    let in_dims = input.dims().to_vec();
    let out_dims = output.dims().to_vec();

    let gen_dims: Vec<usize> = generative_comps.iter().map(|&c| comp_dims[c]).collect();

    for_each_index(&in_dims, |in_idx| {
        let val = alpha * input.get(in_idx);
        // Odometer over generative component dimensions
        let mut gen_idx = vec![0usize; generative_comps.len()];
        loop {
            let mut out_idx = vec![0; out_dims.len()];
            // Set free axes from input
            for (in_pos, &out_ax) in free_axes.iter().enumerate() {
                out_idx[out_ax] = in_idx[in_pos];
            }
            // Set component axes
            for (t, comp) in components.iter().enumerate() {
                if let Some(gi) = generative_comps.iter().position(|&c| c == t) {
                    // Generative: use gen_idx
                    for &ax in comp {
                        out_idx[ax] = gen_idx[gi];
                    }
                } else {
                    // Anchored: propagate from the first axis (already set by free_axes)
                    let anchor_val = out_idx[comp[0]];
                    for &ax in &comp[1..] {
                        out_idx[ax] = anchor_val;
                    }
                }
            }
            let old = output.get(&out_idx);
            output.set(&out_idx, old + val);

            if gen_dims.is_empty() {
                break;
            }
            // Increment odometer for generative components
            let mut carry = true;
            for g in 0..gen_dims.len() {
                if carry {
                    gen_idx[g] += 1;
                    if gen_idx[g] < gen_dims[g] {
                        carry = false;
                    } else {
                        gen_idx[g] = 0;
                    }
                }
            }
            if carry {
                break;
            }
        }
    });
    Ok(())
}

fn execute_elementwise_mul<T: Scalar>(
    alpha: T,
    inputs: &[&StridedView<T>],
    beta: T,
    output: &mut StridedViewMut<T>,
) -> Result<()> {
    let a = inputs[0];
    let b = inputs[1];
    let dims = output.dims().to_vec();
    for_each_index(&dims, |idx| {
        let val = alpha * (a.get(idx) * b.get(idx));
        if beta == T::zero() {
            output.set(idx, val);
        } else {
            output.set(idx, val + beta * output.get(idx));
        }
    });
    Ok(())
}

/// Apply a unary function element-wise: C = alpha * f(A) + beta * C.
fn execute_unary_map<T: Scalar>(
    alpha: T,
    input: &StridedView<T>,
    beta: T,
    output: &mut StridedViewMut<T>,
    f: impl Fn(T) -> T,
) -> Result<()> {
    let dims = output.dims().to_vec();
    for_each_index(&dims, |idx| {
        let val = alpha * f(input.get(idx));
        if beta == T::zero() {
            output.set(idx, val);
        } else {
            output.set(idx, val + beta * output.get(idx));
        }
    });
    Ok(())
}

/// Execute element-wise unary operation with type-based dispatch.
///
/// Since `Scalar` does not provide `Neg`, `Div`, or floating-point ops,
/// we dispatch to concrete type implementations (f32, f64, Complex32, Complex64)
/// at runtime using `TypeId`. This keeps the `TensorPrims` trait generic while
/// supporting all standard unary operations on the CPU backend.
fn execute_elementwise_unary<T: Scalar>(
    alpha: T,
    input: &StridedView<T>,
    beta: T,
    output: &mut StridedViewMut<T>,
    op: &UnaryOp,
) -> Result<()> {
    match op {
        UnaryOp::Conj => {
            let tid = TypeId::of::<T>();
            if tid == TypeId::of::<f64>() || tid == TypeId::of::<f32>() {
                // Real types: conjugation is identity
                execute_make_contiguous(alpha, input, beta, output)
            } else if tid == TypeId::of::<Complex64>() {
                execute_unary_map(alpha, input, beta, output, |v| {
                    let x = unsafe { *(&v as *const T as *const Complex64) };
                    let r = x.conj();
                    unsafe { *(&r as *const Complex64 as *const T) }
                })
            } else if tid == TypeId::of::<Complex32>() {
                execute_unary_map(alpha, input, beta, output, |v| {
                    let x = unsafe { *(&v as *const T as *const Complex32) };
                    let r = x.conj();
                    unsafe { *(&r as *const Complex32 as *const T) }
                })
            } else {
                Err(Error::InvalidArgument(format!(
                    "Conj not supported for this scalar type"
                )))
            }
        }
        UnaryOp::Negate => {
            let tid = TypeId::of::<T>();
            if tid == TypeId::of::<f64>() {
                execute_unary_map(alpha, input, beta, output, |v| {
                    // SAFETY: T is f64; transmute is safe because we checked TypeId
                    let x = unsafe { *(&v as *const T as *const f64) };
                    let r = -x;
                    unsafe { *(&r as *const f64 as *const T) }
                })
            } else if tid == TypeId::of::<f32>() {
                execute_unary_map(alpha, input, beta, output, |v| {
                    let x = unsafe { *(&v as *const T as *const f32) };
                    let r = -x;
                    unsafe { *(&r as *const f32 as *const T) }
                })
            } else if tid == TypeId::of::<Complex64>() {
                execute_unary_map(alpha, input, beta, output, |v| {
                    let x = unsafe { *(&v as *const T as *const Complex64) };
                    let r = -x;
                    unsafe { *(&r as *const Complex64 as *const T) }
                })
            } else if tid == TypeId::of::<Complex32>() {
                execute_unary_map(alpha, input, beta, output, |v| {
                    let x = unsafe { *(&v as *const T as *const Complex32) };
                    let r = -x;
                    unsafe { *(&r as *const Complex32 as *const T) }
                })
            } else {
                Err(Error::InvalidArgument(format!(
                    "Negate not supported for this scalar type"
                )))
            }
        }
        UnaryOp::Reciprocal => {
            let tid = TypeId::of::<T>();
            if tid == TypeId::of::<f64>() {
                execute_unary_map(alpha, input, beta, output, |v| {
                    let x = unsafe { *(&v as *const T as *const f64) };
                    let r = 1.0_f64 / x;
                    unsafe { *(&r as *const f64 as *const T) }
                })
            } else if tid == TypeId::of::<f32>() {
                execute_unary_map(alpha, input, beta, output, |v| {
                    let x = unsafe { *(&v as *const T as *const f32) };
                    let r = 1.0_f32 / x;
                    unsafe { *(&r as *const f32 as *const T) }
                })
            } else if tid == TypeId::of::<Complex64>() {
                execute_unary_map(alpha, input, beta, output, |v| {
                    let x = unsafe { *(&v as *const T as *const Complex64) };
                    let r = Complex64::new(1.0, 0.0) / x;
                    unsafe { *(&r as *const Complex64 as *const T) }
                })
            } else if tid == TypeId::of::<Complex32>() {
                execute_unary_map(alpha, input, beta, output, |v| {
                    let x = unsafe { *(&v as *const T as *const Complex32) };
                    let r = Complex32::new(1.0, 0.0) / x;
                    unsafe { *(&r as *const Complex32 as *const T) }
                })
            } else {
                Err(Error::InvalidArgument(format!(
                    "Reciprocal not supported for this scalar type"
                )))
            }
        }
        UnaryOp::Abs => {
            let tid = TypeId::of::<T>();
            if tid == TypeId::of::<f64>() {
                execute_unary_map(alpha, input, beta, output, |v| {
                    let x = unsafe { *(&v as *const T as *const f64) };
                    let r = x.abs();
                    unsafe { *(&r as *const f64 as *const T) }
                })
            } else if tid == TypeId::of::<f32>() {
                execute_unary_map(alpha, input, beta, output, |v| {
                    let x = unsafe { *(&v as *const T as *const f32) };
                    let r = x.abs();
                    unsafe { *(&r as *const f32 as *const T) }
                })
            } else if tid == TypeId::of::<Complex64>() {
                // For complex, abs returns the modulus as a real number.
                // But since T is Complex64, we return it as Complex64 with zero imaginary part.
                execute_unary_map(alpha, input, beta, output, |v| {
                    let x = unsafe { *(&v as *const T as *const Complex64) };
                    let r = Complex64::new(x.norm(), 0.0);
                    unsafe { *(&r as *const Complex64 as *const T) }
                })
            } else if tid == TypeId::of::<Complex32>() {
                execute_unary_map(alpha, input, beta, output, |v| {
                    let x = unsafe { *(&v as *const T as *const Complex32) };
                    let r = Complex32::new(x.norm(), 0.0);
                    unsafe { *(&r as *const Complex32 as *const T) }
                })
            } else {
                Err(Error::InvalidArgument(format!(
                    "Abs not supported for this scalar type"
                )))
            }
        }
        UnaryOp::Sqrt => {
            let tid = TypeId::of::<T>();
            if tid == TypeId::of::<f64>() {
                execute_unary_map(alpha, input, beta, output, |v| {
                    let x = unsafe { *(&v as *const T as *const f64) };
                    let r = x.sqrt();
                    unsafe { *(&r as *const f64 as *const T) }
                })
            } else if tid == TypeId::of::<f32>() {
                execute_unary_map(alpha, input, beta, output, |v| {
                    let x = unsafe { *(&v as *const T as *const f32) };
                    let r = x.sqrt();
                    unsafe { *(&r as *const f32 as *const T) }
                })
            } else if tid == TypeId::of::<Complex64>() {
                execute_unary_map(alpha, input, beta, output, |v| {
                    let x = unsafe { *(&v as *const T as *const Complex64) };
                    let r = x.sqrt();
                    unsafe { *(&r as *const Complex64 as *const T) }
                })
            } else if tid == TypeId::of::<Complex32>() {
                execute_unary_map(alpha, input, beta, output, |v| {
                    let x = unsafe { *(&v as *const T as *const Complex32) };
                    let r = x.sqrt();
                    unsafe { *(&r as *const Complex32 as *const T) }
                })
            } else {
                Err(Error::InvalidArgument(format!(
                    "Sqrt not supported for this scalar type"
                )))
            }
        }
    }
}

fn execute_contract<T: Scalar>(
    alpha: T,
    inputs: &[&StridedView<T>],
    beta: T,
    output: &mut StridedViewMut<T>,
    modes_a: &[u32],
    modes_b: &[u32],
    modes_c: &[u32],
) -> Result<()> {
    let a = inputs[0];
    let b = inputs[1];

    // Determine contracted modes: in both A and B but not in C
    let contracted_modes: Vec<u32> = modes_a
        .iter()
        .filter(|m| modes_b.contains(m) && !modes_c.contains(m))
        .copied()
        .collect();
    let contracted_dims: Vec<usize> = contracted_modes
        .iter()
        .map(|&m| {
            let a_pos = modes_a.iter().position(|&mm| mm == m).unwrap();
            a.dims()[a_pos]
        })
        .collect();
    let contracted_total: usize = if contracted_dims.is_empty() {
        1
    } else {
        contracted_dims.iter().product()
    };

    let out_dims = output.dims().to_vec();

    for_each_index(&out_dims, |c_idx| {
        let mut sum = T::zero();
        for k_flat in 0..contracted_total {
            let k_idx = unflatten_index(k_flat, &contracted_dims);

            // Build A indices
            let mut a_idx = vec![0; modes_a.len()];
            for (ax, &mode) in modes_a.iter().enumerate() {
                if let Some(c_pos) = modes_c.iter().position(|&m| m == mode) {
                    a_idx[ax] = c_idx[c_pos];
                } else if let Some(k_pos) = contracted_modes.iter().position(|&m| m == mode) {
                    a_idx[ax] = k_idx[k_pos];
                }
            }

            // Build B indices
            let mut b_idx = vec![0; modes_b.len()];
            for (ax, &mode) in modes_b.iter().enumerate() {
                if let Some(c_pos) = modes_c.iter().position(|&m| m == mode) {
                    b_idx[ax] = c_idx[c_pos];
                } else if let Some(k_pos) = contracted_modes.iter().position(|&m| m == mode) {
                    b_idx[ax] = k_idx[k_pos];
                }
            }

            sum = sum + a.get(&a_idx) * b.get(&b_idx);
        }
        let old = if beta == T::zero() {
            T::zero()
        } else {
            beta * output.get(c_idx)
        };
        output.set(c_idx, alpha * sum + old);
    });
    Ok(())
}

// ===========================================================================
// CPU backend TensorPrims implementation
// ===========================================================================

impl<S: Scalar> TensorPrims<Standard<S>> for CpuBackend {
    type Plan = CpuPlan<S>;
    type Context = CpuContext;

    fn plan(
        ctx: &mut CpuContext,
        desc: &PrimDescriptor,
        shapes: &[&[usize]],
    ) -> Result<CpuPlan<S>> {
        // Check cache first
        if let Some(cached) = ctx.plan_cache.get::<CpuPlan<S>>(desc, shapes) {
            return Ok(cached);
        }

        let plan = Self::build_plan::<S>(desc, shapes)?;

        // Store in cache for future reuse
        ctx.plan_cache.insert(desc, shapes, plan.clone());

        Ok(plan)
    }

    fn execute(
        _ctx: &mut CpuContext,
        plan: &CpuPlan<S>,
        alpha: S,
        inputs: &[&Tensor<S>],
        beta: S,
        output: &mut Tensor<S>,
    ) -> Result<()> {
        // Convert Tensor inputs to StridedView for internal dispatch
        let views: Vec<StridedView<S>> = inputs
            .iter()
            .map(|t| tensor_to_view(t))
            .collect::<Result<Vec<_>>>()?;
        let view_refs: Vec<&StridedView<S>> = views.iter().collect();
        let mut out_view = tensor_to_view_mut(output)?;

        match plan {
            CpuPlan::Permute { perm, .. } => {
                validate_execute_inputs(inputs, 1, "Permute")?;
                execute_permute(alpha, view_refs[0], beta, &mut out_view, perm)
            }

            CpuPlan::MakeContiguous { .. } => {
                validate_execute_inputs(inputs, 1, "MakeContiguous")?;
                execute_make_contiguous(alpha, view_refs[0], beta, &mut out_view)
            }

            CpuPlan::BatchedGemm {
                batch_dims,
                m,
                n,
                k,
                ..
            } => {
                validate_execute_inputs(inputs, 2, "BatchedGemm")?;
                execute_batched_gemm(
                    alpha,
                    &view_refs,
                    beta,
                    &mut out_view,
                    batch_dims,
                    *m,
                    *n,
                    *k,
                )
            }

            CpuPlan::Reduce {
                reduced_axes, op, ..
            } => {
                validate_execute_inputs(inputs, 1, "Reduce")?;
                match op {
                    ReduceOp::Sum => {
                        execute_reduce_sum(alpha, view_refs[0], beta, &mut out_view, reduced_axes)
                    }
                    ReduceOp::Max | ReduceOp::Min => Err(Error::InvalidArgument(
                        "Max/Min reduction requires PartialOrd, not available via Scalar".into(),
                    )),
                }
            }

            CpuPlan::Trace {
                components,
                comp_dims,
                free_axes,
                ..
            } => {
                validate_execute_inputs(inputs, 1, "Trace")?;
                execute_trace(
                    alpha,
                    view_refs[0],
                    beta,
                    &mut out_view,
                    components,
                    comp_dims,
                    free_axes,
                )
            }

            CpuPlan::AntiTrace {
                free_axes,
                components,
                comp_dims,
                ..
            } => {
                validate_execute_inputs(inputs, 1, "AntiTrace")?;
                execute_anti_trace(
                    alpha,
                    view_refs[0],
                    beta,
                    &mut out_view,
                    components,
                    comp_dims,
                    free_axes,
                )
            }

            CpuPlan::AntiDiag {
                free_axes,
                components,
                comp_dims,
                generative_comps,
                ..
            } => {
                validate_execute_inputs(inputs, 1, "AntiDiag")?;
                execute_anti_diag(
                    alpha,
                    view_refs[0],
                    beta,
                    &mut out_view,
                    components,
                    comp_dims,
                    free_axes,
                    generative_comps,
                )
            }

            CpuPlan::ElementwiseUnary { op, .. } => {
                validate_execute_inputs(inputs, 1, "ElementwiseUnary")?;
                execute_elementwise_unary(alpha, view_refs[0], beta, &mut out_view, op)
            }

            CpuPlan::ElementwiseMul { .. } => {
                validate_execute_inputs(inputs, 2, "ElementwiseMul")?;
                execute_elementwise_mul(alpha, &view_refs, beta, &mut out_view)
            }

            CpuPlan::Contract {
                modes_a,
                modes_b,
                modes_c,
                ..
            } => {
                validate_execute_inputs(inputs, 2, "Contract")?;
                execute_contract(
                    alpha,
                    &view_refs,
                    beta,
                    &mut out_view,
                    modes_a,
                    modes_b,
                    modes_c,
                )
            }
        }
    }

    fn has_extension_for(_ext: Extension) -> bool {
        // CPU backend supports both Contract and ElementwiseMul
        true
    }
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
    // If beta == 1, output is unchanged (identity scaling).
}
