use std::marker::PhantomData;

use tenferro_algebra::Scalar;

use crate::SemiringBinaryOp;

/// Compute connected components from a list of paired axis positions using union-find.
///
/// Returns `(components, comp_dims)` where:
/// - `components[i]` = sorted list of all axis positions in the i-th component
/// - `comp_dims[i]` = the shared dimension of the i-th component (looked up from `shape`)
pub(super) fn compute_paired_components(
    paired_axes: &[(usize, usize)],
    shape: &[usize],
) -> (Vec<Vec<usize>>, Vec<usize>) {
    use std::collections::HashMap;

    if paired_axes.is_empty() {
        return (vec![], vec![]);
    }

    let mut all_axes: Vec<usize> = Vec::new();
    for &(ax1, ax2) in paired_axes {
        all_axes.push(ax1);
        all_axes.push(ax2);
    }
    all_axes.sort();
    all_axes.dedup();

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

    for &(ax1, ax2) in paired_axes {
        let r1 = find(&mut parent, ax1);
        let r2 = find(&mut parent, ax2);
        if r1 != r2 {
            let (lo, hi) = if r1 < r2 { (r1, r2) } else { (r2, r1) };
            parent.insert(hi, lo);
        }
    }

    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for &ax in &all_axes {
        let root = find(&mut parent, ax);
        groups.entry(root).or_default().push(ax);
    }

    let mut components: Vec<Vec<usize>> = groups.into_values().collect();
    components.sort_by_key(|c| c[0]);

    let comp_dims: Vec<usize> = components.iter().map(|c| shape[c[0]]).collect();

    (components, comp_dims)
}

/// Pre-computed mode analysis for Contract GEMM fast path.
#[derive(Debug, Clone)]
pub(super) struct ContractGemmSpec {
    /// Target mode order for A: [batch, m, k]
    pub(super) a_target: Vec<u32>,
    /// Target mode order for B: [batch, k, n]
    pub(super) b_target: Vec<u32>,
    /// Target mode order for C: [batch, m, n]
    pub(super) c_target: Vec<u32>,
    pub(super) batch_modes: Vec<u32>,
    pub(super) m_modes: Vec<u32>,
    pub(super) n_modes: Vec<u32>,
    pub(super) k_modes: Vec<u32>,
}

/// Build a [`ContractGemmSpec`] from mode labels, or `None` if the
/// contraction is not a valid batched-GEMM pattern.
pub(super) fn build_contract_gemm_spec(
    modes_a: &[u32],
    modes_b: &[u32],
    modes_c: &[u32],
) -> Option<ContractGemmSpec> {
    let batch_modes: Vec<u32> = modes_c
        .iter()
        .copied()
        .filter(|m| modes_a.contains(m) && modes_b.contains(m))
        .collect();
    let m_modes: Vec<u32> = modes_c
        .iter()
        .copied()
        .filter(|m| modes_a.contains(m) && !modes_b.contains(m))
        .collect();
    let n_modes: Vec<u32> = modes_c
        .iter()
        .copied()
        .filter(|m| modes_b.contains(m) && !modes_a.contains(m))
        .collect();
    let k_modes: Vec<u32> = modes_a
        .iter()
        .copied()
        .filter(|m| modes_b.contains(m) && !modes_c.contains(m))
        .collect();

    let expected_a = batch_modes.len() + m_modes.len() + k_modes.len();
    let expected_b = batch_modes.len() + k_modes.len() + n_modes.len();
    if expected_a != modes_a.len() || expected_b != modes_b.len() {
        return None;
    }
    if batch_modes.len() + m_modes.len() + n_modes.len() != modes_c.len() {
        return None;
    }

    let a_target: Vec<u32> = batch_modes
        .iter()
        .chain(m_modes.iter())
        .chain(k_modes.iter())
        .copied()
        .collect();
    let b_target: Vec<u32> = batch_modes
        .iter()
        .chain(k_modes.iter())
        .chain(n_modes.iter())
        .copied()
        .collect();
    let c_target: Vec<u32> = batch_modes
        .iter()
        .chain(m_modes.iter())
        .chain(n_modes.iter())
        .copied()
        .collect();

    Some(ContractGemmSpec {
        a_target,
        b_target,
        c_target,
        batch_modes,
        m_modes,
        n_modes,
        k_modes,
    })
}

/// CPU plan — concrete enum, no type erasure.
///
/// Created by the family planners on [`crate::CpuBackend`] and consumed by the
/// semiring family executors.
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
    /// Plan for semiring-add reduction.
    ReduceAdd {
        /// Axes to reduce over (positions in input tensor).
        reduced_axes: Vec<usize>,
        _marker: PhantomData<T>,
    },
    /// Plan for trace.
    Trace {
        /// Output axis positions mapping.
        free_axes: Vec<usize>,
        /// Connected components of paired axes (union-find groups).
        components: Vec<Vec<usize>>,
        /// Dimension of each component (all axes in a component share the same dim).
        comp_dims: Vec<usize>,
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
    /// Plan for fused contraction.
    Contract {
        /// Mode labels for input A.
        modes_a: Vec<u32>,
        /// Mode labels for input B.
        modes_b: Vec<u32>,
        /// Mode labels for output C.
        modes_c: Vec<u32>,
        /// Cached GEMM mode analysis (None if not a valid GEMM pattern).
        #[allow(private_interfaces)]
        gemm_spec: Option<ContractGemmSpec>,
        _marker: PhantomData<T>,
    },
    /// Plan for optional semiring binary fast paths.
    ElementwiseBinary {
        /// The semiring binary operation to apply.
        op: SemiringBinaryOp,
        _marker: PhantomData<T>,
    },
    /// Plan for making a tensor contiguous.
    MakeContiguous { _marker: PhantomData<T> },
}
