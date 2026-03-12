use std::collections::HashSet;
use std::marker::PhantomData;

use tenferro_algebra::Scalar;
use tenferro_device::{Error, Result};
use tenferro_prims::SemiringCoreDescriptor;

use super::prims_view::mode_position;

/// Execution plan for tropical primitive operations on CPU.
///
/// Analogous to [`CpuPlan`](tenferro_prims::CpuPlan) but for tropical
/// algebras. The plan captures pre-computed kernel selection information.
///
/// # Examples
///
/// ```ignore
/// use tenferro_device::LogicalMemorySpace;
/// use tenferro_prims::{CpuBackend, CpuContext, SemiringCoreDescriptor, TensorSemiringCore};
/// use tenferro_tensor::{MemoryOrder, Tensor};
/// use tenferro_tropical::{MaxPlus, MaxPlusAlgebra, TropicalPlan};
///
/// let mut ctx = CpuContext::new(1);
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let a = Tensor::<MaxPlus<f64>>::zeros(&[3, 4], mem, col);
/// let mut c = Tensor::<MaxPlus<f64>>::zeros(&[3], mem, col);
/// let desc = SemiringCoreDescriptor::ReduceAdd {
///     modes_a: vec![0, 1],
///     modes_c: vec![0],
/// };
/// let plan =
///     <CpuBackend as TensorSemiringCore<MaxPlusAlgebra<f64>>>::plan(
///         &mut ctx,
///         &desc,
///         &[&[3, 4], &[3]],
///     )
///         .unwrap();
/// <CpuBackend as TensorSemiringCore<MaxPlusAlgebra<f64>>>::execute(
///     &mut ctx,
///     &plan,
///     MaxPlus::one(),
///     &[&a],
///     MaxPlus::zero(),
///     &mut c,
/// )
/// .unwrap();
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

pub(crate) fn tropical_plan<T: Scalar>(
    desc: &SemiringCoreDescriptor,
    shapes: &[&[usize]],
) -> Result<TropicalPlan<T>> {
    match desc {
        SemiringCoreDescriptor::BatchedGemm {
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
            if a_shape[0] != *m || a_shape[1] != *k {
                return Err(Error::InvalidArgument(
                    "BatchedGemm A shape mismatch".into(),
                ));
            }
            if b_shape[0] != *k || b_shape[1] != *n {
                return Err(Error::InvalidArgument(
                    "BatchedGemm B shape mismatch".into(),
                ));
            }
            if c_shape[0] != *m || c_shape[1] != *n {
                return Err(Error::InvalidArgument(
                    "BatchedGemm C shape mismatch".into(),
                ));
            }
            for (i, &bd) in batch_dims.iter().enumerate() {
                if a_shape[2 + i] != bd || b_shape[2 + i] != bd || c_shape[2 + i] != bd {
                    return Err(Error::InvalidArgument(
                        "BatchedGemm batch dimensions do not match shapes".into(),
                    ));
                }
            }

            Ok(TropicalPlan::BatchedGemm {
                batch_dims: batch_dims.clone(),
                m: *m,
                n: *n,
                k: *k,
                _marker: PhantomData,
            })
        }
        SemiringCoreDescriptor::ReduceAdd { modes_a, modes_c } => {
            ensure_shape_count(shapes, 2, "ReduceAdd")?;
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
                _marker: PhantomData,
            })
        }
        SemiringCoreDescriptor::Trace {
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
        SemiringCoreDescriptor::AntiTrace {
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
        SemiringCoreDescriptor::AntiDiag {
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
        SemiringCoreDescriptor::MakeContiguous => {
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
    }
}
