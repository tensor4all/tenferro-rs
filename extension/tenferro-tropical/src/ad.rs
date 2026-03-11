//! Tropical automatic differentiation via argmax routing.
//!
//! Tropical operations (max/min-based) are not smooth, so standard AD does not
//! apply. Instead, we use a **subgradient** approach: during the forward pass,
//! we record which input element "won" each tropical addition (max or min);
//! during the backward pass, gradients flow only to the winner.
//!
//! ## Key insight
//!
//! The forward pass operates on tropical scalars (`Tensor<MaxPlus<T>>`), but
//! the backward pass produces standard real gradients (`Tensor<T>`). This is
//! because the subgradient of `max(a, b)` w.r.t. `a` is `1` if `a >= b` and
//! `0` otherwise -- a standard real-valued quantity.
//!
//! ## Architecture
//!
//! - `tropical_forward_with_argmax`: runs tropical einsum forward and records
//!   winner indices in an [`ArgmaxTracker`]
//! - [`tropical_einsum_rrule`]: standalone reverse-mode rule (pullback)
//! - [`TropicalEinsumReverseRule`]: tape integration via
//!   [`ReverseRule<Tensor<T::Inner>>`](chainrules::ReverseRule)
//! - [`tracked_tropical_einsum`]: tape-aware tracked einsum for tropical ops
//!
//! ## Backward rules by semiring
//!
//! For a GEMM `C[i,j] = opt_k (A[i,k] (x) B[k,j])` where `opt` = max/min
//! and `(x)` is tropical multiplication:
//!
//! | Semiring | opt | (x) | `dA[i,k*]` | `dB[k*,j]` |
//! |----------|-----|-----|----------|----------|
//! | MaxPlus | max | + | `dC[i,j]` | `dC[i,j]` |
//! | MinPlus | min | + | `dC[i,j]` | `dC[i,j]` |
//! | MaxMul | max | x | `dC[i,j] * B.0[k*,j]` | `dC[i,j] * A.0[i,k*]` |
//!
//! where `k*` is the winner index from the forward pass.
//!
//! # Examples
//!
//! ## Standalone rrule
//!
//! ```ignore
//! use tenferro_tropical::ad::tropical_einsum_rrule;
//! use tenferro_tropical::{MaxPlus, MaxPlusAlgebra};
//! use tenferro_prims::{CpuBackend, CpuContext};
//! use tenferro_tensor::{Tensor, MemoryOrder};
//!
//! let mut ctx = CpuContext::new(1);
//! let a = Tensor::<MaxPlus<f64>>::from_slice(
//!     &[MaxPlus(1.0), MaxPlus(2.0), MaxPlus(3.0), MaxPlus(4.0)],
//!     &[2, 2], MemoryOrder::ColumnMajor,
//! ).unwrap();
//! let b = Tensor::<MaxPlus<f64>>::from_slice(
//!     &[MaxPlus(5.0), MaxPlus(6.0), MaxPlus(7.0), MaxPlus(8.0)],
//!     &[2, 2], MemoryOrder::ColumnMajor,
//! ).unwrap();
//! let grad_c = Tensor::<f64>::from_slice(
//!     &[1.0, 1.0, 1.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor,
//! ).unwrap();
//!
//! let grads = tropical_einsum_rrule::<MaxPlus<f64>, MaxPlusAlgebra<f64>, CpuBackend>(
//!     &mut ctx, "ij,jk->ik", &[&a, &b], &grad_c,
//! ).unwrap();
//! // grads[0] is dA (Tensor<f64>), grads[1] is dB (Tensor<f64>)
//! ```

use chainrules::{AdResult, Differentiable, NodeId, ReverseRule, TrackedTensor};
use num_traits::Zero;
use tenferro_algebra::{HasAlgebra, Scalar, Semiring};
use tenferro_device::{Error, Result};
use tenferro_einsum::Subscripts;
use tenferro_prims::TensorSemiringCore;
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::argmax::ArgmaxTracker;
use crate::prims::for_each_index;
use crate::prims::unflatten_index;

/// Trait for extracting the inner float type from a tropical scalar wrapper.
///
/// This enables generic code that operates on the inner values for backward
/// pass computations.
///
/// # Examples
///
/// ```ignore
/// use tenferro_tropical::MaxPlus;
/// use tenferro_tropical::ad::TropicalScalar;
///
/// let x = MaxPlus(3.0_f64);
/// assert_eq!(x.inner(), 3.0);
/// ```
pub trait TropicalScalar: Scalar {
    /// The inner floating-point type (f32 or f64).
    type Inner: Scalar
        + num_traits::Float
        + std::ops::AddAssign
        + HasAlgebra<Algebra = tenferro_algebra::Standard<Self::Inner>>;

    /// Extract the inner value.
    fn inner(&self) -> Self::Inner;

    /// Wrap an inner value into the tropical type.
    fn from_inner(v: Self::Inner) -> Self;

    /// Compute the backward contribution for tropical multiplication.
    ///
    /// For additive tropical mul (MaxPlus/MinPlus): d(a+b)/da = 1, so
    /// `mul_backward_a(a, b, dout) = dout`.
    ///
    /// For multiplicative tropical mul (MaxMul): d(a*b)/da = b, so
    /// `mul_backward_a(a, b, dout) = dout * b`.
    fn mul_backward_a(a_inner: Self::Inner, b_inner: Self::Inner, dout: Self::Inner)
        -> Self::Inner;

    /// Compute the backward contribution for tropical multiplication w.r.t. second operand.
    ///
    /// For additive tropical mul (MaxPlus/MinPlus): d(a+b)/db = 1, so
    /// `mul_backward_b(a, b, dout) = dout`.
    ///
    /// For multiplicative tropical mul (MaxMul): d(a*b)/db = a, so
    /// `mul_backward_b(a, b, dout) = dout * a`.
    fn mul_backward_b(a_inner: Self::Inner, b_inner: Self::Inner, dout: Self::Inner)
        -> Self::Inner;
}

/// Macro for additive tropical semirings (MaxPlus, MinPlus) where
/// tropical mul = ordinary addition, so backward is identity.
macro_rules! impl_tropical_scalar_additive {
    ($wrapper:ident, $float:ty) => {
        impl TropicalScalar for crate::$wrapper<$float> {
            type Inner = $float;

            fn inner(&self) -> $float {
                self.0
            }

            fn from_inner(v: $float) -> Self {
                crate::$wrapper(v)
            }

            fn mul_backward_a(_a: $float, _b: $float, dout: $float) -> $float {
                dout
            }

            fn mul_backward_b(_a: $float, _b: $float, dout: $float) -> $float {
                dout
            }
        }
    };
}

/// Macro for multiplicative tropical semirings (MaxMul) where
/// tropical mul = ordinary multiplication, so backward uses product rule.
macro_rules! impl_tropical_scalar_multiplicative {
    ($wrapper:ident, $float:ty) => {
        impl TropicalScalar for crate::$wrapper<$float> {
            type Inner = $float;

            fn inner(&self) -> $float {
                self.0
            }

            fn from_inner(v: $float) -> Self {
                crate::$wrapper(v)
            }

            fn mul_backward_a(_a: $float, b: $float, dout: $float) -> $float {
                dout * b
            }

            fn mul_backward_b(a: $float, _b: $float, dout: $float) -> $float {
                dout * a
            }
        }
    };
}

impl_tropical_scalar_additive!(MaxPlus, f32);
impl_tropical_scalar_additive!(MaxPlus, f64);
impl_tropical_scalar_additive!(MinPlus, f32);
impl_tropical_scalar_additive!(MinPlus, f64);
impl_tropical_scalar_multiplicative!(MaxMul, f32);
impl_tropical_scalar_multiplicative!(MaxMul, f64);

/// Compute column-major flat index for a given multi-dimensional index.
fn col_major_flat_index(shape: &[usize], idx: &[usize]) -> usize {
    let mut flat = 0;
    let mut stride = 1;
    for (d, &i) in idx.iter().enumerate() {
        flat += i * stride;
        stride *= shape[d];
    }
    flat
}

// ============================================================================
// Standalone rrule for tropical einsum
// ============================================================================

/// Reverse-mode rule (rrule) for tropical einsum.
///
/// Given a tropical einsum operation and a cotangent (in standard reals),
/// computes gradients for each input operand (also in standard reals).
///
/// Currently supports unary (1 operand) and binary (2 operand) contractions.
/// Unary patterns include trace (`ii->`), full contraction (`ij->`),
/// and partial reduction (`ij->i`, `ij->j`).
///
/// # Arguments
///
/// * `ctx` - Backend context
/// * `subscripts` - Einsum subscript string (e.g., "ij,jk->ik")
/// * `operands` - Input tropical tensors
/// * `cotangent` - Gradient w.r.t. output in standard reals
///
/// # Returns
///
/// Vector of gradient tensors (one per operand), in standard reals.
///
/// # Examples
///
/// ```ignore
/// use tenferro_tropical::ad::tropical_einsum_rrule;
/// use tenferro_tropical::{MaxPlus, MaxPlusAlgebra};
/// use tenferro_prims::{CpuBackend, CpuContext};
/// use tenferro_tensor::{Tensor, MemoryOrder};
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<MaxPlus<f64>>::from_slice(
///     &[MaxPlus(1.0), MaxPlus(2.0), MaxPlus(3.0), MaxPlus(4.0)],
///     &[2, 2], MemoryOrder::ColumnMajor,
/// ).unwrap();
/// let b = Tensor::<MaxPlus<f64>>::from_slice(
///     &[MaxPlus(5.0), MaxPlus(6.0), MaxPlus(7.0), MaxPlus(8.0)],
///     &[2, 2], MemoryOrder::ColumnMajor,
/// ).unwrap();
/// let grad_c = Tensor::<f64>::from_slice(
///     &[1.0, 1.0, 1.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor,
/// ).unwrap();
///
/// let grads = tropical_einsum_rrule::<MaxPlus<f64>, MaxPlusAlgebra<f64>, CpuBackend>(
///     &mut ctx, "ij,jk->ik", &[&a, &b], &grad_c,
/// ).unwrap();
/// assert_eq!(grads.len(), 2);
/// ```
pub fn tropical_einsum_rrule<T, Alg, Backend>(
    _ctx: &mut Backend::Context,
    subscripts: &str,
    operands: &[&Tensor<T>],
    cotangent: &Tensor<T::Inner>,
) -> Result<Vec<Tensor<T::Inner>>>
where
    Alg: Semiring<Scalar = T>,
    T: TropicalScalar + HasAlgebra<Algebra = Alg>,
    Backend: TensorSemiringCore<Alg>,
{
    let subs = Subscripts::parse(subscripts)?;

    if operands.is_empty() || operands.len() > 2 {
        return Err(Error::InvalidArgument(
            "tropical_einsum_rrule supports 1 or 2 operands".into(),
        ));
    }

    // Contracted: labels in any input but not in output
    let contracted: Vec<u32> = {
        let all_input_labels: std::collections::HashSet<u32> = subs
            .inputs
            .iter()
            .flat_map(|inp| inp.iter())
            .copied()
            .collect();
        all_input_labels
            .into_iter()
            .filter(|m| !subs.output.contains(m))
            .collect()
    };

    // Run forward with argmax tracking so we can validate the cotangent against
    // the logical output shape before any backward indexing happens.
    let (output, tracker) = tropical_forward_with_argmax(operands, &subs, &contracted)?;

    if cotangent.dims() != output.dims() {
        return Err(Error::InvalidArgument(format!(
            "cotangent shape mismatch: expected {:?}, got {:?}",
            output.dims(),
            cotangent.dims()
        )));
    }

    // Compute backward using the tracker
    tropical_backward(operands, cotangent, &tracker, &subs, &contracted)
}

/// Forward-mode rule (frule) for tropical einsum.
///
/// Given tropical primals and optional standard-real tangents, compute the
/// output tangent by routing each tangent contribution through the winner
/// selected during the tropical forward pass.
///
/// Currently supports unary (1 operand) and binary (2 operand) contractions.
///
/// # Examples
///
/// ```ignore
/// use tenferro_tropical::ad::tropical_einsum_frule;
/// use tenferro_tropical::{MaxPlus, MaxPlusAlgebra};
/// use tenferro_prims::{CpuBackend, CpuContext};
/// use tenferro_tensor::{Tensor, MemoryOrder};
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<MaxPlus<f64>>::from_slice(
///     &[MaxPlus(1.0), MaxPlus(2.0), MaxPlus(3.0), MaxPlus(4.0)],
///     &[2, 2], MemoryOrder::ColumnMajor,
/// ).unwrap();
/// let b = Tensor::<MaxPlus<f64>>::from_slice(
///     &[MaxPlus(5.0), MaxPlus(6.0), MaxPlus(7.0), MaxPlus(8.0)],
///     &[2, 2], MemoryOrder::ColumnMajor,
/// ).unwrap();
/// let da = Tensor::<f64>::from_slice(
///     &[1.0, 0.0, 0.0, 0.0], &[2, 2], MemoryOrder::ColumnMajor,
/// ).unwrap();
///
/// let dc = tropical_einsum_frule::<MaxPlus<f64>, MaxPlusAlgebra<f64>, CpuBackend>(
///     &mut ctx, "ij,jk->ik", &[&a, &b], &[Some(&da), None],
/// ).unwrap();
/// assert_eq!(dc.dims(), &[2, 2]);
/// ```
pub fn tropical_einsum_frule<T, Alg, Backend>(
    _ctx: &mut Backend::Context,
    subscripts: &str,
    primals: &[&Tensor<T>],
    tangents: &[Option<&Tensor<T::Inner>>],
) -> Result<Tensor<T::Inner>>
where
    Alg: Semiring<Scalar = T>,
    T: TropicalScalar + HasAlgebra<Algebra = Alg>,
    Backend: TensorSemiringCore<Alg>,
{
    let subs = Subscripts::parse(subscripts)?;

    if primals.is_empty() || primals.len() > 2 {
        return Err(Error::InvalidArgument(
            "tropical_einsum_frule supports 1 or 2 operands".into(),
        ));
    }
    if tangents.len() != primals.len() {
        return Err(Error::InvalidArgument(format!(
            "tropical_einsum_frule expects {} tangents, got {}",
            primals.len(),
            tangents.len()
        )));
    }
    for (idx, (primal, tangent)) in primals.iter().zip(tangents.iter()).enumerate() {
        if let Some(tangent) = tangent {
            if tangent.dims() != primal.dims() {
                return Err(Error::InvalidArgument(format!(
                    "tangent shape mismatch for operand {idx}: expected {:?}, got {:?}",
                    primal.dims(),
                    tangent.dims()
                )));
            }
        }
    }

    let contracted: Vec<u32> = {
        let all_input_labels: std::collections::HashSet<u32> = subs
            .inputs
            .iter()
            .flat_map(|inp| inp.iter())
            .copied()
            .collect();
        all_input_labels
            .into_iter()
            .filter(|m| !subs.output.contains(m))
            .collect()
    };

    let (output, tracker) = tropical_forward_with_argmax(primals, &subs, &contracted)?;
    tropical_forward_tangent(
        primals,
        tangents,
        &tracker,
        &subs,
        &contracted,
        output.dims(),
    )
}

/// Forward pass for tropical N-ary einsum with argmax tracking.
///
/// Handles arbitrary subscript patterns by iterating over output indices
/// and contracted indices according to the subscript structure. Supports
/// 1 or more operands.
fn tropical_forward_with_argmax<T: TropicalScalar>(
    operands: &[&Tensor<T>],
    subs: &Subscripts,
    contracted: &[u32],
) -> Result<(Tensor<T>, ArgmaxTracker)> {
    let output_modes = &subs.output;

    let views: Vec<_> = operands
        .iter()
        .map(|op| crate::prims::tensor_to_view(*op))
        .collect::<Result<_>>()?;

    // Build output shape: resolve each output mode from the first operand that has it
    let output_shape: Vec<usize> = output_modes
        .iter()
        .map(|m| {
            for (op_idx, input_modes) in subs.inputs.iter().enumerate() {
                if let Some(pos) = input_modes.iter().position(|x| x == m) {
                    return Ok(operands[op_idx].dims()[pos]);
                }
            }
            Err(Error::InvalidArgument(format!(
                "output mode {m} not found in inputs"
            )))
        })
        .collect::<Result<Vec<_>>>()?;

    // Build contracted dimension sizes from the first operand that has each label
    let contracted_dims: Vec<usize> = contracted
        .iter()
        .map(|m| {
            for (op_idx, input_modes) in subs.inputs.iter().enumerate() {
                if let Some(pos) = input_modes.iter().position(|x| x == m) {
                    return Ok(operands[op_idx].dims()[pos]);
                }
            }
            Err(Error::InvalidArgument(format!(
                "contracted mode {m} not in any operand"
            )))
        })
        .collect::<Result<Vec<_>>>()?;

    // Build a dimension map and validate all operand dimensions match (#346)
    let mut dim_map: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
    for (pos, &m) in output_modes.iter().enumerate() {
        dim_map.insert(m, output_shape[pos]);
    }
    for (c_pos, &m) in contracted.iter().enumerate() {
        dim_map.insert(m, contracted_dims[c_pos]);
    }
    for (op_idx, input_modes) in subs.inputs.iter().enumerate() {
        let op_dims = operands[op_idx].dims();
        for (pos, &m) in input_modes.iter().enumerate() {
            if let Some(&expected) = dim_map.get(&m) {
                if op_dims[pos] != expected {
                    return Err(Error::InvalidArgument(format!(
                        "dimension mismatch for mode {m}: operand {op_idx} has size {} but expected {expected}",
                        op_dims[pos]
                    )));
                }
            }
        }
    }

    let contracted_total: usize = contracted_dims.iter().product::<usize>().max(1);

    // Pre-validate that all input modes are in output or contracted (#343)
    for (op_idx, input_modes) in subs.inputs.iter().enumerate() {
        for m in input_modes.iter() {
            if !output_modes.contains(m) && !contracted.contains(m) {
                return Err(Error::InvalidArgument(format!(
                    "mode {m} in operand {op_idx} not found in output or contracted modes"
                )));
            }
        }
    }

    let total_output: usize = output_shape.iter().product::<usize>().max(1);
    let mut output_data = vec![T::zero(); total_output];
    let mut tracker = ArgmaxTracker::new(&output_shape);

    for_each_index(&output_shape, |out_idx| {
        let mut mode_values: std::collections::HashMap<u32, usize> =
            std::collections::HashMap::new();
        for (pos, &m) in output_modes.iter().enumerate() {
            mode_values.insert(m, out_idx[pos]);
        }

        let mut best = T::zero();
        let mut best_k = 0_usize;

        for k_flat in 0..contracted_total {
            let k_idx = if contracted_dims.is_empty() {
                vec![]
            } else {
                unflatten_index(k_flat, &contracted_dims)
            };

            for (c_pos, &c_mode) in contracted.iter().enumerate() {
                mode_values.insert(c_mode, k_idx[c_pos]);
            }

            // All modes pre-validated; lookup cannot fail
            let mut product = T::one();
            for (op_idx, input_modes) in subs.inputs.iter().enumerate() {
                let idx: Vec<usize> = input_modes
                    .iter()
                    .map(|m| {
                        *mode_values
                            .get(m)
                            .expect("mode pre-validated to exist in map")
                    })
                    .collect();
                product = product * views[op_idx].get(&idx);
            }

            let new_sum = best + product;
            // Keep the first winner on ties so gradients route to the
            // smallest linear contraction index deterministically.
            if k_flat == 0 || new_sum.inner() != best.inner() {
                best_k = k_flat;
            }
            best = new_sum;
        }

        let out_flat = col_major_flat_index(&output_shape, out_idx);
        output_data[out_flat] = best;
        tracker.indices_mut()[out_flat] = best_k;
    });

    let output = Tensor::<T>::from_slice(&output_data, &output_shape, MemoryOrder::ColumnMajor)
        .map_err(|e| Error::InvalidArgument(format!("{e}")))?;
    Ok((output, tracker))
}

fn tropical_forward_tangent<T: TropicalScalar>(
    primals: &[&Tensor<T>],
    tangents: &[Option<&Tensor<T::Inner>>],
    tracker: &ArgmaxTracker,
    subs: &Subscripts,
    contracted: &[u32],
    output_shape: &[usize],
) -> Result<Tensor<T::Inner>> {
    match primals.len() {
        1 => tropical_forward_tangent_unary(
            primals[0],
            tangents[0],
            tracker,
            subs,
            contracted,
            output_shape,
        ),
        2 => tropical_forward_tangent_binary(
            primals,
            tangents,
            tracker,
            subs,
            contracted,
            output_shape,
        ),
        n => Err(Error::InvalidArgument(format!(
            "tropical forward tangent supports 1 or 2 operands, got {n}"
        ))),
    }
}

fn tropical_forward_tangent_unary<T: TropicalScalar>(
    primal: &Tensor<T>,
    tangent: Option<&Tensor<T::Inner>>,
    tracker: &ArgmaxTracker,
    subs: &Subscripts,
    contracted: &[u32],
    output_shape: &[usize],
) -> Result<Tensor<T::Inner>> {
    let input_modes = &subs.inputs[0];
    let output_modes = &subs.output;
    let tangent_view = tangent.map(crate::prims::tensor_to_view).transpose()?;

    let contracted_dims: Vec<usize> = contracted
        .iter()
        .map(|m| {
            let pos = input_modes.iter().position(|x| x == m).ok_or_else(|| {
                Error::InvalidArgument(format!("contracted mode {m} not in input"))
            })?;
            Ok(primal.dims()[pos])
        })
        .collect::<Result<Vec<_>>>()?;

    let mut output_data = vec![T::Inner::zero(); output_shape.iter().product::<usize>().max(1)];

    for_each_index(output_shape, |out_idx| {
        let mut mode_values: std::collections::HashMap<u32, usize> =
            std::collections::HashMap::new();
        for (pos, &m) in output_modes.iter().enumerate() {
            mode_values.insert(m, out_idx[pos]);
        }

        let out_flat = col_major_flat_index(output_shape, out_idx);
        let k_winner = tracker.indices()[out_flat];
        let k_idx = if contracted_dims.is_empty() {
            vec![]
        } else {
            unflatten_index(k_winner, &contracted_dims)
        };

        for (c_pos, &c_mode) in contracted.iter().enumerate() {
            mode_values.insert(c_mode, k_idx[c_pos]);
        }

        let input_idx: Vec<usize> = input_modes
            .iter()
            .map(|m| {
                *mode_values
                    .get(m)
                    .expect("mode pre-validated to exist in map")
            })
            .collect();

        output_data[out_flat] = tangent_view
            .as_ref()
            .map(|view| view.get(&input_idx))
            .unwrap_or_else(T::Inner::zero);
    });

    Tensor::<T::Inner>::from_slice(&output_data, output_shape, MemoryOrder::ColumnMajor)
        .map_err(|e| Error::InvalidArgument(format!("{e}")))
}

fn tropical_forward_tangent_binary<T: TropicalScalar>(
    primals: &[&Tensor<T>],
    tangents: &[Option<&Tensor<T::Inner>>],
    tracker: &ArgmaxTracker,
    subs: &Subscripts,
    contracted: &[u32],
    output_shape: &[usize],
) -> Result<Tensor<T::Inner>> {
    let a = primals[0];
    let b = primals[1];
    let da_view = tangents[0].map(crate::prims::tensor_to_view).transpose()?;
    let db_view = tangents[1].map(crate::prims::tensor_to_view).transpose()?;
    let a_view = crate::prims::tensor_to_view(a)?;
    let b_view = crate::prims::tensor_to_view(b)?;
    let input_modes_a = &subs.inputs[0];
    let input_modes_b = &subs.inputs[1];
    let output_modes = &subs.output;

    let contracted_dims: Vec<usize> = contracted
        .iter()
        .map(|m| {
            for (op_idx, input_modes) in subs.inputs.iter().enumerate() {
                if let Some(pos) = input_modes.iter().position(|x| x == m) {
                    return Ok(primals[op_idx].dims()[pos]);
                }
            }
            Err(Error::InvalidArgument(format!(
                "contracted mode {m} not in any operand"
            )))
        })
        .collect::<Result<Vec<_>>>()?;

    let mut output_data = vec![T::Inner::zero(); output_shape.iter().product::<usize>().max(1)];

    for_each_index(output_shape, |out_idx| {
        let mut mode_values: std::collections::HashMap<u32, usize> =
            std::collections::HashMap::new();
        for (pos, &m) in output_modes.iter().enumerate() {
            mode_values.insert(m, out_idx[pos]);
        }

        let out_flat = col_major_flat_index(output_shape, out_idx);
        let k_winner = tracker.indices()[out_flat];
        let k_idx = if contracted_dims.is_empty() {
            vec![]
        } else {
            unflatten_index(k_winner, &contracted_dims)
        };

        for (c_pos, &c_mode) in contracted.iter().enumerate() {
            mode_values.insert(c_mode, k_idx[c_pos]);
        }

        let a_idx: Vec<usize> = input_modes_a
            .iter()
            .map(|m| {
                *mode_values
                    .get(m)
                    .expect("mode pre-validated to exist in map")
            })
            .collect();
        let b_idx: Vec<usize> = input_modes_b
            .iter()
            .map(|m| {
                *mode_values
                    .get(m)
                    .expect("mode pre-validated to exist in map")
            })
            .collect();

        let a_inner = a_view.get(&a_idx).inner();
        let b_inner = b_view.get(&b_idx).inner();

        let mut dout = T::Inner::zero();
        if let Some(view) = da_view.as_ref() {
            dout += T::mul_backward_a(a_inner, b_inner, view.get(&a_idx));
        }
        if let Some(view) = db_view.as_ref() {
            dout += T::mul_backward_b(a_inner, b_inner, view.get(&b_idx));
        }
        output_data[out_flat] = dout;
    });

    Tensor::<T::Inner>::from_slice(&output_data, output_shape, MemoryOrder::ColumnMajor)
        .map_err(|e| Error::InvalidArgument(format!("{e}")))
}

/// Backward pass dispatcher for tropical N-ary einsum using argmax routing.
///
/// Dispatches to unary or binary backward based on operand count.
fn tropical_backward<T: TropicalScalar>(
    operands: &[&Tensor<T>],
    cotangent: &Tensor<T::Inner>,
    tracker: &ArgmaxTracker,
    subs: &Subscripts,
    contracted: &[u32],
) -> Result<Vec<Tensor<T::Inner>>> {
    match operands.len() {
        1 => tropical_backward_unary(operands[0], cotangent, tracker, subs, contracted),
        2 => tropical_backward_binary(operands, cotangent, tracker, subs, contracted),
        n => Err(Error::InvalidArgument(format!(
            "tropical backward supports 1 or 2 operands, got {n}"
        ))),
    }
}

/// Backward pass for unary tropical einsum.
///
/// For a unary contraction, the gradient simply scatters the cotangent to the
/// winning input position. There is no tropical multiplication to differentiate
/// through (the product of a single operand is the operand itself).
fn tropical_backward_unary<T: TropicalScalar>(
    operand: &Tensor<T>,
    cotangent: &Tensor<T::Inner>,
    tracker: &ArgmaxTracker,
    subs: &Subscripts,
    contracted: &[u32],
) -> Result<Vec<Tensor<T::Inner>>> {
    let input_modes = &subs.inputs[0];
    let output_modes = &subs.output;
    let output_shape = tracker.output_shape();

    let cot_view = crate::prims::tensor_to_view(cotangent)?;

    let contracted_dims: Vec<usize> = contracted
        .iter()
        .map(|m| {
            let pos = input_modes.iter().position(|x| x == m).ok_or_else(|| {
                Error::InvalidArgument(format!("contracted mode {m} not in input"))
            })?;
            Ok(operand.dims()[pos])
        })
        .collect::<Result<Vec<_>>>()?;

    // Pre-validate that all input modes exist in output or contracted (#344)
    for m in input_modes.iter() {
        if !output_modes.contains(m) && !contracted.contains(m) {
            return Err(Error::InvalidArgument(format!(
                "mode {m} not found in output or contracted modes"
            )));
        }
    }

    let mut grad_data = vec![T::Inner::zero(); operand.len()];
    let mut backward_error: Option<Error> = None;

    for_each_index(output_shape, |out_idx| {
        if backward_error.is_some() {
            return;
        }
        let mut mode_values: std::collections::HashMap<u32, usize> =
            std::collections::HashMap::new();
        for (pos, &m) in output_modes.iter().enumerate() {
            mode_values.insert(m, out_idx[pos]);
        }

        let dout = cot_view.get(out_idx);

        let out_flat = col_major_flat_index(output_shape, out_idx);
        let k_winner = tracker.indices()[out_flat];

        let k_idx = if contracted_dims.is_empty() {
            vec![]
        } else {
            unflatten_index(k_winner, &contracted_dims)
        };

        for (c_pos, &c_mode) in contracted.iter().enumerate() {
            mode_values.insert(c_mode, k_idx[c_pos]);
        }

        let input_idx: Vec<usize> = input_modes
            .iter()
            .map(|m| {
                *mode_values
                    .get(m)
                    .expect("mode pre-validated to exist in map")
            })
            .collect();

        let input_flat = col_major_flat_index(operand.dims(), &input_idx);
        // Bounds check on flat index (#348)
        if let Some(slot) = grad_data.get_mut(input_flat) {
            *slot += dout;
        } else {
            backward_error = Some(Error::InvalidArgument(format!(
                "flat index {input_flat} out of bounds for gradient buffer of size {}",
                grad_data.len()
            )));
        }
    });

    if let Some(err) = backward_error {
        return Err(err);
    }

    let grad = Tensor::<T::Inner>::from_slice(&grad_data, operand.dims(), MemoryOrder::ColumnMajor)
        .map_err(|e| Error::InvalidArgument(format!("{e}")))?;
    Ok(vec![grad])
}

/// Backward pass for binary tropical einsum using argmax routing.
fn tropical_backward_binary<T: TropicalScalar>(
    operands: &[&Tensor<T>],
    cotangent: &Tensor<T::Inner>,
    tracker: &ArgmaxTracker,
    subs: &Subscripts,
    contracted: &[u32],
) -> Result<Vec<Tensor<T::Inner>>> {
    let a = operands[0];
    let b = operands[1];
    let input_modes_a = &subs.inputs[0];
    let input_modes_b = &subs.inputs[1];
    let output_modes = &subs.output;

    let a_view = crate::prims::tensor_to_view(a)?;
    let b_view = crate::prims::tensor_to_view(b)?;
    let cot_view = crate::prims::tensor_to_view(cotangent)?;

    let output_shape = tracker.output_shape();

    let contracted_dims: Vec<usize> = contracted
        .iter()
        .map(|m| {
            for (op_idx, input_modes) in subs.inputs.iter().enumerate() {
                if let Some(pos) = input_modes.iter().position(|x| x == m) {
                    return Ok(operands[op_idx].dims()[pos]);
                }
            }
            Err(Error::InvalidArgument(format!(
                "contracted mode {m} not in any operand"
            )))
        })
        .collect::<Result<Vec<_>>>()?;

    // Pre-validate that all input modes exist in output or contracted (#345)
    for m in input_modes_a.iter().chain(input_modes_b.iter()) {
        if !output_modes.contains(m) && !contracted.contains(m) {
            return Err(Error::InvalidArgument(format!(
                "mode {m} not found in output or contracted modes"
            )));
        }
    }

    let mut da_data = vec![T::Inner::zero(); a.len()];
    let mut db_data = vec![T::Inner::zero(); b.len()];
    let mut backward_error: Option<Error> = None;

    for_each_index(output_shape, |out_idx| {
        if backward_error.is_some() {
            return;
        }
        let mut mode_values: std::collections::HashMap<u32, usize> =
            std::collections::HashMap::new();
        for (pos, &m) in output_modes.iter().enumerate() {
            mode_values.insert(m, out_idx[pos]);
        }

        let dout = cot_view.get(out_idx);

        let out_flat = col_major_flat_index(output_shape, out_idx);
        let k_winner = tracker.indices()[out_flat];

        let k_idx = if contracted_dims.is_empty() {
            vec![]
        } else {
            unflatten_index(k_winner, &contracted_dims)
        };

        for (c_pos, &c_mode) in contracted.iter().enumerate() {
            mode_values.insert(c_mode, k_idx[c_pos]);
        }

        let a_idx: Vec<usize> = input_modes_a
            .iter()
            .map(|m| {
                *mode_values
                    .get(m)
                    .expect("mode pre-validated to exist in map")
            })
            .collect();
        let b_idx: Vec<usize> = input_modes_b
            .iter()
            .map(|m| {
                *mode_values
                    .get(m)
                    .expect("mode pre-validated to exist in map")
            })
            .collect();

        let a_val = a_view.get(&a_idx).inner();
        let b_val = b_view.get(&b_idx).inner();

        let da_contrib = T::mul_backward_a(a_val, b_val, dout);
        let db_contrib = T::mul_backward_b(a_val, b_val, dout);

        let a_flat = col_major_flat_index(a.dims(), &a_idx);
        let b_flat = col_major_flat_index(b.dims(), &b_idx);

        // Bounds check on flat indices (#348)
        if let Some(slot) = da_data.get_mut(a_flat) {
            *slot += da_contrib;
        } else {
            backward_error = Some(Error::InvalidArgument(format!(
                "flat index {a_flat} out of bounds for da buffer of size {}",
                da_data.len()
            )));
            return;
        }
        if let Some(slot) = db_data.get_mut(b_flat) {
            *slot += db_contrib;
        } else {
            backward_error = Some(Error::InvalidArgument(format!(
                "flat index {b_flat} out of bounds for db buffer of size {}",
                db_data.len()
            )));
        }
    });

    if let Some(err) = backward_error {
        return Err(err);
    }

    let da = Tensor::<T::Inner>::from_slice(&da_data, a.dims(), MemoryOrder::ColumnMajor)
        .map_err(|e| Error::InvalidArgument(format!("{e}")))?;
    let db = Tensor::<T::Inner>::from_slice(&db_data, b.dims(), MemoryOrder::ColumnMajor)
        .map_err(|e| Error::InvalidArgument(format!("{e}")))?;

    Ok(vec![da, db])
}

// ============================================================================
// Tape-integrated tropical AD
// ============================================================================

/// Reverse-mode rule for tropical einsum, for integration with [`chainrules::Tape`].
///
/// This rule stores the tropical primal tensors and the argmax tracker from
/// the forward pass. The pullback computes standard real gradients.
///
/// Note: The `ReverseRule` is parameterized by `Tensor<T::Inner>` (e.g.,
/// `Tensor<f64>` or `Tensor<f32>`), not `Tensor<MaxPlus<T>>`, because
/// gradients live in standard reals.
///
/// # Examples
///
/// ```ignore
/// use tenferro_tropical::ad::TropicalEinsumReverseRule;
/// use tenferro_tropical::MaxPlus;
/// use tenferro_tensor::Tensor;
/// use chainrules::ReverseRule;
/// ```
pub struct TropicalEinsumReverseRule<T: TropicalScalar> {
    subscripts: Subscripts,
    primals: Vec<Tensor<T>>,
    tracker: ArgmaxTracker,
    input_node_ids: Vec<Option<NodeId>>,
    contracted: Vec<u32>,
}

impl<T> ReverseRule<Tensor<T::Inner>> for TropicalEinsumReverseRule<T>
where
    T: TropicalScalar,
    T::Inner: Scalar,
    Tensor<T::Inner>: Differentiable<Tangent = Tensor<T::Inner>>,
{
    fn pullback(&self, cotangent: &Tensor<T::Inner>) -> AdResult<Vec<(NodeId, Tensor<T::Inner>)>> {
        let primal_refs: Vec<&Tensor<T>> = self.primals.iter().collect();
        let grads = tropical_backward(
            &primal_refs,
            cotangent,
            &self.tracker,
            &self.subscripts,
            &self.contracted,
        )
        .map_err(|e| chainrules::AutodiffError::InvalidArgument(format!("{e}")))?;

        let mut results = Vec::new();
        for (i, grad) in grads.into_iter().enumerate() {
            if let Some(id) = self.input_node_ids[i] {
                results.push((id, grad));
            }
        }
        Ok(results)
    }

    fn inputs(&self) -> Vec<NodeId> {
        self.input_node_ids.iter().filter_map(|id| *id).collect()
    }
}

/// Tracked tropical einsum for reverse-mode AD.
///
/// Runs the tropical einsum forward pass, records winner indices, and
/// returns a tracked tensor whose value is the output's inner values
/// (standard reals). The reverse-mode rule routes gradients only to
/// the winning elements.
///
/// Currently supports unary (1 operand) and binary (2 operand) contractions.
/// Unary patterns include trace (`ii->`), full contraction (`ij->`),
/// and partial reduction (`ij->i`, `ij->j`).
///
/// The input `TrackedTensor` values wrap `Tensor<Inner>` (standard reals),
/// which are internally promoted to tropical scalars for the forward pass.
///
/// # Examples
///
/// ```ignore
/// use std::cell::RefCell;
/// use std::rc::Rc;
/// use chainrules::Tape;
/// use tenferro_tropical::ad::tracked_tropical_einsum;
/// use tenferro_tropical::{MaxPlus, MaxPlusAlgebra};
/// use tenferro_prims::{CpuBackend, CpuContext};
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let tape = Tape::<Tensor<f64>>::new();
/// let a_data = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap();
/// let b_data = Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap();
/// let a = tape.leaf(a_data);
/// let b = tape.leaf(b_data);
///
/// let c = tracked_tropical_einsum::<MaxPlus<f64>, MaxPlusAlgebra<f64>, CpuBackend>(
///     "ij,jk->ik", &[&a, &b],
/// ).unwrap();
///
/// let grads = tape.pullback(&c).unwrap();
/// ```
pub fn tracked_tropical_einsum<T, Alg, Backend>(
    subscripts: &str,
    operands: &[&TrackedTensor<Tensor<T::Inner>>],
) -> AdResult<TrackedTensor<Tensor<T::Inner>>>
where
    Alg: Semiring<Scalar = T>,
    T: TropicalScalar + HasAlgebra<Algebra = Alg> + 'static,
    T::Inner: Scalar + HasAlgebra,
    Backend: TensorSemiringCore<Alg>,
    Tensor<T::Inner>: Differentiable<Tangent = Tensor<T::Inner>>,
{
    if operands.is_empty() || operands.len() > 2 {
        return Err(chainrules::AutodiffError::InvalidArgument(
            "tracked_tropical_einsum supports 1 or 2 operands".into(),
        ));
    }

    let subs = Subscripts::parse(subscripts)
        .map_err(|e| chainrules::AutodiffError::InvalidArgument(format!("{e}")))?;

    // Contracted: labels in any input but not in output
    let contracted: Vec<u32> = {
        let all_input_labels: std::collections::HashSet<u32> = subs
            .inputs
            .iter()
            .flat_map(|inp| inp.iter())
            .copied()
            .collect();
        all_input_labels
            .into_iter()
            .filter(|m| !subs.output.contains(m))
            .collect()
    };

    // Promote all operands to tropical scalars
    let tropical_operands: Vec<Tensor<T>> = operands
        .iter()
        .map(|op| promote_to_tropical::<T>(op.value()))
        .collect::<std::result::Result<_, _>>()
        .map_err(|e| chainrules::AutodiffError::InvalidArgument(format!("{e}")))?;

    let tropical_refs: Vec<&Tensor<T>> = tropical_operands.iter().collect();

    // Run forward with argmax tracking
    let (output_tropical, tracker) =
        tropical_forward_with_argmax(&tropical_refs, &subs, &contracted)
            .map_err(|e| chainrules::AutodiffError::InvalidArgument(format!("{e}")))?;

    // Extract inner values from tropical output
    let output_inner = extract_inner::<T>(&output_tropical)
        .map_err(|e| chainrules::AutodiffError::InvalidArgument(format!("{e}")))?;

    // Check if any operand requires gradients
    let any_requires_grad = operands.iter().any(|op| op.requires_grad());
    if !any_requires_grad {
        return Ok(TrackedTensor::new(output_inner));
    }

    // Find tape
    let tape = operands
        .iter()
        .filter(|op| op.requires_grad())
        .find_map(|op| op.tape())
        .ok_or(chainrules::AutodiffError::MissingNode)?
        .clone();

    // Verify all tracked operands share the same tape
    for op in operands.iter().filter(|op| op.requires_grad()) {
        if let Some(op_tape) = op.tape() {
            if !tape.same_tape(op_tape) {
                return Err(chainrules::AutodiffError::InvalidArgument(
                    "tracked_tropical_einsum: operands belong to different AD tapes".into(),
                ));
            }
        }
    }

    let rule = TropicalEinsumReverseRule::<T> {
        subscripts: subs,
        primals: tropical_operands,
        tracker,
        input_node_ids: operands.iter().map(|op| op.node_id()).collect(),
        contracted,
    };

    let result = tape.record_op(output_inner, Box::new(rule), None);
    Ok(result)
}

/// Promote a standard real tensor to a tropical scalar tensor.
///
/// # Examples
///
/// ```ignore
/// use tenferro_tropical::MaxPlus;
/// use tenferro_tropical::ad::promote_to_tropical;
/// use tenferro_tensor::{Tensor, MemoryOrder};
///
/// let t = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
/// let tropical = promote_to_tropical::<MaxPlus<f64>>(&t).unwrap();
/// ```
pub fn promote_to_tropical<T: TropicalScalar>(tensor: &Tensor<T::Inner>) -> Result<Tensor<T>> {
    tensor
        .buffer()
        .as_slice()
        .ok_or_else(|| Error::DeviceError("GPU tensor passed to CPU backend".into()))?;
    let contiguous = tensor.contiguous(MemoryOrder::ColumnMajor);
    let data = contiguous
        .buffer()
        .as_slice()
        .expect("CPU tensor must remain CPU-backed after materialization");

    let tropical_data: Vec<T> = data.iter().map(|&v| T::from_inner(v)).collect();
    Tensor::<T>::from_slice(&tropical_data, tensor.dims(), MemoryOrder::ColumnMajor)
        .map_err(|e| Error::InvalidArgument(format!("{e}")))
}

/// Extract inner values from a tropical tensor to a standard real tensor.
///
/// # Examples
///
/// ```ignore
/// use tenferro_tropical::MaxPlus;
/// use tenferro_tropical::ad::extract_inner;
/// use tenferro_tensor::{Tensor, MemoryOrder};
///
/// let t = Tensor::<MaxPlus<f64>>::from_slice(
///     &[MaxPlus(1.0), MaxPlus(2.0)], &[2], MemoryOrder::ColumnMajor,
/// ).unwrap();
/// let inner = extract_inner::<MaxPlus<f64>>(&t).unwrap();
/// ```
pub fn extract_inner<T: TropicalScalar>(tensor: &Tensor<T>) -> Result<Tensor<T::Inner>> {
    tensor
        .buffer()
        .as_slice()
        .ok_or_else(|| Error::DeviceError("GPU tensor passed to CPU backend".into()))?;
    let contiguous = tensor.contiguous(MemoryOrder::ColumnMajor);
    let data = contiguous
        .buffer()
        .as_slice()
        .expect("CPU tensor must remain CPU-backed after materialization");

    let inner_data: Vec<T::Inner> = data.iter().map(|v| v.inner()).collect();
    Tensor::<T::Inner>::from_slice(&inner_data, tensor.dims(), MemoryOrder::ColumnMajor)
        .map_err(|e| Error::InvalidArgument(format!("{e}")))
}

#[cfg(test)]
mod tests;
