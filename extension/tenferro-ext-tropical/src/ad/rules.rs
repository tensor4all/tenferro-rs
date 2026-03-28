use tenferro_algebra::{HasAlgebra, Semiring};
use tenferro_device::{Error, Result};
use tenferro_einsum::Subscripts;
use tenferro_prims::TensorSemiringCore;
use tenferro_tensor::Tensor;

use crate::argmax::ArgmaxTracker;

use super::backward::tropical_backward;
use super::common::contracted_modes;
use super::forward::{tropical_forward_tangent, tropical_forward_with_argmax};
use super::TropicalScalar;

/// Reverse-mode rule (rrule) for tropical einsum.
///
/// Given a tropical einsum operation and a cotangent (in standard reals),
/// computes gradients for each input operand (also in standard reals).
///
/// # Examples
///
/// ```ignore
/// use tenferro_ext_tropical::ad::tropical_einsum_rrule;
/// use tenferro_ext_tropical::{MaxPlus, MaxPlusAlgebra};
/// use tenferro_prims::{CpuBackend, CpuContext};
/// use tenferro_tensor::{MemoryOrder, Tensor};
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
    validate_operand_count(operands.len(), "tropical_einsum_rrule")?;
    let contracted = contracted_modes(&subs);
    let (output, tracker) = tropical_forward_with_argmax(operands, &subs, &contracted)?;

    if cotangent.dims() != output.dims() {
        return Err(Error::InvalidArgument(format!(
            "cotangent shape mismatch: expected {:?}, got {:?}",
            output.dims(),
            cotangent.dims()
        )));
    }

    tropical_backward(operands, cotangent, &tracker, &subs, &contracted)
}

/// Forward-mode rule (frule) for tropical einsum.
///
/// Given tropical primals and optional standard-real tangents, compute the
/// output tangent by routing each tangent contribution through the winner
/// selected during the tropical forward pass.
///
/// # Examples
///
/// ```ignore
/// use tenferro_ext_tropical::ad::tropical_einsum_frule;
/// use tenferro_ext_tropical::{MaxPlus, MaxPlusAlgebra};
/// use tenferro_prims::{CpuBackend, CpuContext};
/// use tenferro_tensor::{MemoryOrder, Tensor};
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
    validate_operand_count(primals.len(), "tropical_einsum_frule")?;
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

    let contracted = contracted_modes(&subs);
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

pub(crate) fn tracked_forward<T: TropicalScalar>(
    operands: &[&Tensor<T>],
    subs: &Subscripts,
    contracted: &[u32],
) -> Result<(Tensor<T>, ArgmaxTracker)> {
    tropical_forward_with_argmax(operands, subs, contracted)
}

fn validate_operand_count(count: usize, api_name: &str) -> Result<()> {
    if count == 0 || count > 2 {
        return Err(Error::InvalidArgument(format!(
            "{api_name} supports 1 or 2 operands"
        )));
    }
    Ok(())
}
