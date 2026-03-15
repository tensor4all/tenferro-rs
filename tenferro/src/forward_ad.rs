use std::cell::Cell;

use crate::{AdTensor, Error, Result, Tensor};

thread_local! {
    static DUAL_LEVEL_ACTIVE: Cell<bool> = const { Cell::new(false) };
}

/// Scoped forward-mode builder.
///
/// Instances are created only inside [`dual_level`].
///
/// # Examples
///
/// ```rust
/// use tenferro::{forward_ad, Tensor};
///
/// let x = Tensor::from_slice(&[1.0_f64, 2.0], &[2]).unwrap();
/// let dx = Tensor::from_slice(&[0.5_f64, -0.25], &[2]).unwrap();
/// let (_primal, tangent) = forward_ad::dual_level(|fw| {
///     let dual = fw.make_dual(&x, &dx)?;
///     fw.unpack_dual(&dual)
/// })
/// .unwrap();
/// assert!(tangent.is_some());
/// ```
pub struct DualLevel;

impl DualLevel {
    /// Creates a dual tensor from a primal tensor and its tangent seed.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro::{forward_ad, Tensor};
    ///
    /// let x = Tensor::from_slice(&[1.0_f64], &[1]).unwrap();
    /// let dx = Tensor::from_slice(&[0.5_f64], &[1]).unwrap();
    /// let (_primal, tangent) = forward_ad::dual_level(|fw| {
    ///     let dual = fw.make_dual(&x, &dx)?;
    ///     fw.unpack_dual(&dual)
    /// })
    /// .unwrap();
    /// assert!(tangent.is_some());
    /// ```
    pub fn make_dual(&self, primal: &Tensor, tangent: &Tensor) -> Result<Tensor> {
        match (primal.detach(), tangent.detach()) {
            (Tensor::F32(primal), Tensor::F32(tangent)) => {
                Ok(Tensor::F32(AdTensor::new_forward(
                    primal.structured_primal().clone(),
                    tangent.structured_primal().clone(),
                )?))
            }
            (Tensor::F64(primal), Tensor::F64(tangent)) => {
                Ok(Tensor::F64(AdTensor::new_forward(
                    primal.structured_primal().clone(),
                    tangent.structured_primal().clone(),
                )?))
            }
            (Tensor::C32(primal), Tensor::C32(tangent)) => {
                Ok(Tensor::C32(AdTensor::new_forward(
                    primal.structured_primal().clone(),
                    tangent.structured_primal().clone(),
                )?))
            }
            (Tensor::C64(primal), Tensor::C64(tangent)) => {
                Ok(Tensor::C64(AdTensor::new_forward(
                    primal.structured_primal().clone(),
                    tangent.structured_primal().clone(),
                )?))
            }
            (primal, tangent) => Err(Error::InvalidAdTensor {
                message: format!(
                    "forward_ad::make_dual requires matching dtypes, got primal {:?} and tangent {:?}",
                    primal.scalar_type(),
                    tangent.scalar_type()
                ),
            }),
        }
    }

    /// Unpacks a dual tensor into its primal value and optional tangent.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro::{forward_ad, Tensor};
    ///
    /// let x = Tensor::from_slice(&[1.0_f64], &[1]).unwrap();
    /// let dx = Tensor::from_slice(&[0.5_f64], &[1]).unwrap();
    /// let (primal, tangent) = forward_ad::dual_level(|fw| {
    ///     let dual = fw.make_dual(&x, &dx)?;
    ///     fw.unpack_dual(&dual)
    /// })
    /// .unwrap();
    /// assert_eq!(primal.dims(), &[1]);
    /// assert!(tangent.is_some());
    /// ```
    pub fn unpack_dual(&self, value: &Tensor) -> Result<(Tensor, Option<Tensor>)> {
        Ok(match value {
            Tensor::F32(value) => (
                Tensor::F32(AdTensor::new_primal(value.structured_primal().clone())),
                value
                    .structured_tangent()
                    .cloned()
                    .map(|tangent| Tensor::F32(AdTensor::new_primal(tangent))),
            ),
            Tensor::F64(value) => (
                Tensor::F64(AdTensor::new_primal(value.structured_primal().clone())),
                value
                    .structured_tangent()
                    .cloned()
                    .map(|tangent| Tensor::F64(AdTensor::new_primal(tangent))),
            ),
            Tensor::C32(value) => (
                Tensor::C32(AdTensor::new_primal(value.structured_primal().clone())),
                value
                    .structured_tangent()
                    .cloned()
                    .map(|tangent| Tensor::C32(AdTensor::new_primal(tangent))),
            ),
            Tensor::C64(value) => (
                Tensor::C64(AdTensor::new_primal(value.structured_primal().clone())),
                value
                    .structured_tangent()
                    .cloned()
                    .map(|tangent| Tensor::C64(AdTensor::new_primal(tangent))),
            ),
        })
    }
}

/// Runs a scoped forward-mode computation.
///
/// Nested forward scopes are rejected.
///
/// # Examples
///
/// ```rust
/// use tenferro::{forward_ad, set_default_runtime, RuntimeContext, Tensor};
/// use tenferro_prims::CpuContext;
///
/// let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
/// let x = Tensor::from_slice(&[0.0_f64, 1.0], &[2]).unwrap();
/// let dx = Tensor::from_slice(&[0.5_f64, -0.25], &[2]).unwrap();
/// let (_primal, tangent) = forward_ad::dual_level(|fw| {
///     let dual = fw.make_dual(&x, &dx)?;
///     let out = dual.exp()?;
///     fw.unpack_dual(&out)
/// })
/// .unwrap();
/// assert!(tangent.is_some());
/// ```
pub fn dual_level<R>(f: impl FnOnce(&DualLevel) -> Result<R>) -> Result<R> {
    DUAL_LEVEL_ACTIVE.with(|active| {
        if active.get() {
            return Err(Error::UnsupportedAdOp {
                op: "forward_ad::dual_level(nested)",
            });
        }
        active.set(true);
        let result = f(&DualLevel);
        active.set(false);
        result
    })
}
