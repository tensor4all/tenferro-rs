use std::cell::Cell;

use crate::{Error, Result, Tensor};

thread_local! {
    static DUAL_LEVEL_ACTIVE: Cell<bool> = const { Cell::new(false) };
}

/// Scoped forward-mode builder.
///
/// Instances are created only inside [`dual_level`].
///
/// # Examples
///
/// ```ignore
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
    /// ```ignore
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
        match (primal.as_f32(), tangent.as_f32()) {
            (Some(primal), Some(tangent)) => Tensor::new_forward(
                primal.structured_primal().clone(),
                tangent.structured_primal().clone(),
            ),
            _ => match (primal.as_f64(), tangent.as_f64()) {
                (Some(primal), Some(tangent)) => Tensor::new_forward(
                    primal.structured_primal().clone(),
                    tangent.structured_primal().clone(),
                ),
                _ => match (primal.as_c32(), tangent.as_c32()) {
                    (Some(primal), Some(tangent)) => Tensor::new_forward(
                        primal.structured_primal().clone(),
                        tangent.structured_primal().clone(),
                    ),
                    _ => match (primal.as_c64(), tangent.as_c64()) {
                        (Some(primal), Some(tangent)) => Tensor::new_forward(
                            primal.structured_primal().clone(),
                            tangent.structured_primal().clone(),
                        ),
                        _ => Err(Error::InvalidAdTensor {
                            message: format!(
                                "forward_ad::make_dual requires matching dtypes, got primal {:?} and tangent {:?}",
                                primal.scalar_type(),
                                tangent.scalar_type()
                            ),
                        }),
                    },
                },
            },
        }
    }

    /// Unpacks a dual tensor into its primal value and optional tangent.
    ///
    /// # Examples
    ///
    /// ```ignore
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
        Ok((
            Tensor::from(value.primal_snapshot()),
            value.tangent_snapshot().map(Tensor::from),
        ))
    }
}

/// Runs a scoped forward-mode computation.
///
/// Nested forward scopes are rejected.
///
/// # Examples
///
/// ```ignore
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
