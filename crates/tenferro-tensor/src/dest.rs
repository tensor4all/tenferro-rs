//! Output destinations for `_into` operations (the #2552 `Dest` grammar).
//!
//! [`Dest`] states where an operation writes and how the result combines with
//! the previous contents. Conjugation is operation configuration and lives in
//! the operation's options, never here.

use crate::{
    Complex32, Complex64, ContractionScalar, DType, DotGeneralAccumulation, Error, Tensor,
    TensorViewMut, TensorWrite, ValidationError,
};

/// Whether an operand enters an operation conjugated. Conjugation is operation
/// configuration, so it lives in the operation's options, never in [`Dest`].
///
/// # Examples
///
/// ```
/// use tenferro_tensor::Conj;
///
/// assert_eq!(Conj::default(), Conj::No);
/// assert!(Conj::Yes.is_yes());
/// assert!(!Conj::No.is_yes());
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Conj {
    /// Use the operand as it is.
    #[default]
    No,
    /// Use the complex conjugate of the operand.
    Yes,
}

impl Conj {
    /// Return `true` for [`Conj::Yes`].
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::Conj;
    ///
    /// assert!(Conj::Yes.is_yes());
    /// ```
    pub fn is_yes(self) -> bool {
        matches!(self, Self::Yes)
    }
}

/// Options of `dot_general`: which operand enters conjugated.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::{Conj, DotGeneralOptions};
///
/// let options = DotGeneralOptions::default().lhs(Conj::Yes);
/// assert_eq!((options.lhs, options.rhs), (Conj::Yes, Conj::No));
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct DotGeneralOptions {
    /// Conjugate the left operand. The default is [`Conj::No`].
    pub lhs: Conj,
    /// Conjugate the right operand. The default is [`Conj::No`].
    pub rhs: Conj,
}

impl DotGeneralOptions {
    /// Return options with the requested left-operand conjugation.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Conj, DotGeneralOptions};
    ///
    /// assert_eq!(DotGeneralOptions::default().lhs(Conj::Yes).lhs, Conj::Yes);
    /// ```
    pub fn lhs(mut self, lhs: Conj) -> Self {
        self.lhs = lhs;
        self
    }

    /// Return options with the requested right-operand conjugation.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Conj, DotGeneralOptions};
    ///
    /// assert_eq!(DotGeneralOptions::default().rhs(Conj::Yes).rhs, Conj::Yes);
    /// ```
    pub fn rhs(mut self, rhs: Conj) -> Self {
        self.rhs = rhs;
        self
    }
}

/// Which [`Dest`] variant a destination is, without its target.
///
/// The operation registry uses this to declare the accepted modes.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::{Dest, DestMode, Tensor};
///
/// let mut t = Tensor::from_vec_col_major(vec![1], vec![0.0_f64])?;
/// let dest: Dest<tenferro_tensor::TensorWrite<'_>> = Dest::accumulate(&mut t);
/// assert_eq!(dest.mode(), DestMode::Accumulate);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DestMode {
    /// [`Dest::Overwrite`].
    Overwrite,
    /// [`Dest::Accumulate`].
    Accumulate,
    /// [`Dest::Scaled`].
    Scaled,
}

/// Where an `_into` operation writes, and how it combines with what is there.
///
/// `W` is `TensorWrite<'a>` for dtype-erased outputs and
/// `TypedTensorWrite<'a, T>` for typed outputs.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::{Dest, DestMode, Tensor, TensorWrite};
///
/// let mut out = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
/// let dest: Dest<TensorWrite<'_>> = Dest::scaled(&mut out, 2.0_f64, 0.5_f64);
/// assert_eq!(dest.mode(), DestMode::Scaled);
/// assert_eq!(dest.target().shape(), &[2]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub enum Dest<W> {
    /// `out = op(...)`; the previous contents are never read.
    Overwrite(W),
    /// `out = op(...) + out`.
    Accumulate(W),
    /// `out = alpha * op(...) + beta * out`; with `beta == 0` the previous
    /// contents are never read (NaN or -0 in `out` does not leak).
    Scaled {
        /// The destination.
        out: W,
        /// Scale applied to the operation result.
        alpha: ContractionScalar,
        /// Scale applied to the previous contents of `out`.
        beta: ContractionScalar,
    },
}

impl<W> Dest<W> {
    /// Build an [`Dest::Overwrite`] destination.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Dest, DestMode, Tensor, TensorWrite};
    ///
    /// let mut t = Tensor::from_vec_col_major(vec![1], vec![0.0_f64])?;
    /// let dest: Dest<TensorWrite<'_>> = Dest::overwrite(&mut t);
    /// assert_eq!(dest.mode(), DestMode::Overwrite);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn overwrite(out: impl Into<W>) -> Self {
        Self::Overwrite(out.into())
    }

    /// Build an [`Dest::Accumulate`] destination.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Dest, DestMode, Tensor, TensorWrite};
    ///
    /// let mut t = Tensor::from_vec_col_major(vec![1], vec![0.0_f64])?;
    /// let dest: Dest<TensorWrite<'_>> = Dest::accumulate(&mut t);
    /// assert_eq!(dest.mode(), DestMode::Accumulate);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn accumulate(out: impl Into<W>) -> Self {
        Self::Accumulate(out.into())
    }

    /// Build a [`Dest::Scaled`] destination, `out = alpha * op + beta * out`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Dest, DestMode, Tensor, TensorWrite};
    ///
    /// let mut t = Tensor::from_vec_col_major(vec![1], vec![0.0_f64])?;
    /// let dest: Dest<TensorWrite<'_>> = Dest::scaled(&mut t, 1.0_f64, 0.0_f64);
    /// assert_eq!(dest.mode(), DestMode::Scaled);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn scaled(
        out: impl Into<W>,
        alpha: impl Into<ContractionScalar>,
        beta: impl Into<ContractionScalar>,
    ) -> Self {
        Self::Scaled {
            out: out.into(),
            alpha: alpha.into(),
            beta: beta.into(),
        }
    }

    /// Return which variant this destination is, without its target.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Dest, DestMode};
    ///
    /// let dest = Dest::Overwrite(());
    /// assert_eq!(dest.mode(), DestMode::Overwrite);
    /// ```
    pub fn mode(&self) -> DestMode {
        match self {
            Self::Overwrite(_) => DestMode::Overwrite,
            Self::Accumulate(_) => DestMode::Accumulate,
            Self::Scaled { .. } => DestMode::Scaled,
        }
    }

    /// Borrow the destination.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::Dest;
    ///
    /// let dest = Dest::Accumulate(7);
    /// assert_eq!(*dest.target(), 7);
    /// ```
    pub fn target(&self) -> &W {
        match self {
            Self::Overwrite(out) | Self::Accumulate(out) | Self::Scaled { out, .. } => out,
        }
    }

    /// Consume the destination and return its target.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::Dest;
    ///
    /// assert_eq!(Dest::Overwrite(7).into_target(), 7);
    /// ```
    pub fn into_target(self) -> W {
        match self {
            Self::Overwrite(out) | Self::Accumulate(out) | Self::Scaled { out, .. } => out,
        }
    }

    /// Map this destination to the backend accumulation contract.
    ///
    /// Owner-scoped bridge owned by `tenferro-tensor` for the #2552
    /// `Dest` to [`DotGeneralAccumulation`] mapping; it is not user API.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Conj, ContractionScalar, DType, Dest};
    ///
    /// let (out, accum) = Dest::Accumulate(5).into_accumulation(DType::F64, Conj::Yes, Conj::No)?;
    /// assert_eq!(out, 5);
    /// assert_eq!(accum.beta, ContractionScalar::F64(1.0));
    /// assert!(accum.lhs_conj && !accum.rhs_conj);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with a [`ValidationError::DTypeMismatch`]
    /// source when `dtype` has no contraction scalar or `alpha`/`beta` of a
    /// scaled destination do not have `dtype`.
    #[doc(hidden)]
    pub fn into_accumulation(
        self,
        dtype: DType,
        lhs: Conj,
        rhs: Conj,
    ) -> crate::Result<(W, DotGeneralAccumulation)> {
        const OP: &str = "Dest::into_accumulation";
        let (out, mut accum) = match self {
            Self::Overwrite(out) => (out, DotGeneralAccumulation::overwrite(dtype)?),
            Self::Accumulate(out) => (out, DotGeneralAccumulation::add_to(dtype)?),
            Self::Scaled { out, alpha, beta } => {
                for scalar in [alpha, beta] {
                    if scalar.dtype() != dtype {
                        return Err(Error::validation(
                            OP,
                            ValidationError::DTypeMismatch {
                                expected: dtype,
                                actual: scalar.dtype(),
                            },
                        ));
                    }
                }
                let accum = DotGeneralAccumulation {
                    lhs_conj: false,
                    rhs_conj: false,
                    alpha,
                    beta,
                };
                (out, accum)
            }
        };
        accum.lhs_conj = lhs.is_yes();
        accum.rhs_conj = rhs.is_yes();
        Ok((out, accum))
    }
}

impl<'a> From<&'a mut Tensor> for TensorWrite<'a> {
    fn from(tensor: &'a mut Tensor) -> Self {
        TensorWrite::from_tensor(tensor)
    }
}

impl<'a> From<TensorViewMut<'a>> for TensorWrite<'a> {
    fn from(view: TensorViewMut<'a>) -> Self {
        TensorWrite::from_view(view)
    }
}

impl From<f32> for ContractionScalar {
    fn from(value: f32) -> Self {
        Self::F32(value)
    }
}

impl From<f64> for ContractionScalar {
    fn from(value: f64) -> Self {
        Self::F64(value)
    }
}

impl From<Complex32> for ContractionScalar {
    fn from(value: Complex32) -> Self {
        Self::C32(value)
    }
}

impl From<Complex64> for ContractionScalar {
    fn from(value: Complex64) -> Self {
        Self::C64(value)
    }
}

#[cfg(test)]
mod tests;
