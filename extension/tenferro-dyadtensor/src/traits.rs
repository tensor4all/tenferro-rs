use std::hash::Hash;

use tenferro_algebra::Scalar;

use crate::core::DynTensorTyped;
use crate::structured::StructuredTensor;
use crate::{AdMode, AdScalar, AdTensor, AdValue, DiffPolicy, DynScalar, Result};

/// AD rule method result type alias.
///
/// # Examples
///
/// ```rust
/// use tenferro_dyadtensor::{AdResult, Error};
///
/// let ok: AdResult<()> = Ok(());
/// let err: AdResult<()> = Err(Error::InvalidAdTensor {
///     message: "demo".into(),
/// });
/// assert!(ok.is_ok());
/// assert!(err.is_err());
/// ```
pub type AdResult<T> = Result<T>;

/// Trait bound for index labels used in contraction/factorization APIs.
///
/// # Examples
///
/// ```rust
/// use tenferro_dyadtensor::IndexLike;
///
/// fn accepts_index<I: IndexLike>(_index: I) {}
/// accepts_index(3_u32);
/// ```
pub trait IndexLike: Clone + Eq + Hash {}

impl<T> IndexLike for T where T: Clone + Eq + Hash {}

/// A value that exposes AD mode plus primal/tangent views.
///
/// # Examples
///
/// ```rust
/// use tenferro_dyadtensor::{AdMode, AdValue, Differentiable};
///
/// fn mode_of<V: Differentiable>(value: &V) -> AdMode {
///     value.mode()
/// }
///
/// let x = AdValue::primal(2.0_f64);
/// assert_eq!(mode_of(&x), AdMode::Primal);
/// ```
pub trait Differentiable: Clone {
    /// Underlying primal payload type.
    type Primal: Clone;

    /// Returns the AD mode.
    fn mode(&self) -> AdMode;

    /// Returns a reference to the primal payload.
    fn primal_ref(&self) -> &Self::Primal;

    /// Returns a reference to the tangent payload when available.
    fn tangent_ref(&self) -> Option<&Self::Primal>;
}

impl<T: Clone> Differentiable for AdValue<T> {
    type Primal = T;

    fn mode(&self) -> AdMode {
        self.mode()
    }

    fn primal_ref(&self) -> &T {
        self.primal_ref()
    }

    fn tangent_ref(&self) -> Option<&T> {
        self.tangent_ref()
    }
}

impl<T: Clone> Differentiable for AdScalar<T> {
    type Primal = T;

    fn mode(&self) -> AdMode {
        self.mode()
    }

    fn primal_ref(&self) -> &T {
        self.primal()
    }

    fn tangent_ref(&self) -> Option<&T> {
        self.tangent()
    }
}

impl<T: Clone + Scalar + DynTensorTyped> Differentiable for AdTensor<T> {
    type Primal = StructuredTensor<T>;

    fn mode(&self) -> AdMode {
        self.mode()
    }

    fn primal_ref(&self) -> &StructuredTensor<T> {
        self.structured_primal()
    }

    fn tangent_ref(&self) -> Option<&StructuredTensor<T>> {
        self.structured_tangent()
    }
}

/// Allowed index-pair restrictions for contraction planning.
///
/// # Examples
///
/// ```rust
/// use tenferro_dyadtensor::AllowedPairs;
///
/// let pairs = AllowedPairs { pairs: &[(0, 1), (2, 3)] };
/// assert_eq!(pairs.pairs.len(), 2);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct AllowedPairs<'a> {
    /// Allowed input-index pairs.
    pub pairs: &'a [(usize, usize)],
}

/// Factorization options for generic tensor-kernel APIs.
///
/// # Examples
///
/// ```rust
/// use tenferro_dyadtensor::{DiffPolicy, FactorizeOptions};
///
/// let opts = FactorizeOptions {
///     max_rank: Some(32),
///     diff_policy: DiffPolicy::StopGradient,
/// };
/// assert_eq!(opts.max_rank, Some(32));
/// ```
#[derive(Debug, Clone)]
pub struct FactorizeOptions {
    /// Optional truncation rank.
    pub max_rank: Option<usize>,
    /// Differentiation behavior at non-smooth points.
    pub diff_policy: DiffPolicy,
}

impl Default for FactorizeOptions {
    fn default() -> Self {
        Self {
            max_rank: None,
            diff_policy: DiffPolicy::StopGradient,
        }
    }
}

/// Generic factorization output container.
///
/// # Examples
///
/// ```rust
/// use tenferro_dyadtensor::FactorizeResult;
///
/// let result = FactorizeResult { left: 1_i32, right: 2_i32 };
/// assert_eq!(result.left + result.right, 3);
/// ```
#[derive(Debug, Clone)]
pub struct FactorizeResult<T> {
    /// Left factor.
    pub left: T,
    /// Right factor.
    pub right: T,
}

/// Numeric tensor-kernel boundary for contraction/factorization operations.
///
/// # Examples
///
/// ```rust
/// use tenferro_dyadtensor::{AllowedPairs, DynScalar, FactorizeOptions, FactorizeResult, Result, TensorKernel};
///
/// #[derive(Clone)]
/// struct DummyKernel;
///
/// impl TensorKernel for DummyKernel {
///     type Index = u8;
///
///     fn contract(_tensors: &[&Self], _allowed: AllowedPairs<'_>) -> Result<Self> {
///         Ok(Self)
///     }
///
///     fn factorize(
///         &self,
///         _left_inds: &[Self::Index],
///         _options: &FactorizeOptions,
///     ) -> Result<FactorizeResult<Self>> {
///         Ok(FactorizeResult { left: Self, right: Self })
///     }
///
///     fn axpby(&self, _a: DynScalar, _other: &Self, _b: DynScalar) -> Result<Self> {
///         Ok(Self)
///     }
///
///     fn scale(&self, _a: DynScalar) -> Result<Self> {
///         Ok(Self)
///     }
///
///     fn inner_product(&self, _other: &Self) -> Result<DynScalar> {
///         Ok(DynScalar::F64(0.0))
///     }
/// }
///
/// let x = DummyKernel;
/// let _ = x.scale(DynScalar::F64(2.0)).unwrap();
/// ```
pub trait TensorKernel: Clone {
    /// Index label type used by this kernel.
    type Index: IndexLike;

    /// Contract a set of tensors under index-pair constraints.
    fn contract(tensors: &[&Self], allowed: AllowedPairs<'_>) -> Result<Self>;

    /// Factorize a tensor into two factors.
    fn factorize(
        &self,
        left_inds: &[Self::Index],
        options: &FactorizeOptions,
    ) -> Result<FactorizeResult<Self>>;

    /// Affine combination `a * self + b * other`.
    fn axpby(&self, a: DynScalar, other: &Self, b: DynScalar) -> Result<Self>;

    /// Scalar multiply.
    fn scale(&self, a: DynScalar) -> Result<Self>;

    /// Inner product result.
    fn inner_product(&self, other: &Self) -> Result<DynScalar>;
}

/// Operation-level AD rules (`rrule`, `frule`, `hvp`).
///
/// # Examples
///
/// ```rust
/// use tenferro_dyadtensor::{AdResult, AdValue, Differentiable, OpRule, Result};
///
/// struct IdentityRule;
///
/// impl OpRule<AdValue<f64>> for IdentityRule {
///     fn eval(&self, inputs: &[&AdValue<f64>]) -> Result<AdValue<f64>> {
///         Ok((*inputs[0]).clone())
///     }
///
///     fn rrule(
///         &self,
///         _inputs: &[&AdValue<f64>],
///         _out: &AdValue<f64>,
///         cotangent: &f64,
///     ) -> AdResult<Vec<f64>> {
///         Ok(vec![*cotangent])
///     }
///
///     fn frule(
///         &self,
///         _inputs: &[&AdValue<f64>],
///         tangents: &[Option<&f64>],
///     ) -> AdResult<f64> {
///         Ok(tangents[0].copied().unwrap_or(0.0))
///     }
///
///     fn hvp(
///         &self,
///         _inputs: &[&AdValue<f64>],
///         cotangent: &f64,
///         cotangent_tangent: Option<&f64>,
///         _input_tangents: &[Option<&f64>],
///     ) -> AdResult<Vec<(f64, f64)>> {
///         Ok(vec![(*cotangent, cotangent_tangent.copied().unwrap_or(0.0))])
///     }
/// }
///
/// let x = AdValue::primal(2.0_f64);
/// let rule = IdentityRule;
/// let y = rule.eval(&[&x]).unwrap();
/// assert_eq!(y.primal_ref(), &2.0);
/// ```
pub trait OpRule<V: Differentiable> {
    /// Compute primal output.
    fn eval(&self, inputs: &[&V]) -> Result<V>;

    /// Reverse-mode pullback.
    fn rrule(&self, inputs: &[&V], out: &V, cotangent: &V::Primal) -> AdResult<Vec<V::Primal>>;

    /// Forward-mode pushforward.
    fn frule(&self, inputs: &[&V], tangents: &[Option<&V::Primal>]) -> AdResult<V::Primal>;

    /// Hessian-vector product.
    fn hvp(
        &self,
        inputs: &[&V],
        cotangent: &V::Primal,
        cotangent_tangent: Option<&V::Primal>,
        input_tangents: &[Option<&V::Primal>],
    ) -> AdResult<Vec<(V::Primal, V::Primal)>>;
}
