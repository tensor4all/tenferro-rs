//! Prototype 2: einsum destination (`Dest`) and spec (`EinsumSpec`) over the
//! current concrete plan API.
//!
//! Grammar under test (tenferro-rs#2552):
//! - `[inputs].einsum(spec, exec)` = defaults + `Allow`
//! - `[inputs].einsum_with(spec, &opts, exec, policy)`
//! - `[inputs].einsum_into(spec, &opts, exec, policy, dest)`
//!
//! Everything here is an adapter over `ConcreteEinsumPlan`; nothing in the
//! einsum crate was changed. See `findings/p2.md` for what did not fit.

use num_complex::{Complex32, Complex64};
use tenferro_einsum::{
    ConcreteEinsumPlan, EinsumNotation, EinsumOptimize, EinsumSubscripts, Error as EinsumError,
};
use tenferro_tensor::{
    ContractionScalar, DotGeneralAccumulation, Engine, Error, Result, Tensor, TensorRead,
    TensorWrite,
};

use crate::policy::MaterializationPolicy;

const OP: &str = "einsum";

/// One spec argument that accepts text, parsed notation, or integer index
/// lists. Construction never validates: `From<&str>` only wraps the text, and
/// parsing/validation happen inside the op, so failures come back as errors.
#[derive(Debug, Clone)]
pub enum EinsumSpec<'a> {
    /// Text such as `"ij,jk->ik"`; parsed inside the op.
    Text(&'a str),
    /// Rank-resolved integer-label subscripts.
    Subscripts(&'a EinsumSubscripts),
    /// Rank-unresolved notation (ellipsis allowed); resolved inside the op.
    Notation(&'a EinsumNotation),
    /// Owned index lists, built by [`EinsumSpec::indices`].
    Indices(EinsumSubscripts),
}

impl<'a> EinsumSpec<'a> {
    /// Build a spec from per-input label lists and an output label list.
    #[must_use]
    pub fn indices(inputs: &[&[u32]], output: &[u32]) -> Self {
        Self::Indices(EinsumSubscripts::new(inputs, output))
    }
}

impl<'a> From<&'a str> for EinsumSpec<'a> {
    fn from(text: &'a str) -> Self {
        Self::Text(text)
    }
}
impl<'a> From<&'a EinsumSubscripts> for EinsumSpec<'a> {
    fn from(value: &'a EinsumSubscripts) -> Self {
        Self::Subscripts(value)
    }
}
impl<'a> From<&'a EinsumNotation> for EinsumSpec<'a> {
    fn from(value: &'a EinsumNotation) -> Self {
        Self::Notation(value)
    }
}

/// Semantic einsum options.
///
/// `optimize` mirrors the traced API's [`EinsumOptimize`], but the concrete plan
/// cannot honour anything except the default (`Auto`): see findings.
/// `conj` is per-input conjugation *configuration* (not destination state).
#[derive(Debug, Default)]
pub struct EinsumOptions {
    optimize: EinsumOptimize,
    conj: Vec<bool>,
}

impl EinsumOptions {
    /// Choose the contraction-order strategy.
    #[must_use]
    pub fn with_optimize(mut self, optimize: EinsumOptimize) -> Self {
        self.optimize = optimize;
        self
    }
    /// Conjugate input `i` iff `conj[i]` (missing entries mean `false`).
    #[must_use]
    pub fn with_conj(mut self, conj: impl Into<Vec<bool>>) -> Self {
        self.conj = conj.into();
        self
    }
}

/// Destination of `einsum_into`: `out = alpha * contraction + beta * out`.
#[derive(Debug)]
pub enum Dest<'a> {
    /// `alpha = 1, beta = 0`; the old contents of `out` are never read.
    Overwrite(TensorWrite<'a>),
    /// `alpha = 1, beta = 1`.
    Accumulate(TensorWrite<'a>),
    /// Explicit scalars; `beta == 0` keeps the no-read guarantee.
    Scaled {
        /// Destination.
        out: TensorWrite<'a>,
        /// Scale of the contraction.
        alpha: ContractionScalar,
        /// Scale of the old destination.
        beta: ContractionScalar,
    },
}

impl<'a> Dest<'a> {
    /// Overwrite destination.
    pub fn overwrite(out: TensorWrite<'a>) -> Self {
        Self::Overwrite(out)
    }
    /// Accumulating destination.
    pub fn accumulate(out: TensorWrite<'a>) -> Self {
        Self::Accumulate(out)
    }
    /// Scaled destination.
    pub fn scaled(out: TensorWrite<'a>, alpha: ContractionScalar, beta: ContractionScalar) -> Self {
        Self::Scaled { out, alpha, beta }
    }
}

/// Complex scalar helpers so callers need not spell `ContractionScalar`.
#[must_use]
pub fn c64(re: f64, im: f64) -> ContractionScalar {
    ContractionScalar::C64(Complex64::new(re, im))
}
/// Complex32 scalar helper.
#[must_use]
pub fn c32(re: f32, im: f32) -> ContractionScalar {
    ContractionScalar::C32(Complex32::new(re, im))
}

fn forbid_not_implemented(policy: MaterializationPolicy) -> Result<()> {
    match policy {
        MaterializationPolicy::Allow => Ok(()),
        MaterializationPolicy::Forbid => Err(Error::unsupported(
            OP,
            "MaterializationPolicy::Forbid enforcement is not implemented by any provider \
             (prototype 4 covers it)",
        )),
    }
}

fn to_tensor_error(e: EinsumError) -> Error {
    e.into_tensor_error(OP)
}

fn check_optimize(opts: &EinsumOptions) -> Result<()> {
    match &opts.optimize {
        EinsumOptimize::Auto(_) => Ok(()),
        other => Err(Error::unsupported(
            OP,
            format!(
                "ConcreteEinsumPlan always plans with ContractionTree::optimize; \
                 EinsumOptimize::{other:?} cannot be honoured by the concrete layer"
            ),
        )),
    }
}

fn prepare(inputs: &[TensorRead<'_>], spec: &EinsumSpec<'_>) -> Result<ConcreteEinsumPlan> {
    match spec {
        EinsumSpec::Text(s) => ConcreteEinsumPlan::prepare_read(inputs, s),
        EinsumSpec::Subscripts(s) => ConcreteEinsumPlan::prepare_read_subscripts(inputs, s),
        EinsumSpec::Notation(n) => ConcreteEinsumPlan::prepare_read_notation(inputs, n),
        EinsumSpec::Indices(s) => ConcreteEinsumPlan::prepare_read_subscripts(inputs, s),
    }
    .map_err(to_tensor_error)
}

fn conj_flags(inputs: &[TensorRead<'_>], opts: &EinsumOptions) -> Result<(bool, bool)> {
    let flag = |i: usize| opts.conj.get(i).copied().unwrap_or(false);
    if opts.conj.len() > inputs.len() {
        return Err(Error::invalid_argument(
            OP,
            "conj",
            format!("{} flags for {} inputs", opts.conj.len(), inputs.len()),
        ));
    }
    // DotGeneralAccumulation has exactly two flags: operands 2.. cannot be named.
    if (2..inputs.len()).any(flag) {
        return Err(Error::unsupported(
            OP,
            "DotGeneralAccumulation only carries lhs/rhs conjugation; \
             conjugation of the third or later einsum operand is not expressible",
        ));
    }
    Ok((flag(0), flag(1)))
}

/// Extension trait on the slice of inputs (the concrete receiver for variadic ops).
pub trait EinsumInputs {
    /// Defaults + `Allow`.
    fn einsum<'s>(&self, spec: impl Into<EinsumSpec<'s>>, exec: &mut dyn Engine) -> Result<Tensor>;
    /// Explicit options and policy; returns a fresh tensor.
    fn einsum_with<'s>(
        &self,
        spec: impl Into<EinsumSpec<'s>>,
        opts: &EinsumOptions,
        exec: &mut dyn Engine,
        policy: MaterializationPolicy,
    ) -> Result<Tensor>;
    /// Explicit options, policy and destination.
    fn einsum_into<'s>(
        &self,
        spec: impl Into<EinsumSpec<'s>>,
        opts: &EinsumOptions,
        exec: &mut dyn Engine,
        policy: MaterializationPolicy,
        dest: Dest<'_>,
    ) -> Result<()>;
}

impl EinsumInputs for [TensorRead<'_>] {
    fn einsum<'s>(&self, spec: impl Into<EinsumSpec<'s>>, exec: &mut dyn Engine) -> Result<Tensor> {
        self.einsum_with(
            spec,
            &EinsumOptions::default(),
            exec,
            MaterializationPolicy::Allow,
        )
    }

    fn einsum_with<'s>(
        &self,
        spec: impl Into<EinsumSpec<'s>>,
        opts: &EinsumOptions,
        exec: &mut dyn Engine,
        policy: MaterializationPolicy,
    ) -> Result<Tensor> {
        forbid_not_implemented(policy)?;
        check_optimize(opts)?;
        if opts.conj.iter().any(|c| *c) {
            return Err(Error::unsupported(
                OP,
                "conjugation on a value-returning einsum: ConcreteEinsumPlan::execute_read has no \
                 conjugation input; only the *_into_accum entry carries it",
            ));
        }
        let plan = prepare(self, &spec.into())?;
        plan.execute_read(self, exec).map_err(to_tensor_error)
    }

    fn einsum_into<'s>(
        &self,
        spec: impl Into<EinsumSpec<'s>>,
        opts: &EinsumOptions,
        exec: &mut dyn Engine,
        policy: MaterializationPolicy,
        dest: Dest<'_>,
    ) -> Result<()> {
        forbid_not_implemented(policy)?;
        check_optimize(opts)?;
        let (lhs_conj, rhs_conj) = conj_flags(self, opts)?;
        let plan = prepare(self, &spec.into())?;
        let first = self.first().ok_or_else(|| {
            Error::invalid_argument(OP, "inputs", "einsum requires at least one input")
        })?;
        let dtype = first.dtype();
        let (mut accum, out) = match dest {
            Dest::Overwrite(out) => {
                if !lhs_conj && !rhs_conj {
                    // Dedicated overwrite entry: no accumulation path at all.
                    return plan
                        .execute_read_into(self, exec, out)
                        .map_err(to_tensor_error);
                }
                (DotGeneralAccumulation::overwrite(dtype)?, out)
            }
            Dest::Accumulate(out) => (DotGeneralAccumulation::add_to(dtype)?, out),
            Dest::Scaled { out, alpha, beta } => {
                (DotGeneralAccumulation::scaled(alpha, beta)?, out)
            }
        };
        accum.lhs_conj = lhs_conj;
        accum.rhs_conj = rhs_conj;
        plan.execute_read_into_accum(self, exec, accum, out)
            .map_err(to_tensor_error)
    }
}
