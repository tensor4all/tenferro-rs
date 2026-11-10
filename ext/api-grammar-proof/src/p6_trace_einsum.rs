//! Prototype 6: owner-receiver einsum on `TraceContext` over `TracedTensor`.
//!
//! The grammar (tenferro-rs#2552, decision 4) puts the owner in the receiver
//! position for variadic ops: `trace.einsum_with(&[a, b], spec, &opts)`. The
//! existing `TraceContextEinsumExt` already owns the names `einsum` /
//! `einsum_with` (taking `&mut self` and `&[TraceValue]`), so this prototype
//! uses the name [`TraceEinsum::trace_einsum_with`] to avoid an ambiguous
//! method call while both traits are in scope (see `findings/p6.md`).

use std::borrow::Borrow;

use tenferro_einsum::{
    EinsumOptimize, Error as EinsumError, TraceContextEinsumExt, EINSUM_EXTENSION_FAMILY_ID,
};
use tenferro_runtime::{Error, ErrorPhase, Result, TraceContext, TraceValue, TracedTensor};

use crate::p2_einsum_dest::EinsumSpec;

/// Options for traced einsum. `optimize` is the only semantic option.
#[derive(Debug, Default)]
pub struct TraceEinsumOptions {
    optimize: EinsumOptimize,
}

impl TraceEinsumOptions {
    /// Choose the contraction-order strategy.
    #[must_use]
    pub fn with_optimize(mut self, optimize: EinsumOptimize) -> Self {
        self.optimize = optimize;
        self
    }
}

/// `EinsumOptimize` is neither `Clone` nor passed by reference by the current
/// traced API, so an `&opts` signature has to duplicate it by hand.
fn duplicate(optimize: &EinsumOptimize) -> Result<EinsumOptimize> {
    Ok(match optimize {
        EinsumOptimize::Auto(o) => EinsumOptimize::Auto(o.clone()),
        EinsumOptimize::False => EinsumOptimize::False,
        EinsumOptimize::Nested(n) => EinsumOptimize::Nested(n.clone()),
        EinsumOptimize::Path(p) => EinsumOptimize::Path(p.clone()),
        // ContractionTree is not Clone, so a borrowed `&opts` cannot hand it to the
        // by-value `EinsumOptimize` parameter of the current traced API.
        EinsumOptimize::Tree(_) => {
            return Err(Error::unsupported(
                "einsum",
                ErrorPhase::GraphBuild,
                "EinsumOptimize::Tree cannot be passed through borrowed options: ContractionTree is not Clone",
            ))
        }
    })
}

fn to_runtime_error(error: EinsumError) -> Error {
    match error {
        EinsumError::Runtime(inner) => inner,
        other => {
            let kind = other.kind();
            Error::extension(
                "einsum",
                ErrorPhase::GraphBuild,
                EINSUM_EXTENSION_FAMILY_ID,
                kind,
                other,
            )
        }
    }
}

/// Owner-receiver einsum for traced tensors.
pub trait TraceEinsum {
    /// Add one einsum node over `inputs` and return its traced output.
    ///
    /// # Errors
    ///
    /// Returns a validation error when an input belongs to another trace
    /// context, and the einsum parse/planning/build error otherwise.
    fn trace_einsum_with<'s, T: Borrow<TracedTensor>>(
        &self,
        inputs: &[T],
        spec: impl Into<EinsumSpec<'s>>,
        opts: &TraceEinsumOptions,
    ) -> Result<TracedTensor>;
}

impl TraceEinsum for TraceContext {
    fn trace_einsum_with<'s, T: Borrow<TracedTensor>>(
        &self,
        inputs: &[T],
        spec: impl Into<EinsumSpec<'s>>,
        opts: &TraceEinsumOptions,
    ) -> Result<TracedTensor> {
        let mut values: Vec<TraceValue> = Vec::with_capacity(inputs.len());
        for (position, input) in inputs.iter().enumerate() {
            let input = input.borrow();
            if !input.context().same_context(self) {
                return Err(Error::invalid_argument(
                    "einsum",
                    ErrorPhase::GraphBuild,
                    "inputs",
                    format!("input {position} belongs to a different trace context"),
                ));
            }
            values.push(input.trace_value());
        }
        // TraceContext is an Rc handle; the Ext trait demands `&mut self` only
        // because of its signature, so clone the handle (shares the same trace).
        let mut ctx = self.clone();
        let optimize = duplicate(&opts.optimize)?;
        let spec = spec.into();
        let value = match &spec {
            EinsumSpec::Text(text) => ctx.einsum_with(&values, text, optimize),
            EinsumSpec::Subscripts(s) => ctx.einsum_subscripts_with(&values, s, optimize),
            EinsumSpec::Notation(n) => ctx.einsum_notation_with(&values, n, optimize),
            EinsumSpec::Indices(s) => ctx.einsum_subscripts_with(&values, s, optimize),
        }
        .map_err(to_runtime_error)?;
        // No public TraceValue -> TracedTensor constructor exists outside this
        // doc(hidden) hook.
        self.derive_tensor(value)
    }
}
