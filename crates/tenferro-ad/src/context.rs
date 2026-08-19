//! Explicit ownership for automatic-differentiation rule sets.

use std::sync::Arc;

use tenferro_runtime::program::FrozenProgram;
use tenferro_runtime::{CacheStats, Result, TracedTensor};

// SemanticCompatDispatcher removed in Unification 7.
// Extension AD is handled exclusively by SemanticExtensionRuleSet.
use crate::semantic_extension::{SemanticExtensionRegistryError, SemanticExtensionRuleSet};
use crate::semantic_transform::{
    semantic_jvp, semantic_vjp, SemanticAdProgram, SemanticAdTransformError,
};
use crate::transform_cache::{
    AdTransformCache, AdTransformCacheLimits, SemanticAdTransformCacheKey,
};

/// Stats for caches owned by an [`AdContext`].
///
/// `retained_bytes` fields are logical payload estimates, not process RSS.
///
/// # Examples
///
/// ```rust
/// use tenferro_ad::AdContext;
///
/// let ad = AdContext::builder().build().unwrap();
/// assert_eq!(ad.cache_stats().unwrap().ad_transforms.entries, 0);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AdContextCacheStats {
    /// AD transform graph memoization cache.
    pub ad_transforms: CacheStats,
}

/// Explicit automatic-differentiation context.
///
/// `AdContext` owns the extension AD rules used by traced AD transforms.
/// It also owns the AD transform cache shared by context-driven traced AD and
/// eager runtimes created from this context.
///
/// # Examples
///
/// ```rust
/// use tenferro_ad::AdContext;
///
/// let ad = AdContext::builder().build().unwrap();
/// assert!(ad
///     .semantic_extension_rules()
///     .lookup_linearize("example.missing.v1")
///     .is_none());
/// ```
#[derive(Clone, Debug)]
pub struct AdContext {
    semantic_extension_rules: SemanticExtensionRuleSet,
    ad_transform_cache: Arc<AdTransformCache>,
}

impl AdContext {
    /// Start building an explicit AD context.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ad::AdContext;
    ///
    /// let _builder = AdContext::builder();
    /// ```
    pub fn builder() -> AdContextBuilder {
        AdContextBuilder::default()
    }

    pub(crate) fn with_rules_and_transform_cache(
        semantic_extension_rules: SemanticExtensionRuleSet,
        ad_transform_cache: Arc<AdTransformCache>,
    ) -> Self {
        Self {
            semantic_extension_rules,
            ad_transform_cache,
        }
    }

    /// Return semantic-program extension AD rules owned by this context.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ad::AdContext;
    ///
    /// let ad = AdContext::builder().build().unwrap();
    /// assert!(ad
    ///     .semantic_extension_rules()
    ///     .lookup_linearize("example.missing.v1")
    ///     .is_none());
    /// ```
    pub fn semantic_extension_rules(&self) -> &SemanticExtensionRuleSet {
        &self.semantic_extension_rules
    }

    /// Transform a frozen semantic program into its forward-mode derivative.
    ///
    /// `active_inputs` follows source-program input order. Active tangent
    /// seeds are appended after all primal inputs.
    ///
    /// # Errors
    ///
    /// Returns [`SemanticAdTransformError::ActivityArity`] when
    /// `active_inputs` has the wrong length,
    /// [`SemanticAdTransformError::Extension`] when an extension rule rejects
    /// the transform, or the corresponding `Query`, `Build`, `Finish`, or
    /// `Cache` variant when program import, construction, finalization, or
    /// cache access fails.
    pub fn jvp_program(
        &self,
        input: &FrozenProgram,
        active_inputs: &[bool],
    ) -> std::result::Result<SemanticAdProgram, SemanticAdTransformError> {
        let key = SemanticAdTransformCacheKey::jvp(input, active_inputs);
        if let Some(cached) = self
            .ad_transform_cache
            .get_semantic(&key, input)
            .map_err(SemanticAdTransformError::Cache)?
        {
            return cached
                .as_ref()
                .with_input_prefix_bindings_from(input)
                .map_err(SemanticAdTransformError::from);
        }
        let transformed = semantic_jvp(input, active_inputs, &self.semantic_extension_rules)?;
        self.ad_transform_cache
            .put_semantic(key, input, Arc::new(transformed.clone()))
            .map_err(SemanticAdTransformError::Cache)?;
        Ok(transformed)
    }

    /// Transform a frozen semantic program into its reverse-mode derivative.
    ///
    /// `active_inputs` selects requested primal-input cotangents and
    /// `active_outputs` selects primal outputs that receive appended seeds.
    ///
    /// # Errors
    ///
    /// Returns [`SemanticAdTransformError::ActivityArity`] when either activity
    /// mask has the wrong length,
    /// [`SemanticAdTransformError::Extension`] when an extension rule rejects
    /// the transform, or the corresponding `Query`, `Build`, `Finish`, or
    /// `Cache` variant when program import, construction, finalization, or
    /// cache access fails.
    pub fn vjp_program(
        &self,
        input: &FrozenProgram,
        active_inputs: &[bool],
        active_outputs: &[bool],
    ) -> std::result::Result<SemanticAdProgram, SemanticAdTransformError> {
        let key = SemanticAdTransformCacheKey::vjp(input, active_inputs, active_outputs);
        if let Some(cached) = self
            .ad_transform_cache
            .get_semantic(&key, input)
            .map_err(SemanticAdTransformError::Cache)?
        {
            return cached
                .as_ref()
                .with_input_prefix_bindings_from(input)
                .map_err(SemanticAdTransformError::from);
        }
        let transformed = semantic_vjp(
            input,
            active_inputs,
            active_outputs,
            &self.semantic_extension_rules,
        )?;
        self.ad_transform_cache
            .put_semantic(key, input, Arc::new(transformed.clone()))
            .map_err(SemanticAdTransformError::Cache)?;
        Ok(transformed)
    }

    pub(crate) fn ad_transform_cache(&self) -> Arc<AdTransformCache> {
        Arc::clone(&self.ad_transform_cache)
    }

    /// Return AD transform cache retention limits.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ad::AdContext;
    ///
    /// let ad = AdContext::builder().build().unwrap();
    /// assert!(ad.ad_transform_cache_limits().unwrap().max_entries().get() > 0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_runtime::Error::RuntimeState`] if the cache lock is
    /// poisoned or its state cannot be inspected.
    pub fn ad_transform_cache_limits(&self) -> Result<AdTransformCacheLimits> {
        self.ad_transform_cache.limits()
    }

    /// Replace AD transform cache retention limits.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::num::NonZeroUsize;
    /// use tenferro_ad::{AdContext, AdTransformCacheLimits};
    ///
    /// let ad = AdContext::builder().build().unwrap();
    /// let limits = AdTransformCacheLimits::new(NonZeroUsize::new(1).unwrap());
    /// ad.set_ad_transform_cache_limits(limits).unwrap();
    /// assert_eq!(ad.ad_transform_cache_limits().unwrap(), limits);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_runtime::Error::RuntimeState`] if the cache lock is
    /// poisoned while updating the limits.
    pub fn set_ad_transform_cache_limits(&self, limits: AdTransformCacheLimits) -> Result<()> {
        self.ad_transform_cache.set_limits(limits)
    }

    /// Clear AD transform cache entries owned by this context.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ad::AdContext;
    ///
    /// let ad = AdContext::builder().build().unwrap();
    /// ad.clear_ad_transform_caches().unwrap();
    /// assert_eq!(ad.ad_transform_cache_stats().unwrap().entries, 0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_runtime::Error::RuntimeState`] if the cache lock is
    /// poisoned while clearing entries.
    pub fn clear_ad_transform_caches(&self) -> Result<()> {
        self.ad_transform_cache.clear()
    }

    /// Return AD transform cache-entry and retained-byte stats.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ad::AdContext;
    ///
    /// let ad = AdContext::builder().build().unwrap();
    /// assert_eq!(ad.ad_transform_cache_stats().unwrap().entries, 0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_runtime::Error::RuntimeState`] if the cache lock is
    /// poisoned while collecting statistics.
    pub fn ad_transform_cache_stats(&self) -> Result<CacheStats> {
        self.ad_transform_cache.stats()
    }

    /// Clear every cache owned by this AD context.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ad::AdContext;
    ///
    /// let ad = AdContext::builder().build().unwrap();
    /// ad.clear_caches().unwrap();
    /// assert_eq!(ad.cache_stats().unwrap().ad_transforms.entries, 0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_runtime::Error::RuntimeState`] if either owned cache
    /// cannot be locked because its state is poisoned.
    pub fn clear_caches(&self) -> Result<()> {
        self.clear_ad_transform_caches()
    }

    /// Return aggregate cache-entry and retained-byte stats for this AD context.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ad::AdContext;
    ///
    /// let ad = AdContext::builder().build().unwrap();
    /// assert_eq!(ad.cache_stats().unwrap().ad_transforms.retained_bytes, 0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_runtime::Error::RuntimeState`] if an owned cache lock
    /// is poisoned while collecting statistics.
    pub fn cache_stats(&self) -> Result<AdContextCacheStats> {
        Ok(AdContextCacheStats {
            ad_transforms: self.ad_transform_cache_stats()?,
        })
    }

    /// Gradient of a scalar traced output with respect to a traced input.
    ///
    /// For complex scalar outputs, tenferro returns the Hermitian-adjoint
    /// cotangent. To compare seed-`1` scalar gradients with JAX's public
    /// `grad` values, use the complex conjugate of this result. See
    /// <https://tensor4all.org/tenferro-rs/guides/complex-ad.html>.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ad::AdContext;
    /// use tenferro_runtime::TracedTensor;
    ///
    /// let ad = AdContext::builder().build().unwrap();
    /// let x = TracedTensor::from_vec_col_major(vec![], vec![3.0_f64]).unwrap();
    /// let loss = (&x * &x).unwrap();
    /// let grad = ad.grad(&loss, &x).unwrap();
    /// assert_eq!(grad.rank, 0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_runtime::Error::NonScalarGrad`] when `output` is not
    /// scalar, [`tenferro_runtime::Error::UnsupportedAdRule`] when a graph op
    /// lacks a registered rule, or a typed [`tenferro_runtime::Error::Validation`]
    /// / backend error when graph metadata or execution is invalid.
    pub fn grad(&self, output: &TracedTensor, wrt: &TracedTensor) -> Result<TracedTensor> {
        crate::traced::grad_with_rules_and_cache(
            output,
            wrt,
            &self.semantic_extension_rules,
            Some(self.ad_transform_cache.as_ref()),
        )
    }

    /// Gradient that returns `None` when `wrt` is inactive.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ad::AdContext;
    /// use tenferro_runtime::TracedTensor;
    ///
    /// let ad = AdContext::builder().build().unwrap();
    /// let x = TracedTensor::from_vec_col_major(vec![], vec![3.0_f64]).unwrap();
    /// let loss = (&x * &x).unwrap();
    /// assert!(ad.grad_optional(&loss, &x).unwrap().is_some());
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_runtime::Error::NonScalarGrad`] for a non-scalar
    /// output, [`tenferro_runtime::Error::UnsupportedAdRule`] for an
    /// unregistered AD rule, or a typed [`tenferro_runtime::Error::Validation`]
    /// / backend error from graph construction and
    /// execution.
    pub fn grad_optional(
        &self,
        output: &TracedTensor,
        wrt: &TracedTensor,
    ) -> Result<Option<TracedTensor>> {
        crate::traced::grad_optional_with_rules_and_cache(
            output,
            wrt,
            &self.semantic_extension_rules,
            Some(self.ad_transform_cache.as_ref()),
        )
    }

    /// Forward-mode Jacobian-vector product.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ad::AdContext;
    /// use tenferro_runtime::TracedTensor;
    ///
    /// let ad = AdContext::builder().build().unwrap();
    /// let x = TracedTensor::from_vec_col_major(vec![], vec![3.0_f64]).unwrap();
    /// let dx = TracedTensor::from_vec_col_major(vec![], vec![1.0_f64]).unwrap();
    /// let y = (&x * &x).unwrap();
    /// let dy = ad.jvp(&y, &x, &dx).unwrap();
    /// assert_eq!(dy.rank, 0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_runtime::Error::UnsupportedAdRule`] when the graph
    /// has no JVP rule, [`tenferro_runtime::Error::Validation`] for
    /// inconsistent tangent metadata, or a typed backend/runtime-state error
    /// during evaluation.
    pub fn jvp(
        &self,
        output: &TracedTensor,
        wrt: &TracedTensor,
        tangent: &TracedTensor,
    ) -> Result<TracedTensor> {
        crate::traced::jvp_with_rules_and_cache(
            output,
            wrt,
            tangent,
            &self.semantic_extension_rules,
            Some(self.ad_transform_cache.as_ref()),
        )
    }

    /// Forward-mode Jacobian-vector product that returns `None` for inactive output.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ad::AdContext;
    /// use tenferro_runtime::TracedTensor;
    ///
    /// let ad = AdContext::builder().build().unwrap();
    /// let x = TracedTensor::from_vec_col_major(vec![], vec![3.0_f64]).unwrap();
    /// let dx = TracedTensor::from_vec_col_major(vec![], vec![1.0_f64]).unwrap();
    /// let y = (&x * &x).unwrap();
    /// assert!(ad.jvp_optional(&y, &x, &dx).unwrap().is_some());
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_runtime::Error::UnsupportedAdRule`] when the graph
    /// has no JVP rule, [`tenferro_runtime::Error::Validation`] for
    /// inconsistent tangent metadata, or a typed backend/runtime-state error
    /// during evaluation.
    pub fn jvp_optional(
        &self,
        output: &TracedTensor,
        wrt: &TracedTensor,
        tangent: &TracedTensor,
    ) -> Result<Option<TracedTensor>> {
        crate::traced::jvp_optional_with_rules_and_cache(
            output,
            wrt,
            tangent,
            &self.semantic_extension_rules,
            Some(self.ad_transform_cache.as_ref()),
        )
    }

    /// Forward-mode directional derivative for multiple distinct traced leaves.
    ///
    /// Reachable leaves are transformed together in one derivative graph.
    /// Unreachable leaves contribute nothing; an empty or fully unreachable
    /// request returns `None`. Duplicate `wrt` leaves are rejected before the
    /// transform because one semantic seed slot cannot accept two tangents.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ad::AdContext;
    /// use tenferro_runtime::TracedTensor;
    ///
    /// let ad = AdContext::builder().build().unwrap();
    /// let x = TracedTensor::from_vec_col_major(vec![], vec![2.0_f64]).unwrap();
    /// let y = TracedTensor::from_vec_col_major(vec![], vec![3.0_f64]).unwrap();
    /// let dx = TracedTensor::from_vec_col_major(vec![], vec![1.0_f64]).unwrap();
    /// let dy = TracedTensor::from_vec_col_major(vec![], vec![4.0_f64]).unwrap();
    /// let output = (&x * &y).unwrap();
    /// assert!(ad.jvp_many(&output, &[(&x, &dx), (&y, &dy)]).unwrap().is_some());
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_runtime::Error::Validation`] for duplicate leaves or
    /// incompatible tangent metadata, [`tenferro_runtime::Error::UnsupportedAdRule`]
    /// when a required rule is unavailable, or a typed runtime-state error when
    /// derivative graph construction fails.
    ///
    /// # Deferred errors
    ///
    /// Symbolic shape constraints may fail during later compilation or execution.
    pub fn jvp_many(
        &self,
        output: &TracedTensor,
        wrt_tangents: &[(&TracedTensor, &TracedTensor)],
    ) -> Result<Option<TracedTensor>> {
        crate::traced::jvp_many_with_rules_and_cache(
            output,
            wrt_tangents,
            &self.semantic_extension_rules,
            Some(self.ad_transform_cache.as_ref()),
        )
    }

    /// Reverse-mode vector-Jacobian product.
    ///
    /// Complex cotangents use tenferro's Hermitian real-inner-product
    /// convention. Non-real complex cotangent seeds therefore need an explicit
    /// seed-convention comparison when matching JAX. See
    /// <https://tensor4all.org/tenferro-rs/guides/complex-ad.html>.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ad::AdContext;
    /// use tenferro_runtime::TracedTensor;
    ///
    /// let ad = AdContext::builder().build().unwrap();
    /// let x = TracedTensor::from_vec_col_major(vec![], vec![3.0_f64]).unwrap();
    /// let dy = TracedTensor::from_vec_col_major(vec![], vec![1.0_f64]).unwrap();
    /// let y = (&x * &x).unwrap();
    /// let dx = ad.vjp(&y, &x, &dy).unwrap();
    /// assert_eq!(dx.rank, 0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_runtime::Error::Validation`] when the cotangent
    /// metadata is incompatible, [`tenferro_runtime::Error::UnsupportedAdRule`]
    /// when a VJP rule is unavailable, or a typed backend/runtime-state error
    /// during execution.
    pub fn vjp(
        &self,
        output: &TracedTensor,
        wrt: &TracedTensor,
        cotangent: &TracedTensor,
    ) -> Result<TracedTensor> {
        crate::traced::vjp_with_rules_and_cache(
            output,
            wrt,
            cotangent,
            &self.semantic_extension_rules,
            Some(self.ad_transform_cache.as_ref()),
        )
    }

    /// Reverse-mode vector-Jacobian product that returns `None` for inactive input.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ad::AdContext;
    /// use tenferro_runtime::TracedTensor;
    ///
    /// let ad = AdContext::builder().build().unwrap();
    /// let x = TracedTensor::from_vec_col_major(vec![], vec![3.0_f64]).unwrap();
    /// let dy = TracedTensor::from_vec_col_major(vec![], vec![1.0_f64]).unwrap();
    /// let y = (&x * &x).unwrap();
    /// assert!(ad.vjp_optional(&y, &x, &dy).unwrap().is_some());
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_runtime::Error::Validation`] when the cotangent
    /// metadata is incompatible, [`tenferro_runtime::Error::UnsupportedAdRule`]
    /// when a VJP rule is unavailable, or a typed backend/runtime-state error
    /// during execution.
    pub fn vjp_optional(
        &self,
        output: &TracedTensor,
        wrt: &TracedTensor,
        cotangent: &TracedTensor,
    ) -> Result<Option<TracedTensor>> {
        crate::traced::vjp_optional_with_rules_and_cache(
            output,
            wrt,
            cotangent,
            &self.semantic_extension_rules,
            Some(self.ad_transform_cache.as_ref()),
        )
    }

    /// Reverse-mode products for multiple traced leaves in one derivative graph.
    ///
    /// Results align with `wrts`; unreachable leaves produce `None`. Duplicate
    /// leaves are allowed and repeat the same traced derivative without
    /// accumulating the cotangent twice. An empty request validates that the
    /// cotangent has concrete data, then returns an empty vector.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ad::AdContext;
    /// use tenferro_runtime::TracedTensor;
    ///
    /// let ad = AdContext::builder().build().unwrap();
    /// let x = TracedTensor::from_vec_col_major(vec![], vec![2.0_f64]).unwrap();
    /// let y = TracedTensor::from_vec_col_major(vec![], vec![3.0_f64]).unwrap();
    /// let seed = TracedTensor::from_vec_col_major(vec![], vec![1.0_f64]).unwrap();
    /// let output = (&x * &y).unwrap();
    /// let products = ad.vjp_many(&output, &[&x, &y], &seed).unwrap();
    /// assert!(products.iter().all(Option::is_some));
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_runtime::Error::Validation`] for invalid cotangent
    /// metadata, [`tenferro_runtime::Error::UnsupportedAdRule`] when a required
    /// rule is unavailable, or a typed runtime-state error when derivative graph
    /// construction fails.
    ///
    /// # Deferred errors
    ///
    /// Symbolic shape constraints may fail during later compilation or execution.
    pub fn vjp_many(
        &self,
        output: &TracedTensor,
        wrts: &[&TracedTensor],
        cotangent: &TracedTensor,
    ) -> Result<Vec<Option<TracedTensor>>> {
        crate::traced::vjp_many_with_rules_and_cache(
            output,
            wrts,
            cotangent,
            &self.semantic_extension_rules,
            Some(self.ad_transform_cache.as_ref()),
        )
    }
}

/// Builder for [`AdContext`].
///
/// # Examples
///
/// ```rust
/// use tenferro_ad::AdContextBuilder;
///
/// let ad = AdContextBuilder::new().build().unwrap();
/// assert!(ad
///     .semantic_extension_rules()
///     .lookup_linearize("example.missing.v1")
///     .is_none());
/// ```
#[derive(Clone, Debug, Default)]
pub struct AdContextBuilder {
    semantic_extension_rules: SemanticExtensionRuleSet,
}

impl AdContextBuilder {
    /// Create an empty builder.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ad::AdContextBuilder;
    ///
    /// let _builder = AdContextBuilder::new();
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Include an owned semantic-program extension AD rule set.
    ///
    /// # Errors
    ///
    /// Returns [`SemanticExtensionRegistryError::MalformedFamilyId`] when a
    /// family identifier is invalid, or
    /// [`SemanticExtensionRegistryError::DuplicateRule`] when the same family
    /// and role were already supplied.
    pub fn with_semantic_extension_rules(
        mut self,
        rules: SemanticExtensionRuleSet,
    ) -> std::result::Result<Self, SemanticExtensionRegistryError> {
        self.semantic_extension_rules.merge(rules)?;
        Ok(self)
    }

    /// Build the context.
    ///
    /// Semantic extension rules have already been validated and merged by
    /// [`Self::with_semantic_extension_rules`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ad::AdContext;
    ///
    /// let ad = AdContext::builder().build().unwrap();
    /// assert!(ad
    ///     .semantic_extension_rules()
    ///     .lookup_linearize("example.missing.v1")
    ///     .is_none());
    /// ```
    ///
    /// # Errors
    ///
    /// The error type is [`std::convert::Infallible`], so this finalization step
    /// never returns `Err` after semantic rule registration. It retains a
    /// `Result` so callers can compose it with the fallible registration step.
    pub fn build(self) -> std::result::Result<AdContext, std::convert::Infallible> {
        Ok(AdContext {
            semantic_extension_rules: self.semantic_extension_rules,
            ad_transform_cache: Arc::new(AdTransformCache::new()),
        })
    }
}
