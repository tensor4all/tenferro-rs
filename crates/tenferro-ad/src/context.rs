//! Explicit ownership for automatic-differentiation rule sets.

use std::sync::Arc;

use tenferro_ops::{ExtensionRegistryError, ExtensionRuleSet};
use tenferro_runtime::{CacheStats, Result, TracedTensor};

use crate::transform_cache::{AdTransformCache, AdTransformCacheLimits};

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
/// assert!(ad.extension_rules().lookup_linearize("example.missing.v1").is_none());
/// ```
#[derive(Clone, Debug)]
pub struct AdContext {
    extension_rules: ExtensionRuleSet,
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

    /// Return the extension rules owned by this context.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ad::AdContext;
    ///
    /// let ad = AdContext::builder().build().unwrap();
    /// assert!(!ad.extension_rules().is_linearize_registered("example.missing.v1"));
    /// ```
    pub fn extension_rules(&self) -> &ExtensionRuleSet {
        &self.extension_rules
    }

    pub(crate) fn extension_rule_set(&self) -> ExtensionRuleSet {
        self.extension_rules.clone()
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
    pub fn grad(&self, output: &TracedTensor, wrt: &TracedTensor) -> Result<TracedTensor> {
        crate::traced::grad_with_rules_and_cache(
            output,
            wrt,
            &self.extension_rules,
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
    pub fn grad_optional(
        &self,
        output: &TracedTensor,
        wrt: &TracedTensor,
    ) -> Result<Option<TracedTensor>> {
        crate::traced::grad_optional_with_rules_and_cache(
            output,
            wrt,
            &self.extension_rules,
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
            &self.extension_rules,
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
            &self.extension_rules,
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
            &self.extension_rules,
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
            &self.extension_rules,
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
/// assert!(ad.extension_rules().lookup_linearize("example.missing.v1").is_none());
/// ```
#[derive(Clone, Debug, Default)]
pub struct AdContextBuilder {
    extension_rule_sets: Vec<ExtensionRuleSet>,
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

    /// Include an owned extension rule set.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ad::{AdContext, extension::ExtensionRuleSet};
    ///
    /// let _ad = AdContext::builder()
    ///     .with_extension_rules(ExtensionRuleSet::new())
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn with_extension_rules(mut self, rules: ExtensionRuleSet) -> Self {
        self.extension_rule_sets.push(rules);
        self
    }

    /// Build the context.
    ///
    /// Duplicate extension family registrations are rejected.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ad::AdContext;
    ///
    /// let ad = AdContext::builder().build().unwrap();
    /// assert!(ad.extension_rules().lookup_linearize("example.missing.v1").is_none());
    /// ```
    pub fn build(self) -> std::result::Result<AdContext, ExtensionRegistryError> {
        let mut extension_rules = ExtensionRuleSet::new();
        for rules in self.extension_rule_sets {
            extension_rules.merge(rules)?;
        }
        Ok(AdContext {
            extension_rules,
            ad_transform_cache: Arc::new(AdTransformCache::new()),
        })
    }
}
