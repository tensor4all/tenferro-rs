//! The final application role of the external-scalar consumer.
//!
//! The application composes a canonical support with an optional scalar contribution and
//! binds the capabilities the algorithm states. It is the only role that names a scalar,
//! a provider, or a concrete dtype.
//!
//! The bindings live with the tests that exercise them, because a binding is only
//! meaningful together with the runtime it registers into.
//!
//! The application also declares its own scalar set. Together with the contribution's set
//! this gives two sets that both contain the external scalar, which is what #1785 asks for
//! when it requires reuse of a contribution-owned kernel by two containing sets. A set is
//! membership and dispatch only: the numerical entry point is parameterized by the
//! contribution's own scalar and operation type, so a second containing set adds no
//! numerical specialization.

use tenferro_df64_proof::Df64;

tenferro_tensor_core::define_scalar_set! {
    /// Tag for the set the application declares.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_scalar_consumer_application::ApplicationTag;
    ///
    /// assert_ne!(ApplicationTag::F64, ApplicationTag::Df64);
    /// ```
    pub enum ApplicationTag {
        /// Standard single precision.
        F32 => f32 : Float 0 32,
        /// Standard double precision.
        F64 => f64 : Float 1 64,
        /// The contribution's extended-precision scalar, ranked above double precision.
        Df64 => Df64 : Float 2 64,
    }
    /// Value enum for the set the application declares.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_df64_proof::Df64;
    /// use tenferro_scalar_consumer_application::{ApplicationSet, ApplicationTag};
    /// use tenferro_tensor_core::{HostTensor, ScalarSet};
    ///
    /// let value = ApplicationSet::Df64(
    ///     HostTensor::from_vec_col_major(vec![1], vec![Df64::from_f64(2.0)])?,
    /// );
    /// assert_eq!(value.tag(), ApplicationTag::Df64);
    /// # Ok::<(), tenferro_tensor_core::ValidationError>(())
    /// ```
    pub enum ApplicationSet;
}
