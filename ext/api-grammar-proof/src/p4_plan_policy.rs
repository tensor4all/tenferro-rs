//! Prototype 4: `MaterializationPolicy` on a borrowed SVD and on prepared
//! einsum plans.
//!
//! Thin grammar adapters over two internal, `#[doc(hidden)]` hooks:
//! `tenferro_linalg::svd_read_with_policy` and
//! `ConcreteEinsumPlan::{execute_read_with_policy, execute_read_into_with_policy}`.
//! The policy type itself lives in `tenferro_tensor::MaterializationPolicy`
//! (lower crates cannot depend on this proof crate); [`crate::policy`] is the
//! prototype copy and is mapped onto it by [`lower`]. See `findings/p4.md`.

use tenferro_einsum::ConcreteEinsumPlan;
use tenferro_linalg::svd_read_with_policy;
use tenferro_tensor::{Engine, Tensor, TensorRead, TensorWrite};

use crate::policy::MaterializationPolicy;

/// Map the grammar-level policy onto the one lower crates understand.
///
/// The grammar re-exports `tenferro_tensor::MaterializationPolicy`, so this is
/// the identity; it stays as the single place an adapter would translate if
/// the two ever diverged.
pub fn lower(policy: MaterializationPolicy) -> tenferro_tensor::MaterializationPolicy {
    policy
}

/// `read.svd_with(engine, policy)` on a borrowed erased tensor.
pub trait ReadSvdPolicyExt<'a> {
    /// `(U, S, Vh)`; `Forbid` refuses with
    /// `Error::InputMaterializationForbidden { op: "svd", input: 0 }` before
    /// any buffer or output work when the provider would gather the input.
    fn svd_with(
        self,
        engine: &mut dyn Engine,
        policy: MaterializationPolicy,
    ) -> tenferro_tensor::Result<(Tensor, Tensor, Tensor)>;
}

impl<'a> ReadSvdPolicyExt<'a> for TensorRead<'a> {
    fn svd_with(
        self,
        engine: &mut dyn Engine,
        policy: MaterializationPolicy,
    ) -> tenferro_tensor::Result<(Tensor, Tensor, Tensor)> {
        svd_read_with_policy(self, engine, lower(policy))
    }
}

/// `plan.execute_with(inputs, engine, policy)` and
/// `plan.execute_into_with(inputs, engine, policy, dest)`.
///
/// The destination variant is named `execute_into_with` because
/// `ConcreteEinsumPlan` already has an inherent `execute_into` with a different
/// arity, and an inherent method shadows a trait method of the same name.
pub trait PlanPolicyExt {
    /// Execute on borrowed inputs, returning a fresh output.
    fn execute_with(
        &self,
        inputs: &[TensorRead<'_>],
        engine: &mut dyn Engine,
        policy: MaterializationPolicy,
    ) -> tenferro_einsum::Result<Tensor>;

    /// Execute into a caller destination; the destination is untouched on refusal.
    fn execute_into_with(
        &self,
        inputs: &[TensorRead<'_>],
        engine: &mut dyn Engine,
        policy: MaterializationPolicy,
        dest: TensorWrite<'_>,
    ) -> tenferro_einsum::Result<()>;
}

impl PlanPolicyExt for ConcreteEinsumPlan {
    fn execute_with(
        &self,
        inputs: &[TensorRead<'_>],
        engine: &mut dyn Engine,
        policy: MaterializationPolicy,
    ) -> tenferro_einsum::Result<Tensor> {
        self.execute_read_with_policy(inputs, engine, lower(policy))
    }

    fn execute_into_with(
        &self,
        inputs: &[TensorRead<'_>],
        engine: &mut dyn Engine,
        policy: MaterializationPolicy,
        dest: TensorWrite<'_>,
    ) -> tenferro_einsum::Result<()> {
        self.execute_read_into_with_policy(inputs, engine, lower(policy), dest)
    }
}
