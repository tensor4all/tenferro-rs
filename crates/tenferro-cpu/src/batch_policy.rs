//! Overridable batch execution strategy for batched CPU operations (#2142).
//!
//! A batched CPU operation (a grouped GEMM, a strided batched contraction or a
//! packed-LU batch) splits its independent items over the thread budget the
//! lower library receives on `tcontract_exec::Exec`. The vocabulary for that
//! split lives in `tcontract_exec` and is re-exported here so callers keep a
//! `tenferro_cpu` spelling:
//!
//! * [`CpuBatchPolicy`] is `tcontract_exec::BatchPolicy` (a [`CpuBatchStrategy`]
//!   plus the [`CpuBatchThresholds`] the `Auto` arm uses),
//! * [`CpuBatchStrategy`] is `tcontract_exec::BatchStrategy`,
//! * [`CpuBatchThresholds`] is `tcontract_exec::WorkModel`.
//!
//! Two variants were renamed when the vocabulary moved into `tcontract-exec`:
//! the old `OuterParallel` is [`CpuBatchStrategy::Outer`], and the old
//! `ProviderItems` is [`CpuBatchStrategy::Inner`]. `Sequential`, `Auto` and
//! `WholeBatchVendor` kept their names. The old vendor-dimension cutoff is
//! deliberately not restored: it was an inert knob once the decision moved into
//! the lower library.
//!
//! # Precedence
//!
//! A batched operation resolves its policy as the innermost scoped override,
//! then the [`crate::CpuBackend`] default:
//!
//! 1. the policy of an enclosing [`with_batch_policy`] scope, if any;
//! 2. otherwise [`crate::CpuBackend::batch_policy`], whose default is
//!    [`CpuBatchStrategy::Auto`].
//!
//! Wrapping one call in [`with_batch_policy`] therefore expresses a
//! per-operation choice. A non-CPU session is rejected with a typed
//! [`tenferro_tensor::ErrorKind::Unsupported`] error; there is no process-global
//! or thread-local policy outside the scope callback, and each scope restores
//! the previous policy on return, error and unwind.
//!
//! # Route-specific `Auto`
//!
//! `Auto` is resolved inside the lower library and keeps that library's legacy
//! rule until the upstream shared model lands (tcontract-rs#123 step 3):
//! `tlinalg`'s 64-extent cutoff that counts RHS columns in the linalg lane,
//! and `cpueinsum`'s width rule in the contraction lane. `CpuBatchThresholds`
//! feeds those rules where the lower library reads the policy's model; a forced
//! strategy ignores the model and never falls back.
//!
//! # The BLAS linalg lane
//!
//! `tlinalg-blas` takes no `Exec` and has no tenferro lanes, so the BLAS linalg
//! lane refuses a forced [`CpuBatchStrategy::Outer`],
//! [`CpuBatchStrategy::Inner`], `Split` or
//! [`CpuBatchStrategy::WholeBatchVendor`] before any write with a typed
//! unsupported error. `Auto` and `Sequential` are accepted; the lane is already
//! serial and the vendor call owns its own threading.

/// How the items of one batched operation are split over a thread budget.
///
/// This is `tcontract_exec::BatchStrategy`. The old `OuterParallel` variant is
/// [`CpuBatchStrategy::Outer`] and the old `ProviderItems` variant is
/// [`CpuBatchStrategy::Inner`]; `Sequential`, `Auto` and `WholeBatchVendor` kept
/// their names.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::CpuBatchStrategy;
///
/// assert_eq!(CpuBatchStrategy::default(), CpuBatchStrategy::Auto);
/// ```
pub use cpueinsum::BatchStrategy as CpuBatchStrategy;

/// A batch strategy together with the [`CpuBatchThresholds`] `Auto` applies.
///
/// This is `tcontract_exec::BatchPolicy`.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::{CpuBatchPolicy, CpuBatchStrategy};
///
/// let policy = CpuBatchPolicy::new(CpuBatchStrategy::Outer);
/// assert_eq!(policy.strategy, CpuBatchStrategy::Outer);
/// ```
pub use cpueinsum::BatchPolicy as CpuBatchPolicy;

/// The cost and fan-out model [`CpuBatchStrategy::Auto`] applies.
///
/// This is `tcontract_exec::WorkModel`; the old vendor-dimension cutoff is not
/// part of it.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::CpuBatchThresholds;
///
/// assert_eq!(CpuBatchThresholds::default().min_items, 2);
/// ```
pub use cpueinsum::WorkModel as CpuBatchThresholds;

/// Run `f` on `session` with `policy` as the effective CPU batch policy.
///
/// This is the scoped override of the precedence described in the [module
/// docs](self): wrapping a single call expresses a per-operation choice. The
/// previous policy is restored when `f` returns, returns an error or unwinds,
/// and scopes nest with the innermost winning. `f` runs on the caller's own
/// session, so a forwarding session keeps its own overrides inside the scope.
/// Nothing here mutates process-global state.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::{
///     with_batch_policy, with_cpu_exec_session, CpuBackend, CpuBatchPolicy, CpuBatchStrategy,
/// };
/// use tenferro_tensor::BackendSessionHost;
///
/// let mut backend = CpuBackend::with_threads(1)?;
/// let observed = backend.with_backend_session(|session| {
///     with_batch_policy(
///         session,
///         CpuBatchPolicy::new(CpuBatchStrategy::Outer),
///         |session| {
///             with_cpu_exec_session(session, |cpu| {
///                 cpu.with_linalg_pool(|context, _| Ok(context.batch_policy().strategy))
///             })
///             .expect("CpuBackend exposes its CpuExecSession")
///         },
///     )
/// })???;
/// assert_eq!(observed, CpuBatchStrategy::Outer);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Errors
///
/// Returns [`tenferro_tensor::Error::Unsupported`] without running `f` when
/// `session` is neither a CPU execution session nor a session that forwards one
/// through [`tenferro_tensor::Engine::native_engine`] (a CUDA or other
/// custom session). `f`'s own result is returned unchanged inside `Ok`.
///
/// # Panics
///
/// A panic in `f` propagates after the previous policy is restored.
pub fn with_batch_policy<R>(
    session: &mut dyn tenferro_tensor::Engine,
    policy: CpuBatchPolicy,
    f: impl FnOnce(&mut dyn tenferro_tensor::Engine) -> R,
) -> tenferro_tensor::Result<R> {
    let Some(previous) =
        crate::with_cpu_exec_session(session, |cpu| cpu.replace_batch_policy(policy))
    else {
        return Err(tenferro_tensor::Error::unsupported(
            "tenferro_cpu::with_batch_policy",
            "the session is not a CPU execution session and does not forward one; CPU batch \
             policies apply only to CpuBackend sessions",
        ));
    };
    // `f` runs on the caller's own session, so a wrapping session keeps its
    // overrides inside the scope. The policy is restored before an unwind
    // continues; nothing observes the session between the panic and the
    // restore, so asserting unwind safety is sound.
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(&mut *session)));
    // INVARIANT: the same session yielded a CPU execution session above and
    // native tokens are stable for a session's lifetime, so this visit runs.
    let _ = crate::with_cpu_exec_session(session, |cpu| cpu.replace_batch_policy(previous));
    match outcome {
        Ok(value) => Ok(value),
        Err(payload) => std::panic::resume_unwind(payload),
    }
}
