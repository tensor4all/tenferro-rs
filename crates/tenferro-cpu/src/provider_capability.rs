use std::num::NonZeroUsize;

use thiserror::Error;

use crate::{CpuPlacementGuarantee, CpuSet, ParallelMode};

/// Per-call control over the maximum number of threads used by a CPU provider.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::CpuThreadCountControl;
/// assert_ne!(
///     CpuThreadCountControl::PerCallUpperBound,
///     CpuThreadCountControl::GlobalOrUncontrolled,
/// );
/// ```
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum CpuThreadCountControl {
    /// The provider is sequential by construction.
    Sequential,
    /// Every call accepts an arbitrary positive upper bound.
    PerCallUpperBound,
    /// Every finite-budget call is clamped to one thread by the adapter.
    ///
    /// The adapter must never select its provider-controlled `auto` mode for a
    /// resource-domain call. Providers that cannot make that guarantee must
    /// report [`CpuThreadCountControl::GlobalOrUncontrolled`] instead.
    BinaryClampToOne,
    /// Control is global, startup-fixed, absent, or otherwise unsafe per call.
    #[default]
    GlobalOrUncontrolled,
}

/// Per-call control over where a CPU provider executes.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::CpuPlacementControl;
/// assert_ne!(
///     CpuPlacementControl::EngineWorkers,
///     CpuPlacementControl::ExternalWorkers,
/// );
/// ```
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum CpuPlacementControl {
    /// Parallel work stays on workers supplied by the selected executor.
    EngineWorkers,
    /// The provider executes entirely on the calling worker.
    CallingThread,
    /// Parallel work may use a provider-owned worker pool.
    ExternalWorkers,
    /// The provider makes no enforceable placement claim.
    #[default]
    None,
}

/// Immutable execution capabilities declared by one CPU provider.
///
/// The conservative default only permits provider-owned inner execution. A
/// provider must opt in explicitly to sequential or engine-owned outer modes.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::{
///     CpuPlacementControl, CpuProviderExecutionCapabilities, CpuThreadCountControl,
/// };
/// let capabilities = CpuProviderExecutionCapabilities {
///     thread_count: CpuThreadCountControl::Sequential,
///     placement: CpuPlacementControl::CallingThread,
///     worker_local_sequential: true,
///     accepts_sequential: true,
///     accepts_outer: true,
///     accepts_inner: true,
/// };
/// assert!(capabilities.worker_local_sequential);
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CpuProviderExecutionCapabilities {
    /// Per-call thread-count control implemented by the provider adapter.
    pub thread_count: CpuThreadCountControl,
    /// Placement control implemented independently from thread-count control.
    pub placement: CpuPlacementControl,
    /// Whether a call can be forced to stay sequential on its current worker.
    pub worker_local_sequential: bool,
    /// Whether the provider accepts a no-fan-out operation context.
    pub accepts_sequential: bool,
    /// Whether the provider accepts engine-owned fan-out with sequential children.
    pub accepts_outer: bool,
    /// Whether the provider accepts ownership of one inner parallel region.
    pub accepts_inner: bool,
}

impl Default for CpuProviderExecutionCapabilities {
    fn default() -> Self {
        Self {
            thread_count: CpuThreadCountControl::GlobalOrUncontrolled,
            placement: CpuPlacementControl::None,
            worker_local_sequential: false,
            accepts_sequential: false,
            accepts_outer: false,
            accepts_inner: true,
        }
    }
}

/// Typed incompatibility between a CPU provider and a selected CPU domain.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::{CpuProviderDomainError, CpuThreadCountControl};
/// let error = CpuProviderDomainError::ThreadCountNotEnforceable {
///     thread_budget: 4,
///     control: CpuThreadCountControl::GlobalOrUncontrolled,
/// };
/// assert!(error.to_string().contains("thread budget 4"));
/// ```
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum CpuProviderDomainError {
    /// The provider cannot enforce the domain's per-call thread upper bound.
    #[error(
        "provider thread-count control {control:?} cannot enforce thread budget {thread_budget}"
    )]
    ThreadCountNotEnforceable {
        /// Requested maximum number of participating threads.
        thread_budget: usize,
        /// Provider thread-count classification.
        control: CpuThreadCountControl,
    },
    /// The provider cannot enforce the domain's placement guarantee.
    #[error(
        "provider placement control {placement:?} cannot enforce {guarantee:?} placement for thread budget {thread_budget}"
    )]
    PlacementNotEnforceable {
        /// Requested maximum number of participating threads.
        thread_budget: usize,
        /// Provider placement classification.
        placement: CpuPlacementControl,
        /// Placement guarantee requested by the domain.
        guarantee: CpuPlacementGuarantee,
    },
    /// The provider cannot honor the engine-selected fan-out mode.
    #[error("provider cannot honor requested CPU parallel mode {mode:?}")]
    ParallelModeNotSupported {
        /// Mode selected by the execution engine.
        mode: ParallelMode,
    },
}

impl CpuProviderExecutionCapabilities {
    pub(crate) fn accepts_mode(self, mode: ParallelMode) -> bool {
        match mode {
            ParallelMode::Sequential => self.accepts_sequential && self.worker_local_sequential,
            ParallelMode::Outer => self.accepts_outer && self.worker_local_sequential,
            ParallelMode::Inner => self.accepts_inner,
        }
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum OpenBlasParallelism {
    Sequential,
    Pthread,
    OpenMp,
    Unknown,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct OpenBlasProbe {
    pub(crate) parallelism: OpenBlasParallelism,
    pub(crate) process_global_set_restore_wired: bool,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct AccelerateProbe {
    pub(crate) binary_thread_local_control_wired: bool,
}

/// Construction-time facts supplied by provider-specific adapters.
///
/// A discovered symbol is not enough: a corresponding `*_wired` field is true
/// only when the adapter applies and restores that control around every
/// provider call. OpenBLAS set-and-restore remains process-global even when it
/// is wired, so it never becomes per-call count control.
#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CpuProviderProbe {
    FaerOrNative,
    Mkl { thread_local_setter_wired: bool },
    OpenBlas(OpenBlasProbe),
    Accelerate(AccelerateProbe),
    ArmPlOpenMp,
    ArmPlSerial,
    NvplSerial,
    UnknownBlas,
    Injected(Option<CpuProviderExecutionCapabilities>),
}

#[cfg(test)]
pub(crate) fn classify_provider(probe: CpuProviderProbe) -> CpuProviderExecutionCapabilities {
    match probe {
        CpuProviderProbe::FaerOrNative => engine_worker_capabilities(),
        CpuProviderProbe::Mkl {
            thread_local_setter_wired: true,
        } => controlled_external_capabilities(CpuThreadCountControl::PerCallUpperBound),
        CpuProviderProbe::Mkl {
            thread_local_setter_wired: false,
        }
        | CpuProviderProbe::ArmPlOpenMp => uncontrolled_external_capabilities(),
        CpuProviderProbe::OpenBlas(probe) => classify_openblas(probe),
        CpuProviderProbe::Accelerate(probe) => classify_accelerate(probe),
        CpuProviderProbe::ArmPlSerial | CpuProviderProbe::NvplSerial => serial_capabilities(),
        CpuProviderProbe::UnknownBlas | CpuProviderProbe::Injected(None) => {
            CpuProviderExecutionCapabilities::default()
        }
        CpuProviderProbe::Injected(Some(capabilities)) => capabilities,
    }
}

#[cfg(test)]
fn classify_openblas(probe: OpenBlasProbe) -> CpuProviderExecutionCapabilities {
    match (probe.parallelism, probe.process_global_set_restore_wired) {
        (OpenBlasParallelism::Sequential, _) => serial_capabilities(),
        (OpenBlasParallelism::Pthread | OpenBlasParallelism::OpenMp, _) => {
            uncontrolled_external_capabilities()
        }
        (OpenBlasParallelism::Unknown, _) => CpuProviderExecutionCapabilities::default(),
    }
}

#[cfg(test)]
fn classify_accelerate(probe: AccelerateProbe) -> CpuProviderExecutionCapabilities {
    if probe.binary_thread_local_control_wired {
        controlled_external_capabilities(CpuThreadCountControl::BinaryClampToOne)
    } else {
        uncontrolled_external_capabilities()
    }
}

pub(crate) fn engine_worker_capabilities() -> CpuProviderExecutionCapabilities {
    CpuProviderExecutionCapabilities {
        thread_count: CpuThreadCountControl::PerCallUpperBound,
        placement: CpuPlacementControl::EngineWorkers,
        worker_local_sequential: true,
        accepts_sequential: true,
        accepts_outer: true,
        accepts_inner: true,
    }
}

#[cfg(test)]
fn controlled_external_capabilities(
    thread_count: CpuThreadCountControl,
) -> CpuProviderExecutionCapabilities {
    CpuProviderExecutionCapabilities {
        thread_count,
        placement: CpuPlacementControl::ExternalWorkers,
        worker_local_sequential: true,
        accepts_sequential: true,
        accepts_outer: true,
        accepts_inner: true,
    }
}

#[cfg(any(test, feature = "cpu-blas"))]
fn uncontrolled_external_capabilities() -> CpuProviderExecutionCapabilities {
    CpuProviderExecutionCapabilities {
        thread_count: CpuThreadCountControl::GlobalOrUncontrolled,
        placement: CpuPlacementControl::ExternalWorkers,
        worker_local_sequential: false,
        accepts_sequential: false,
        accepts_outer: false,
        accepts_inner: true,
    }
}

pub(crate) fn serial_capabilities() -> CpuProviderExecutionCapabilities {
    CpuProviderExecutionCapabilities {
        thread_count: CpuThreadCountControl::Sequential,
        placement: CpuPlacementControl::CallingThread,
        worker_local_sequential: true,
        accepts_sequential: true,
        accepts_outer: true,
        accepts_inner: true,
    }
}

/// Capabilities of the current built-in BLAS adapter.
///
/// The adapter does not yet install and restore any provider-specific local
/// thread-count setter, so all BLAS builds are classified conservatively.
#[cfg(any(test, feature = "cpu-blas"))]
pub(crate) fn builtin_blas_execution_capabilities() -> CpuProviderExecutionCapabilities {
    uncontrolled_external_capabilities()
}

pub(crate) fn validate_provider_for_domain(
    capabilities: CpuProviderExecutionCapabilities,
    thread_budget: NonZeroUsize,
    placement_guarantee: CpuPlacementGuarantee,
    domain_cpus: &CpuSet,
    process_allowed_cpus: &CpuSet,
) -> Result<(), CpuProviderDomainError> {
    if enforced_provider_thread_limit(capabilities.thread_count, thread_budget).is_none() {
        return Err(CpuProviderDomainError::ThreadCountNotEnforceable {
            thread_budget: thread_budget.get(),
            control: capabilities.thread_count,
        });
    }

    match capabilities.placement {
        CpuPlacementControl::EngineWorkers | CpuPlacementControl::CallingThread => Ok(()),
        CpuPlacementControl::ExternalWorkers => {
            if thread_budget.get() == 1 && capabilities.worker_local_sequential {
                return Ok(());
            }
            if placement_guarantee == CpuPlacementGuarantee::AdvisoryDeclared
                || domain_cpus == process_allowed_cpus
            {
                return Ok(());
            }
            Err(CpuProviderDomainError::PlacementNotEnforceable {
                thread_budget: thread_budget.get(),
                placement: capabilities.placement,
                guarantee: placement_guarantee,
            })
        }
        CpuPlacementControl::None => {
            if placement_guarantee == CpuPlacementGuarantee::AdvisoryDeclared {
                Ok(())
            } else {
                Err(CpuProviderDomainError::PlacementNotEnforceable {
                    thread_budget: thread_budget.get(),
                    placement: capabilities.placement,
                    guarantee: placement_guarantee,
                })
            }
        }
    }
}

fn enforced_provider_thread_limit(
    control: CpuThreadCountControl,
    thread_budget: NonZeroUsize,
) -> Option<NonZeroUsize> {
    match control {
        CpuThreadCountControl::Sequential | CpuThreadCountControl::BinaryClampToOne => {
            NonZeroUsize::new(1)
        }
        CpuThreadCountControl::PerCallUpperBound => Some(thread_budget),
        CpuThreadCountControl::GlobalOrUncontrolled => None,
    }
}

#[cfg(test)]
mod tests;
