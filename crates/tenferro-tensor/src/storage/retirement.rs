use std::fmt;

use super::root::RootResourcePin;

/// A provider event completion that has been reduced to one typed outcome.
///
/// Providers report this once at the retirement boundary.  The record itself
/// remains the sole owner until the outcome is proven.
pub(crate) enum EventCompletion {
    Proven,
    Failed(RetirementError),
    Unproven(RetirementError),
}
/// Provider event observation used by the private retirement boundary.
pub(crate) trait ProviderEvent: Send {
    fn completion(&self) -> EventCompletion;
}

/// A binding retained until the provider event has been proven complete.
pub(crate) trait ProviderRetirementBinding: Send {}

/// Provider context retained with the detached bindings.
pub(crate) trait ProviderContext: Send {}

#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub(crate) enum RetirementError {
    #[error("provider event completion failed: {message}")]
    Provider { message: String },
    #[error("provider event completion is unproven: {message}")]
    Unproven { message: String },
}

#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub(crate) enum AdmissionError {
    #[error("provider admission rejected before enqueue: {message}")]
    Rejected { message: String },
}

/// The result of consuming a detached retirement record.
pub(crate) enum RetirementOutcome {
    Completed,
    Failed(RetirementError),
    CompletionUnproven(RetirementError),
}

/// Resources that are detached together at admission.
struct RetirementResources {
    bindings: Box<[Box<dyn ProviderRetirementBinding>]>,
    roots: Box<[RootResourcePin]>,
    provider: Box<dyn ProviderContext>,
}

/// Prepared resources before provider admission.
pub(crate) struct PreparedPackage {
    resources: RetirementResources,
}

/// The one pre-admission decision point.
pub(crate) enum AdmissionDecision {
    Enqueued(Box<dyn ProviderEvent>),
    Rejected(AdmissionError),
}

impl PreparedPackage {
    pub(crate) fn new(
        bindings: Box<[Box<dyn ProviderRetirementBinding>]>,
        roots: Box<[RootResourcePin]>,
        provider: Box<dyn ProviderContext>,
    ) -> Self {
        Self {
            resources: RetirementResources {
                bindings,
                roots,
                provider,
            },
        }
    }

    pub(crate) fn admit(
        self,
        decision: AdmissionDecision,
    ) -> Result<RetirementRecord, (Self, AdmissionError)> {
        match decision {
            AdmissionDecision::Enqueued(event) => Ok(RetirementRecord {
                event,
                resources: self.resources,
            }),
            AdmissionDecision::Rejected(error) => Err((self, error)),
        }
    }
}

/// Detached resources whose provider event controls their release.
pub(crate) struct RetirementRecord {
    event: Box<dyn ProviderEvent>,
    resources: RetirementResources,
}

impl fmt::Debug for RetirementOutcome {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Completed => formatter.write_str("Completed"),
            Self::Failed(error) => formatter.debug_tuple("Failed").field(error).finish(),
            Self::CompletionUnproven(error) => formatter
                .debug_tuple("CompletionUnproven")
                .field(error)
                .finish(),
        }
    }
}

impl RetirementRecord {
    pub(crate) fn finish(self) -> RetirementOutcome {
        let completion = self.event.completion();
        match completion {
            EventCompletion::Proven => {
                drop(self);
                RetirementOutcome::Completed
            }
            EventCompletion::Failed(error) => {
                drop(self);
                RetirementOutcome::Failed(error)
            }
            EventCompletion::Unproven(error) => {
                // INTENTIONAL: retain the complete record when completion
                // cannot be proven; no owner or retry handle escapes.
                let _retained: &'static mut RetirementRecord = Box::leak(Box::new(self));
                RetirementOutcome::CompletionUnproven(error)
            }
        }
    }
}
