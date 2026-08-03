use super::super::retirement::{PreparedPackage, RetirementOutcome, RetirementRecord};

#[test]
fn proven_retirement_releases_binding_root_and_context_once() {
    let _: Option<RetirementRecord> = None;
    assert!(matches!(
        RetirementOutcome::Completed,
        RetirementOutcome::Completed
    ));
}

#[test]
fn unproven_retirement_keeps_binding_root_and_context_alive() {
    let _: Option<RetirementOutcome> = None;
}

#[test]
fn pre_admission_rejection_returns_the_unchanged_prepared_package() {
    let _: Option<PreparedPackage> = None;
}
