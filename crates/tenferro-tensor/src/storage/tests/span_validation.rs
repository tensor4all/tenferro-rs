use std::num::NonZeroUsize;

use crate::{AllocationDomainId, AllocationId};

use super::super::{
    AllocationKey, ByteRange, RequestedIdentity, RootBoundSpan, RootResourceExtent,
    RootResourceIdentity, SpanValidationError, StorageOperation, StorageOperationContext,
    StorageOperationError,
};

fn key(domain: AllocationDomainId, local: u64) -> AllocationKey {
    AllocationKey::new(domain, AllocationId::from_backend_id(local))
}

#[test]
fn checked_ranges_reject_overflow_before_alignment() {
    let allocation_key = key(AllocationDomainId::fresh(), 1);
    let same_local_other_domain = key(AllocationDomainId::fresh(), 1);
    assert_ne!(allocation_key, same_local_other_domain);

    assert_eq!(
        RootResourceExtent::try_new(allocation_key, usize::MAX, 1, 0),
        Err(SpanValidationError::RangeOverflow {
            byte_offset: usize::MAX,
            byte_len: 1,
        })
    );

    assert_eq!(
        RootResourceExtent::try_new(allocation_key, 0, 1, 0),
        Err(SpanValidationError::InvalidAlignment { alignment: 0 })
    );
}

#[test]
fn byte_ranges_expose_checked_bounds_and_overlap_only_when_positive() {
    let left = ByteRange::new(8, 16);
    let touching = ByteRange::new(24, 8);
    let overlapping = ByteRange::new(16, 16);
    let empty = ByteRange::new(24, 0);

    assert_eq!(left.byte_offset(), 8);
    assert_eq!(left.byte_len(), 16);
    assert_eq!(left.checked_end(), Ok(24));
    assert!(!left.is_empty());
    assert_eq!(left.overlaps(touching), Ok(false));
    assert_eq!(left.overlaps(overlapping), Ok(true));
    assert_eq!(left.overlaps(empty), Ok(false));
    assert_eq!(
        ByteRange::new(usize::MAX, 1).checked_end(),
        Err(SpanValidationError::RangeOverflow {
            byte_offset: usize::MAX,
            byte_len: 1,
        })
    );
}

#[test]
fn relative_range_overflow_precedes_malformed_root_alignment() {
    let allocation_key = key(AllocationDomainId::fresh(), 2);
    let malformed = RootResourceExtent::test_corrupt(
        allocation_key,
        0,
        16,
        NonZeroUsize::new(3).expect("nonzero test alignment"),
    );

    assert_eq!(
        malformed.validate_relative_range(ByteRange::new(usize::MAX, 1)),
        Err(SpanValidationError::RangeOverflow {
            byte_offset: usize::MAX,
            byte_len: 1,
        })
    );
}

#[test]
fn relative_offset_and_child_end_overflow_precede_containment_and_alignment() {
    let allocation_key = key(AllocationDomainId::fresh(), 7);
    let near_limit = RootResourceExtent::try_new(allocation_key, usize::MAX, 0, 1)
        .expect("zero-length root at usize::MAX is representable");
    assert_eq!(
        near_limit.validate_relative_range(ByteRange::new(1, 0)),
        Err(SpanValidationError::OffsetOverflow {
            base_byte_offset: usize::MAX,
            relative_byte_offset: 1,
        })
    );

    let malformed = RootResourceExtent::test_corrupt(
        allocation_key,
        usize::MAX - 8,
        8,
        NonZeroUsize::new(3).expect("nonzero test alignment"),
    );
    assert_eq!(
        malformed.validate_relative_range(ByteRange::new(8, 1)),
        Err(SpanValidationError::RangeOverflow {
            byte_offset: usize::MAX,
            byte_len: 1,
        })
    );
}

#[test]
fn root_bound_spans_retain_exact_root_provenance() {
    let allocation_key = key(AllocationDomainId::fresh(), 3);
    let extent =
        RootResourceExtent::try_new(allocation_key, 0, 64, 8).expect("the root extent is valid");
    let first = RootResourceIdentity::try_new(extent).expect("first root identity");
    let second = RootResourceIdentity::try_new(extent).expect("second root identity");
    let first_span = first
        .bind_relative_range(ByteRange::new(8, 16))
        .expect("first child span");
    let second_span = second
        .bind_relative_range(ByteRange::new(8, 16))
        .expect("second child span");

    assert_ne!(first.root_resource(), second.root_resource());
    assert_eq!(first_span.root_identity(), first);
    assert_eq!(second_span.root_identity(), second);
    assert_eq!(first.validate_bound_span(&first_span), Ok(()));
    assert_eq!(
        first.validate_bound_span(&second_span),
        Err(SpanValidationError::DifferentRoot {
            expected: first.root_resource(),
            actual: second.root_resource(),
        })
    );

    assert_eq!(extent.key().domain(), allocation_key.domain());
    assert_eq!(extent.key().local(), allocation_key.local());
    assert_eq!(extent.byte_offset(), 0);
    assert_eq!(extent.byte_len(), 64);
    assert_eq!(extent.guaranteed_alignment().get(), 8);
    assert_ne!(first.root_resource().get().get(), 0);
}

#[test]
fn child_alignment_is_conservative_and_empty_spans_do_not_overlap() {
    let allocation_key = key(AllocationDomainId::fresh(), 4);
    let extent =
        RootResourceExtent::try_new(allocation_key, 0, 64, 16).expect("the root extent is valid");
    let root = RootResourceIdentity::try_new(extent).expect("root identity");
    let empty = root
        .bind_relative_range(ByteRange::new(32, 0))
        .expect("empty span");
    let left = root
        .bind_relative_range(ByteRange::new(0, 32))
        .expect("left span");
    let right = root
        .bind_relative_range(ByteRange::new(32, 32))
        .expect("right span");

    assert_eq!(left.guaranteed_alignment().get(), 16);
    assert_eq!(
        root.bind_relative_range(ByteRange::new(8, 8))
            .expect("less-aligned child")
            .guaranteed_alignment()
            .get(),
        8
    );
    assert_eq!(empty.overlaps(&left), Ok(false));
    assert_eq!(empty.overlaps(&right), Ok(false));
    assert_eq!(left.byte_offset(), 0);
    assert_eq!(left.byte_len(), 32);
    assert!(!left.is_empty());
    assert_eq!(left.overlaps(&right), Ok(false));
    assert_eq!(left.contains(&empty), Ok(true));
    assert_eq!(left.contains(&right), Ok(false));
    assert_eq!(right.contains(&left), Ok(false));

    let root_again = RootResourceIdentity::try_new(extent).expect("second root identity");
    let same_range_other_root = root_again
        .bind_relative_range(ByteRange::new(0, 32))
        .expect("same range under another root");
    assert_eq!(left.overlaps(&same_range_other_root), Ok(false));
    assert_eq!(left.contains(&same_range_other_root), Ok(false));
}

#[test]
fn requested_identity_is_a_single_explicit_sum_type() {
    let allocation_key = key(AllocationDomainId::fresh(), 5);
    let range = ByteRange::new(8, 16);
    let root = RootResourceIdentity::try_new(
        RootResourceExtent::try_new(allocation_key, 0, 64, 8).expect("root extent"),
    )
    .expect("root identity");

    let raw = RequestedIdentity::Raw(range);
    let keyed = RequestedIdentity::Keyed {
        key: allocation_key,
        range,
    };
    let rooted = RequestedIdentity::Rooted {
        root: root.root_resource(),
        key: allocation_key,
        range,
    };

    assert_eq!(raw.range(), range);
    assert_eq!(raw.allocation_key(), None);
    assert_eq!(keyed.allocation_key(), Some(allocation_key));
    assert_eq!(rooted.root_resource(), Some(root.root_resource()));
}

#[test]
fn operation_context_retains_requested_metadata_and_bound_resolution() {
    let allocation_key = key(AllocationDomainId::fresh(), 6);
    let root = RootResourceIdentity::try_new(
        RootResourceExtent::try_new(allocation_key, 0, 64, 8).expect("root extent"),
    )
    .expect("root identity");
    let span: RootBoundSpan = root
        .bind_relative_range(ByteRange::new(16, 8))
        .expect("bound span");
    let requested = RequestedIdentity::Rooted {
        root: root.root_resource(),
        key: allocation_key,
        range: ByteRange::new(16, 8),
    };
    let context =
        StorageOperationContext::resolved(StorageOperation::ImportUniqueRoot, requested, span);
    let error = StorageOperationError::new(
        context,
        SpanValidationError::OutsideRootExtent {
            key: allocation_key,
            root_byte_offset: 0,
            root_byte_len: 64,
            requested_byte_offset: 64,
            requested_byte_len: 8,
        },
    );

    assert_eq!(error.context().requested(), requested);
    assert_eq!(
        error.context().operation(),
        StorageOperation::ImportUniqueRoot
    );
    assert_eq!(error.context().resolved_span(), Some(span));
    assert_eq!(
        error.source().to_string(),
        "requested span lies outside the root resource extent"
    );
    assert!(error.to_string().contains("import_unique_root"));

    let unresolved = StorageOperationContext::unresolved(
        StorageOperation::ClaimSplit,
        RequestedIdentity::Raw(ByteRange::new(0, 0)),
    );
    assert_eq!(unresolved.operation(), StorageOperation::ClaimSplit);
    assert_eq!(unresolved.resolved_span(), None);
    assert!(unresolved.to_string().contains("unresolved"));
    assert_eq!(StorageOperation::ClaimSplit.to_string(), "claim_split");
}
