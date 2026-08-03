use std::any::Any;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::{AllocationDomainId, AllocationId, BackendId};

use super::super::{
    import_unique_root, AllocationKey, BackendAllocation, ByteRange, ProviderCapabilities,
    ProviderKind, RequestedIdentity, RootBoundSpan, RootResourceExtent, StorageOperation,
    StorageOperationError,
};

#[derive(Debug)]
struct CountingAllocation {
    extent: RootResourceExtent,
    drops: Arc<AtomicUsize>,
}

impl Drop for CountingAllocation {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::Relaxed);
    }
}

unsafe impl BackendAllocation for CountingAllocation {
    fn root_extent(&self) -> RootResourceExtent {
        self.extent
    }

    fn provider_kind(&self) -> ProviderKind {
        BackendId::Cpu
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::none()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn allocation_key(local: u64) -> AllocationKey {
    AllocationKey::new(
        AllocationDomainId::fresh(),
        AllocationId::from_backend_id(local),
    )
}

fn valid_extent(local: u64) -> RootResourceExtent {
    RootResourceExtent::try_new(allocation_key(local), 0, 64, 8).expect("valid root extent")
}

fn counting_allocation(
    extent: RootResourceExtent,
) -> (Box<dyn BackendAllocation>, Arc<AtomicUsize>) {
    let drops = Arc::new(AtomicUsize::new(0));
    (
        Box::new(CountingAllocation {
            extent,
            drops: Arc::clone(&drops),
        }),
        drops,
    )
}

#[test]
fn import_builds_one_checked_owner_and_borrow_capabilities() {
    let (allocation, drops) = counting_allocation(valid_extent(1));
    let mut owner = import_unique_root(allocation).expect("unique root import");

    {
        let read = owner.as_ref();
        assert_eq!(read.span().byte_offset(), 0);
        assert_eq!(read.span().byte_len(), 64);
        assert_eq!(read.span().guaranteed_alignment().get(), 8);
        assert_eq!(read.root_identity(), read.span().root_identity());
    }

    {
        let write = owner.as_mut();
        assert_eq!(write.span().byte_len(), 64);
        assert_eq!(write.root_identity(), write.span().root_identity());
    }

    drop(owner);
    assert_eq!(drops.load(Ordering::Relaxed), 1);
}

#[test]
fn invalid_import_returns_unresolved_typed_diagnostic_without_drop_of_missing_owner() {
    let key = allocation_key(2);
    let malformed = RootResourceExtent::test_corrupt(
        key,
        0,
        64,
        std::num::NonZeroUsize::new(3).expect("nonzero malformed alignment"),
    );
    let (allocation, drops) = counting_allocation(malformed);

    let error: Box<StorageOperationError<_>> = match import_unique_root(allocation) {
        Ok(_) => panic!("invalid extent must be rejected"),
        Err(error) => error,
    };
    assert_eq!(
        error.context().operation(),
        StorageOperation::ImportUniqueRoot
    );
    assert_eq!(error.context().resolved_span(), None);
    assert_eq!(
        error.context().requested(),
        RequestedIdentity::Keyed {
            key,
            range: ByteRange::new(0, 64),
        }
    );
    assert!(error.to_string().contains("import_unique_root"));
    assert_eq!(drops.load(Ordering::Relaxed), 1);
}

#[test]
fn imported_owner_drops_provider_allocation_exactly_once() {
    let (allocation, drops) = counting_allocation(valid_extent(3));
    let owner = import_unique_root(allocation).expect("unique root import");
    drop(owner);
    assert_eq!(drops.load(Ordering::Relaxed), 1);
}

#[test]
fn root_span_is_the_full_checked_extent() {
    let extent = valid_extent(4);
    let (allocation, _) = counting_allocation(extent);
    let owner = import_unique_root(allocation).expect("unique root import");
    let span: RootBoundSpan = owner.as_ref().span();
    assert_eq!(span.byte_offset(), extent.byte_offset());
    assert_eq!(span.byte_len(), extent.byte_len());
    assert_eq!(span.guaranteed_alignment(), extent.guaranteed_alignment());
}
