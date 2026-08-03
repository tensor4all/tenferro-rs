use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::{AllocationDomainId, AllocationId, BackendId, DType, DynRank, TensorScalar};

use super::super::import_unique_root;
use super::super::prepared::{
    prepare_read, prepare_write, AccessError, AccessTarget, CheckedLayout, CheckedRead,
    CheckedWrite, PreparedRead, PreparedWrite, ProviderReadMapping, ProviderWriteMapping,
};
use super::super::{
    AllocationKey, BackendAllocation, ProviderCapabilities, ProviderKind, RootResourceExtent,
};

#[derive(Debug)]
struct ByteAllocation {
    extent: RootResourceExtent,
    bytes: Mutex<Vec<u8>>,
    reads: Arc<AtomicUsize>,
    writes: Arc<AtomicUsize>,
}

unsafe impl BackendAllocation for ByteAllocation {
    fn root_extent(&self) -> RootResourceExtent {
        self.extent
    }

    fn provider_kind(&self) -> ProviderKind {
        BackendId::Cpu
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::none()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn map_read(
        &self,
        span: super::super::RootBoundSpan,
        dtype: DType,
    ) -> Result<ProviderReadMapping<'_>, AccessError> {
        if dtype != DType::Bool {
            return Err(AccessError::DTypeMismatch {
                expected: DType::Bool,
                actual: dtype,
            });
        }
        self.reads.fetch_add(1, Ordering::Relaxed);
        let guard = self.bytes.lock().expect("byte allocation lock");
        if guard.len() != span.byte_len() {
            return Err(AccessError::LengthMismatch {
                expected: span.byte_len(),
                actual: guard.len(),
            });
        }
        Ok(ProviderReadMapping::from_guard(guard))
    }

    fn map_write(
        &self,
        span: super::super::RootBoundSpan,
        dtype: DType,
    ) -> Result<ProviderWriteMapping<'_>, AccessError> {
        if dtype != DType::Bool {
            return Err(AccessError::DTypeMismatch {
                expected: DType::Bool,
                actual: dtype,
            });
        }
        self.writes.fetch_add(1, Ordering::Relaxed);
        let guard = self.bytes.lock().expect("byte allocation lock");
        if guard.len() != span.byte_len() {
            return Err(AccessError::LengthMismatch {
                expected: span.byte_len(),
                actual: guard.len(),
            });
        }
        Ok(ProviderWriteMapping::from_guard(guard))
    }
}

fn key(local: u64) -> AllocationKey {
    AllocationKey::new(
        AllocationDomainId::fresh(),
        AllocationId::from_backend_id(local),
    )
}

fn byte_owner(
    local: u64,
    bytes: Vec<u8>,
) -> (
    super::super::root::OwnedStorage,
    Arc<AtomicUsize>,
    Arc<AtomicUsize>,
) {
    let extent =
        RootResourceExtent::try_new(key(local), 0, bytes.len(), 1).expect("byte root extent");
    let reads = Arc::new(AtomicUsize::new(0));
    let writes = Arc::new(AtomicUsize::new(0));
    let allocation = ByteAllocation {
        extent,
        bytes: Mutex::new(bytes),
        reads: Arc::clone(&reads),
        writes: Arc::clone(&writes),
    };
    (
        import_unique_root(Box::new(allocation)).expect("byte root import"),
        reads,
        writes,
    )
}

#[test]
fn checked_layout_rejects_invalid_dtype_and_out_of_span_before_mapping() {
    let (owner, reads, _) = byte_owner(10, vec![1, 0, 1, 0]);
    let span = owner.as_ref().span();
    let invalid = CheckedRead::<DynRank>::new::<bool>(
        owner.as_ref(),
        span,
        vec![5].into(),
        vec![1].into(),
        0,
    );
    assert!(matches!(invalid, Err(AccessError::InvalidLayout { .. })));
    assert_eq!(reads.load(Ordering::Relaxed), 0);

    let checked_bool = CheckedRead::<DynRank>::new::<bool>(
        owner.as_ref(),
        span,
        vec![4].into(),
        vec![1].into(),
        0,
    )
    .expect("checked bool read");
    let wrong_dtype = prepare_read::<f64, DynRank>(checked_bool, AccessTarget::Host);
    assert!(matches!(
        wrong_dtype,
        Err(ref error) if matches!(&error.1, AccessError::DTypeMismatch { .. })
    ));
    assert_eq!(reads.load(Ordering::Relaxed), 0);
    drop(wrong_dtype);

    let mut owner = owner;
    let overlapping = CheckedWrite::<DynRank>::new::<bool>(
        owner.as_mut(),
        span,
        vec![2, 2].into(),
        vec![1, 1].into(),
        0,
    );
    assert!(matches!(overlapping, Err(AccessError::InvalidLayout { .. })));
}

#[test]
fn prepared_contiguous_read_and_write_use_typed_slices() {
    let (owner, reads, _) = byte_owner(11, vec![1, 0, 1, 0]);
    let span = owner.as_ref().span();
    let checked = CheckedRead::<DynRank>::new::<bool>(
        owner.as_ref(),
        span,
        vec![4].into(),
        vec![1].into(),
        0,
    )
    .expect("checked read");
    let prepared: PreparedRead<'_, bool, DynRank> =
        prepare_read(checked, AccessTarget::Host).expect("prepared read");
    assert_eq!(
        prepared.as_slice().expect("contiguous slice"),
        &[true, false, true, false]
    );
    assert_eq!(reads.load(Ordering::Relaxed), 1);
    drop(prepared);

    let mut owner = owner;
    let checked = CheckedWrite::<DynRank>::new::<bool>(
        owner.as_mut(),
        span,
        vec![4].into(),
        vec![1].into(),
        0,
    )
    .expect("checked write");
    let mut prepared: PreparedWrite<'_, bool, DynRank> =
        prepare_write(checked, AccessTarget::Host).expect("prepared write");
    prepared.as_slice_mut().expect("mutable slice")[1] = true;
}

#[test]
fn prepared_strided_iterators_cover_reverse_and_empty_layouts() {
    let (owner, _, _) = byte_owner(12, vec![1, 0, 0, 1]);
    let span = owner.as_ref().span();
    let checked = CheckedRead::<DynRank>::new::<bool>(
        owner.as_ref(),
        span,
        vec![2, 2].into(),
        vec![1, -2].into(),
        2,
    )
    .expect("checked reverse read");
    let prepared: PreparedRead<'_, bool, DynRank> =
        prepare_read(checked, AccessTarget::Host).expect("prepared reverse read");
    let values: Vec<bool> = prepared
        .iter_strided()
        .expect("strided iterator")
        .copied()
        .collect();
    assert_eq!(values, [false, true, true, false]);

    let (empty_owner, _, _) = byte_owner(13, Vec::new());
    let empty = CheckedRead::<DynRank>::new::<bool>(
        empty_owner.as_ref(),
        empty_owner.as_ref().span(),
        vec![0].into(),
        vec![1].into(),
        0,
    )
    .expect("empty checked read");
    let empty: PreparedRead<'_, bool, DynRank> =
        prepare_read(empty, AccessTarget::Host).expect("empty prepared read");
    assert_eq!(empty.as_slice().expect("empty contiguous slice"), &[]);
}

#[test]
fn prepared_mutable_strided_iterator_requires_injective_layout() {
    let (mut owner, _, writes) = byte_owner(16, vec![0, 0, 0, 0]);
    let span = owner.as_ref().span();
    let checked = CheckedWrite::<DynRank>::new::<bool>(
        owner.as_mut(),
        span,
        vec![2, 2].into(),
        vec![1, -2].into(),
        2,
    )
    .expect("checked mutable reverse layout");
    let mut prepared: PreparedWrite<'_, bool, DynRank> =
        prepare_write(checked, AccessTarget::Host).expect("prepared mutable reverse layout");
    for (index, value) in prepared
        .iter_strided_mut()
        .expect("mutable strided iterator")
        .enumerate()
    {
        *value = index % 2 == 0;
    }
    assert_eq!(writes.load(Ordering::Relaxed), 1);
}

#[test]
fn provider_resolution_counts_do_not_depend_on_element_count() {
    let (small, small_reads, _) = byte_owner(14, vec![1]);
    let small = CheckedRead::<DynRank>::new::<bool>(
        small.as_ref(),
        small.as_ref().span(),
        vec![1].into(),
        vec![1].into(),
        0,
    )
    .expect("small checked read");
    let _: PreparedRead<'_, bool, DynRank> =
        prepare_read(small, AccessTarget::Host).expect("small prepared read");

    let (large, large_reads, _) = byte_owner(15, vec![1; 4096]);
    let large = CheckedRead::<DynRank>::new::<bool>(
        large.as_ref(),
        large.as_ref().span(),
        vec![4096].into(),
        vec![1].into(),
        0,
    )
    .expect("large checked read");
    let prepared: PreparedRead<'_, bool, DynRank> =
        prepare_read(large, AccessTarget::Host).expect("large prepared read");
    assert_eq!(prepared.as_slice().expect("large slice").len(), 4096);

    assert_eq!(small_reads.load(Ordering::Relaxed), 1);
    assert_eq!(large_reads.load(Ordering::Relaxed), 1);
}

#[allow(dead_code)]
fn _rank_marker<T: TensorScalar>() -> DType {
    T::dtype()
}

#[allow(dead_code)]
fn _layout_marker() -> CheckedLayout<DynRank> {
    CheckedLayout::Contiguous {
        element_range: 0..0,
    }
}
