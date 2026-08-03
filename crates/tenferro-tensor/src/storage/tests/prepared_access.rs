use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::{AllocationDomainId, AllocationId, BackendId, DType, DynRank, TensorScalar};

use super::super::import_unique_root;
use super::super::prepared::{
    prepare_read, prepare_write, AccessError, AccessTarget, CheckedLayout, CheckedRead,
    CheckedWrite, PreparedHostRead, PreparedHostWrite, PreparedRead, PreparedWrite,
    ProviderReadMapping, ProviderWriteMapping,
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

#[derive(Debug)]
struct UnsupportedAllocation {
    extent: RootResourceExtent,
}

unsafe impl BackendAllocation for UnsupportedAllocation {
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
}

#[derive(Debug)]
struct WrongLengthAllocation {
    extent: RootResourceExtent,
    bytes: Mutex<Vec<u8>>,
}

#[derive(Debug)]
struct MisalignedGuard<'a>(std::sync::MutexGuard<'a, Vec<u8>>);

impl Deref for MisalignedGuard<'_> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.0[1..]
    }
}

impl DerefMut for MisalignedGuard<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0[1..]
    }
}

#[derive(Debug)]
struct MisalignedAllocation {
    extent: RootResourceExtent,
    bytes: Mutex<Vec<u8>>,
}

unsafe impl BackendAllocation for MisalignedAllocation {
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

    fn map_write(
        &self,
        _span: super::super::RootBoundSpan,
        _dtype: DType,
    ) -> Result<ProviderWriteMapping<'_>, AccessError> {
        Ok(ProviderWriteMapping::from_guard(MisalignedGuard(
            self.bytes.lock().expect("misaligned write lock"),
        )))
    }
}

unsafe impl BackendAllocation for WrongLengthAllocation {
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
        _span: super::super::RootBoundSpan,
        _dtype: DType,
    ) -> Result<ProviderReadMapping<'_>, AccessError> {
        Ok(ProviderReadMapping::from_guard(
            self.bytes.lock().expect("wrong-length read lock"),
        ))
    }

    fn map_write(
        &self,
        _span: super::super::RootBoundSpan,
        _dtype: DType,
    ) -> Result<ProviderWriteMapping<'_>, AccessError> {
        Ok(ProviderWriteMapping::from_guard(
            self.bytes.lock().expect("wrong-length write lock"),
        ))
    }
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
    byte_owner_with_alignment(local, bytes, 1)
}

fn byte_owner_with_alignment(
    local: u64,
    bytes: Vec<u8>,
    alignment: usize,
) -> (
    super::super::root::OwnedStorage,
    Arc<AtomicUsize>,
    Arc<AtomicUsize>,
) {
    let extent = RootResourceExtent::try_new(key(local), 0, bytes.len(), alignment)
        .expect("byte root extent");
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
fn provider_mapping_failures_return_the_checked_state_unchanged() {
    let extent = RootResourceExtent::try_new(key(21), 0, 4, 1).expect("wrong-length extent");
    let owner = import_unique_root(Box::new(WrongLengthAllocation {
        extent,
        bytes: Mutex::new(vec![0]),
    }))
    .expect("wrong-length root import");
    let checked = CheckedRead::<DynRank>::new::<bool>(
        owner.as_ref(),
        owner.as_ref().span(),
        vec![4].into(),
        vec![1].into(),
        0,
    )
    .expect("checked wrong-length read");
    let error = match prepare_read::<bool, DynRank>(checked, AccessTarget::Host) {
        Err(error) => error,
        Ok(_) => panic!("mapping must fail"),
    };
    assert!(matches!(error.1, AccessError::LengthMismatch { .. }));

    let extent = RootResourceExtent::try_new(key(23), 0, 4, 1).expect("wrong-length write extent");
    let mut owner = import_unique_root(Box::new(WrongLengthAllocation {
        extent,
        bytes: Mutex::new(vec![0]),
    }))
    .expect("wrong-length write root import");
    let span = owner.as_ref().span();
    let checked = CheckedWrite::<DynRank>::new::<bool>(
        owner.as_mut(),
        span,
        vec![4].into(),
        vec![1].into(),
        0,
    )
    .expect("checked wrong-length write");
    let error = match prepare_write::<bool, DynRank>(checked, AccessTarget::Host) {
        Err(error) => error,
        Ok(_) => panic!("write mapping must fail"),
    };
    assert!(matches!(error.1, AccessError::LengthMismatch { .. }));

    let extent = RootResourceExtent::try_new(key(22), 0, 4, 1).expect("unsupported extent");
    let owner = import_unique_root(Box::new(UnsupportedAllocation { extent }))
        .expect("unsupported root import");
    let checked = CheckedRead::<DynRank>::new::<bool>(
        owner.as_ref(),
        owner.as_ref().span(),
        vec![4].into(),
        vec![1].into(),
        0,
    )
    .expect("checked unsupported read");
    let error = match prepare_read::<bool, DynRank>(checked, AccessTarget::Host) {
        Err(error) => error,
        Ok(_) => panic!("provider unsupported"),
    };
    assert!(matches!(error.1, AccessError::Unsupported { .. }));

    let extent = RootResourceExtent::try_new(key(24), 0, 4, 1).expect("unsupported write extent");
    let mut owner = import_unique_root(Box::new(UnsupportedAllocation { extent }))
        .expect("unsupported write root import");
    let span = owner.as_ref().span();
    let checked = CheckedWrite::<DynRank>::new::<bool>(
        owner.as_mut(),
        span,
        vec![4].into(),
        vec![1].into(),
        0,
    )
    .expect("checked unsupported write");
    let error = match prepare_write::<bool, DynRank>(checked, AccessTarget::Host) {
        Err(error) => error,
        Ok(_) => panic!("provider write unsupported"),
    };
    assert!(matches!(error.1, AccessError::Unsupported { .. }));

    let extent = RootResourceExtent::try_new(key(25), 0, 8, 8).expect("misaligned extent");
    let mut owner = import_unique_root(Box::new(MisalignedAllocation {
        extent,
        bytes: Mutex::new(vec![0; 9]),
    }))
    .expect("misaligned root import");
    let span = owner.as_ref().span();
    let checked = CheckedWrite::<DynRank>::new::<f64>(
        owner.as_mut(),
        span,
        vec![1].into(),
        vec![1].into(),
        0,
    )
    .expect("checked misaligned write");
    let error = match prepare_write::<f64, DynRank>(checked, AccessTarget::Host) {
        Err(error) => error,
        Ok(_) => panic!("misaligned mapping must fail"),
    };
    assert!(matches!(error.1, AccessError::Misaligned { required: 8 }));
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

    let (aligned_owner, _, _) = byte_owner_with_alignment(17, vec![0; 4], 8);
    let length_mismatch = CheckedRead::<DynRank>::new::<f64>(
        aligned_owner.as_ref(),
        aligned_owner.as_ref().span(),
        vec![1].into(),
        vec![1].into(),
        0,
    );
    assert!(matches!(
        length_mismatch,
        Err(AccessError::LengthMismatch { .. })
    ));

    let (foreign_owner, _, _) = byte_owner(20, vec![0, 0, 0, 0]);
    let foreign_span = CheckedRead::<DynRank>::new::<bool>(
        owner.as_ref(),
        foreign_owner.as_ref().span(),
        vec![4].into(),
        vec![1].into(),
        0,
    );
    assert!(matches!(
        foreign_span,
        Err(AccessError::InvalidLayout { .. })
    ));

    let mut owner = owner;
    let overlapping = CheckedWrite::<DynRank>::new::<bool>(
        owner.as_mut(),
        span,
        vec![2, 2].into(),
        vec![1, 1].into(),
        0,
    );
    assert!(matches!(
        overlapping,
        Err(AccessError::InvalidLayout { .. })
    ));
}

#[test]
fn mapping_wrappers_and_checked_descriptor_views_are_private_and_borrowed() {
    let read_bytes = [1_u8, 2, 3];
    let read_mapping = ProviderReadMapping::from_slice(&read_bytes);
    assert_eq!(read_mapping.bytes(), &read_bytes);
    assert!(format!("{read_mapping:?}").contains("byte_len: 3"));

    let mut write_bytes = [0_u8, 0, 0];
    let mut write_mapping = ProviderWriteMapping::from_slice(&mut write_bytes);
    assert_eq!(write_mapping.len(), 3);
    write_mapping.bytes_mut()[1] = 7;
    assert!(format!("{write_mapping:?}").contains("byte_len: 3"));
    drop(write_mapping);
    assert_eq!(write_bytes, [0, 7, 0]);

    let (owner, _, _) = byte_owner(18, vec![1, 0, 1, 0, 1]);
    let span = owner.as_ref().span();
    let checked = CheckedRead::<DynRank>::new::<bool>(
        owner.as_ref(),
        span,
        vec![2, 2].into(),
        vec![1, 3].into(),
        0,
    )
    .expect("checked descriptor");
    assert_eq!(checked.descriptor().span(), span);
    assert_eq!(checked.descriptor().dtype(), DType::Bool);
    assert_eq!(checked.descriptor().element_size(), 1);
    assert_eq!(checked.descriptor().layout().element_count(), 4);
    match checked.descriptor().layout() {
        CheckedLayout::Strided(plan) => {
            assert_eq!(plan.shape(), &[2, 2]);
            assert_eq!(plan.strides(), &[1, 3]);
            assert_eq!(plan.carry(), &[-1, -3]);
            assert_eq!(plan.offset(), 0);
            assert_eq!(plan.element_count(), 4);
        }
        CheckedLayout::Contiguous { .. } => panic!("expected strided descriptor"),
    }
    assert!(format!("{checked:?}").contains("CheckedRead"));
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
    if let PreparedRead::Host(PreparedHostRead::Contiguous(access)) = &prepared {
        assert_eq!(
            access.iter_contiguous().copied().collect::<Vec<_>>(),
            [true, false, true, false]
        );
    } else {
        panic!("expected contiguous read");
    }
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
    assert_eq!(
        checked.descriptor().proof(),
        super::super::prepared::WriteInjectivityProof
    );
    assert!(format!("{checked:?}").contains("CheckedWrite"));
    let mut prepared: PreparedWrite<'_, bool, DynRank> =
        prepare_write(checked, AccessTarget::Host).expect("prepared write");
    if let PreparedWrite::Host(PreparedHostWrite::Contiguous(access)) = &mut prepared {
        *access.iter_contiguous_mut().nth(1).expect("second element") = true;
    } else {
        panic!("expected contiguous write");
    }
    assert!(prepared.as_slice_mut().is_some());
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
fn device_preparation_keeps_host_access_unavailable() {
    let (mut owner, _, _) = byte_owner(19, vec![1, 0]);
    let span = owner.as_ref().span();
    let checked = CheckedRead::<DynRank>::new::<bool>(
        owner.as_ref(),
        span,
        vec![2].into(),
        vec![1].into(),
        0,
    )
    .expect("device checked read");
    let device: PreparedRead<'_, bool, DynRank> =
        prepare_read(checked, AccessTarget::Device).expect("device prepared read");
    assert!(device.as_slice().is_none());
    assert!(device.iter_strided().is_none());
    drop(device);

    let checked = CheckedWrite::<DynRank>::new::<bool>(
        owner.as_mut(),
        span,
        vec![2].into(),
        vec![1].into(),
        0,
    )
    .expect("device checked write");
    let mut device: PreparedWrite<'_, bool, DynRank> =
        prepare_write(checked, AccessTarget::Device).expect("device prepared write");
    assert!(device.as_slice_mut().is_none());
    assert!(device.iter_strided_mut().is_none());
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
