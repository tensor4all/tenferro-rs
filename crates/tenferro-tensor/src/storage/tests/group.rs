use std::any::Any;
use std::ops::{Deref, DerefMut, Range};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use crate::{AllocationDomainId, AllocationId, BackendId, DType, DynRank, ShapeVec, StrideVec};

use super::super::group::{
    AllocationGroup, AllocationSlot, DescriptorInput, DescriptorSlot, DisjointViewError,
    ExtractError, GroupError,
};
use super::super::identity::RootResourceIdentity;
use super::super::prepared::{
    validate_descriptor, AccessError, CheckedLayout, ProviderReadMapping, ProviderWriteMapping,
};
use super::super::{
    import_unique_root, AllocationKey, BackendAllocation, ByteRange, ProviderCapabilities,
    ProviderKind, RootResourceExtent,
};

struct RangeGuard<'a> {
    guard: MutexGuard<'a, Vec<u8>>,
    range: Range<usize>,
}

impl Deref for RangeGuard<'_> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.guard[self.range.clone()]
    }
}

impl DerefMut for RangeGuard<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.guard[self.range.clone()]
    }
}

#[derive(Debug)]
struct ByteAllocation {
    extent: RootResourceExtent,
    bytes: Mutex<Vec<u8>>,
    read_maps: Arc<AtomicUsize>,
    write_maps: Arc<AtomicUsize>,
}

impl ByteAllocation {
    fn build(
        local: u64,
        byte_len: usize,
    ) -> (
        Box<dyn BackendAllocation>,
        Arc<AtomicUsize>,
        Arc<AtomicUsize>,
    ) {
        let key = super::super::AllocationKey::new(
            AllocationDomainId::fresh(),
            AllocationId::from_backend_id(local),
        );
        let extent = RootResourceExtent::try_new(key, 0, byte_len, 8).expect("valid extent");
        let read_maps = Arc::new(AtomicUsize::new(0));
        let write_maps = Arc::new(AtomicUsize::new(0));
        (
            Box::new(Self {
                extent,
                bytes: Mutex::new(vec![0; byte_len]),
                read_maps: Arc::clone(&read_maps),
                write_maps: Arc::clone(&write_maps),
            }),
            read_maps,
            write_maps,
        )
    }

    fn range(&self, span: super::super::RootBoundSpan) -> Result<Range<usize>, AccessError> {
        let start = span
            .byte_offset()
            .checked_sub(self.extent.byte_offset())
            .ok_or_else(|| AccessError::Provider {
                message: "span precedes allocation".to_owned(),
            })?;
        let end = start
            .checked_add(span.byte_len())
            .ok_or_else(|| AccessError::Provider {
                message: "span end overflows allocation".to_owned(),
            })?;
        (end <= self.extent.byte_len())
            .then_some(start..end)
            .ok_or_else(|| AccessError::Provider {
                message: "span exceeds allocation".to_owned(),
            })
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

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn map_read(
        &self,
        span: super::super::RootBoundSpan,
        _dtype: DType,
    ) -> Result<ProviderReadMapping<'_>, AccessError> {
        self.read_maps.fetch_add(1, Ordering::Relaxed);
        let range = self.range(span)?;
        Ok(ProviderReadMapping::from_guard(RangeGuard {
            guard: self.bytes.lock().map_err(|_| AccessError::Provider {
                message: "byte allocation mutex poisoned".to_owned(),
            })?,
            range,
        }))
    }

    fn map_write(
        &self,
        span: super::super::RootBoundSpan,
        _dtype: DType,
    ) -> Result<ProviderWriteMapping<'_>, AccessError> {
        self.write_maps.fetch_add(1, Ordering::Relaxed);
        let range = self.range(span)?;
        Ok(ProviderWriteMapping::from_guard(RangeGuard {
            guard: self.bytes.lock().map_err(|_| AccessError::Provider {
                message: "byte allocation mutex poisoned".to_owned(),
            })?,
            range,
        }))
    }
}

fn owner(
    local: u64,
    byte_len: usize,
) -> (
    super::super::root::OwnedStorage,
    Arc<AtomicUsize>,
    Arc<AtomicUsize>,
) {
    let (allocation, read_maps, write_maps) = ByteAllocation::build(local, byte_len);
    (
        import_unique_root(allocation).expect("unique root import"),
        read_maps,
        write_maps,
    )
}

fn add_owner(
    group: &mut AllocationGroup,
    local: u64,
) -> (AllocationSlot, Arc<AtomicUsize>, Arc<AtomicUsize>) {
    let (owner, reads, writes) = owner(local, 64);
    (
        group.insert_owner(owner).expect("owner slot"),
        reads,
        writes,
    )
}

fn descriptor<T: crate::TensorScalar>(
    group: &mut AllocationGroup,
    allocation: AllocationSlot,
    offset: usize,
    byte_len: usize,
    shape: Vec<usize>,
    strides: Vec<isize>,
    require_injective: bool,
) -> Result<DescriptorSlot, GroupError> {
    group.insert_descriptor::<T, DynRank>(
        allocation,
        DescriptorInput::new(
            ByteRange::new(offset, byte_len),
            shape.iter().copied().collect(),
            strides.iter().copied().collect(),
            0,
            require_injective,
        ),
    )
}

#[test]
fn allocation_group_red_probe_names_the_p5_entry_point() {
    let _group = AllocationGroup::new();
}

#[test]
fn construction_retains_checked_metadata_and_rejects_bad_inputs() {
    let mut group = AllocationGroup::new();
    let (allocation, _, _) = add_owner(&mut group, 1);
    let slot = descriptor::<i32>(&mut group, allocation, 0, 16, vec![4], vec![1], true)
        .expect("valid descriptor");
    let read = group.view::<i32, DynRank>(slot).expect("read view");
    assert_eq!(read.descriptor().dtype(), DType::I32);
    assert_eq!(read.descriptor().allocation(), allocation);
    assert_eq!(read.descriptor().span().byte_len(), 16);
    assert_eq!(read.descriptor().element_size(), 4);
    assert_eq!(read.descriptor().element_count(), 4);
    assert_eq!(read.descriptor().provider(), BackendId::Cpu);
    assert!(read.descriptor().write_injective());
    assert_eq!(read.descriptor().layout().shape(), &[4]);
    assert!(format!("{read:?}").contains("GroupReadView"));
    {
        let write = group.view_mut::<i32, DynRank>(slot).expect("write view");
        assert_eq!(write.descriptor().allocation(), allocation);
        assert!(format!("{write:?}").contains("GroupWriteView"));
    }
    assert!(matches!(
        group.view::<i64, DynRank>(slot),
        Err(GroupError::DTypeMismatch { .. })
    ));
    assert!(matches!(
        group.view::<i32, crate::Rank<2>>(slot),
        Err(GroupError::RankMismatch { .. })
    ));
    assert!(matches!(
        group.view::<i32, DynRank>(DescriptorSlot::test_raw(u32::MAX)),
        Err(GroupError::DescriptorSlotOutOfBounds { .. })
    ));
    assert!(matches!(
        group.insert_descriptor::<i32, DynRank>(
            AllocationSlot::test_raw(u32::MAX),
            DescriptorInput::new(
                ByteRange::new(0, 4),
                vec![1].into_iter().collect(),
                vec![1].into_iter().collect(),
                0,
                true
            ),
        ),
        Err(GroupError::AllocationSlotOutOfBounds { .. })
    ));
    assert!(matches!(
        descriptor::<i32>(&mut group, allocation, 0, 6, vec![1], vec![1], true),
        Err(GroupError::InvalidDescriptor { .. })
    ));
    assert!(matches!(
        descriptor::<i32>(&mut group, allocation, 0, 4, vec![2], vec![1], true),
        Err(GroupError::InvalidDescriptor { .. })
    ));
    assert!(matches!(
        descriptor::<i32>(&mut group, allocation, 0, 8, vec![2], vec![0], true),
        Err(GroupError::InvalidDescriptor { .. })
    ));
    assert!(matches!(
        descriptor::<i64>(&mut group, allocation, 4, 8, vec![1], vec![1], true),
        Err(GroupError::InvalidDescriptor { .. })
    ));
    assert!(matches!(
        descriptor::<i32>(&mut group, allocation, 60, 8, vec![2], vec![1], true),
        Err(GroupError::InvalidDescriptor { .. })
    ));
}

#[test]
fn borrowed_views_map_only_the_selected_span() {
    let mut group = AllocationGroup::new();
    let (allocation, reads, writes) = add_owner(&mut group, 2);
    let first = descriptor::<i32>(&mut group, allocation, 0, 8, vec![2], vec![1], true)
        .expect("first descriptor");
    let second = descriptor::<i32>(&mut group, allocation, 8, 8, vec![2], vec![1], true)
        .expect("second descriptor");

    let first_read = group.view::<i32, DynRank>(first).expect("first read");
    let second_read = group.view::<i32, DynRank>(second).expect("second read");
    let first_mapping = first_read.map_read().expect("first mapping");
    assert_eq!(first_mapping.bytes().len(), 8);
    drop(first_mapping);
    let second_mapping = second_read.map_read().expect("second mapping");
    assert_eq!(second_mapping.bytes().len(), 8);
    drop(second_mapping);
    drop(first_read);
    drop(second_read);
    assert_eq!(reads.load(Ordering::Relaxed), 2);
    assert_eq!(writes.load(Ordering::Relaxed), 0);
}

#[test]
fn mutable_view_proves_injectivity_before_returning_a_child() {
    let mut group = AllocationGroup::new();
    let (allocation, _, _) = add_owner(&mut group, 9);
    let overlapping = descriptor::<i32>(&mut group, allocation, 0, 8, vec![2], vec![0], false)
        .expect("descriptor metadata can defer injectivity");
    assert!(matches!(
        group.view_mut::<i32, DynRank>(overlapping),
        Err(GroupError::InvalidDescriptor { .. })
    ));
}

#[test]
fn split_mut_accepts_zero_one_and_many_disjoint_children_without_mapping() {
    let mut group = AllocationGroup::new();
    let (allocation, _, writes) = add_owner(&mut group, 3);
    let first = descriptor::<i32>(&mut group, allocation, 0, 4, vec![1], vec![1], false)
        .expect("first descriptor");
    let second = descriptor::<i32>(&mut group, allocation, 4, 4, vec![1], vec![1], false)
        .expect("second descriptor");
    let third = descriptor::<i32>(&mut group, allocation, 8, 4, vec![1], vec![1], false)
        .expect("third descriptor");

    assert!(group
        .split_mut::<i32, DynRank>(&[])
        .expect("empty split")
        .is_empty());
    {
        let mut one = group
            .split_mut::<i32, DynRank>(&[first])
            .expect("one split");
        let mut mapping = one[0].map_write().expect("one mapping");
        assert_eq!(mapping.len(), 4);
        mapping.bytes_mut()[0] = 7;
    }
    {
        let many = group
            .split_mut::<i32, DynRank>(&[third, first, second])
            .expect("permuted many split");
        assert_eq!(many.len(), 3);
        assert!(many
            .iter()
            .all(|child| child.descriptor().write_injective()));
    }
    assert_eq!(writes.load(Ordering::Relaxed), 1);
}

#[test]
fn split_mut_rejects_overlap_and_preserves_the_group() {
    let mut group = AllocationGroup::new();
    let (allocation, _, _) = add_owner(&mut group, 4);
    let first = descriptor::<i32>(&mut group, allocation, 0, 8, vec![2], vec![1], true)
        .expect("first descriptor");
    let second = descriptor::<i32>(&mut group, allocation, 4, 8, vec![2], vec![1], true)
        .expect("second descriptor");
    assert!(matches!(
        group.split_mut::<i32, DynRank>(&[first, second]),
        Err(DisjointViewError::PairwiseOverlap)
    ));
    assert!(group.view::<i32, DynRank>(first).is_ok());
    assert!(matches!(
        group.split_mut::<i32, DynRank>(&[first, first]),
        Err(DisjointViewError::DuplicateSlot { .. })
    ));

    let (other_allocation, _, _) = add_owner(&mut group, 10);
    let other = descriptor::<i32>(&mut group, other_allocation, 0, 4, vec![1], vec![1], true)
        .expect("other descriptor");
    let other_children = group
        .split_mut::<i32, DynRank>(&[first, other])
        .expect("other root split");
    assert_eq!(other_children.len(), 2);
}

#[test]
fn split_mut_handles_empty_and_reverse_layouts() {
    let mut group = AllocationGroup::new();
    let (allocation, _, _) = add_owner(&mut group, 5);
    let reverse = group
        .insert_descriptor::<i32, DynRank>(
            allocation,
            DescriptorInput::new(
                ByteRange::new(0, 12),
                vec![3].into_iter().collect(),
                vec![-1].into_iter().collect(),
                2,
                false,
            ),
        )
        .expect("reverse descriptor");
    let empty = descriptor::<i32>(&mut group, allocation, 12, 0, vec![0], vec![1], false)
        .expect("empty descriptor");
    let children = group
        .split_mut::<i32, DynRank>(&[reverse, empty])
        .expect("empty and reverse split");
    assert_eq!(children.len(), 2);
    assert!(children[1].descriptor().envelope().is_none());
}

#[test]
fn extraction_is_structural_and_aliased_failures_are_unchanged() {
    let mut group = AllocationGroup::new();
    let (allocation, _, _) = add_owner(&mut group, 6);
    let first = descriptor::<i32>(&mut group, allocation, 0, 4, vec![1], vec![1], true)
        .expect("first descriptor");
    let _second = descriptor::<i32>(&mut group, allocation, 4, 4, vec![1], vec![1], true)
        .expect("second descriptor");
    assert!(matches!(
        group.try_extract(first),
        Err(ExtractError::AliasedAllocation { .. })
    ));
    assert!(group.view::<i32, DynRank>(first).is_ok());

    let mut sole_group = AllocationGroup::new();
    let (sole_allocation, _, _) = add_owner(&mut sole_group, 8);
    let sole = descriptor::<i32>(
        &mut sole_group,
        sole_allocation,
        0,
        4,
        vec![1],
        vec![1],
        true,
    )
    .expect("sole descriptor");
    let owner = sole_group
        .try_extract(sole)
        .expect("sole descriptor extraction");
    assert!(matches!(
        sole_group.view::<i32, DynRank>(sole),
        Err(GroupError::DescriptorSlotVacant { .. })
    ));
    drop(owner);
}

#[test]
fn consuming_extraction_returns_unchanged_group_on_failure() {
    let mut group = AllocationGroup::new();
    let (allocation, _, _) = add_owner(&mut group, 7);
    let slot = descriptor::<i32>(&mut group, allocation, 0, 4, vec![1], vec![1], true)
        .expect("descriptor");
    let (group, error) = match group.into_owner(DescriptorSlot::test_raw(u32::MAX)) {
        Ok(_) => panic!("invalid slot must be rejected"),
        Err(parts) => parts,
    };
    assert!(matches!(
        error,
        ExtractError::Group(GroupError::DescriptorSlotOutOfBounds { .. })
    ));
    assert!(group.view::<i32, DynRank>(slot).is_ok());
}

#[test]
fn consuming_extraction_moves_the_selected_owner_and_drops_the_remainder() {
    let mut group = AllocationGroup::new();
    let (allocation, _, _) = add_owner(&mut group, 11);
    let slot = descriptor::<i32>(&mut group, allocation, 0, 4, vec![1], vec![1], true)
        .expect("descriptor");
    let owner = match group.into_owner(slot) {
        Ok(owner) => owner,
        Err(_) => panic!("sole owner extraction must succeed"),
    };
    drop(owner);
}

#[test]
fn checked_group_helpers_report_overflow_without_panicking() {
    assert!(matches!(
        super::super::group::test_logical_element_count(&[usize::MAX, 2]),
        Err(GroupError::InvalidDescriptor { .. })
    ));

    let key = super::super::AllocationKey::new(
        AllocationDomainId::fresh(),
        AllocationId::from_backend_id(12),
    );
    let extent = super::super::RootResourceExtent::try_new(key, 0, usize::MAX, 1)
        .expect("maximum non-overflowing root extent");
    let identity = super::super::RootResourceIdentity::try_new(extent).expect("valid identity");
    let layout = crate::TensorLayout::<DynRank>::from_parts(
        vec![2].into_iter().collect(),
        vec![1].into_iter().collect(),
        0,
        2,
    )
    .expect("small layout");
    assert!(matches!(
        super::super::group::test_reachable_envelope(identity.root_span(), layout, usize::MAX),
        Err(GroupError::InvalidDescriptor { .. })
    ));

    let end_overflow = super::super::RootResourceExtent::try_new(
        super::super::AllocationKey::new(
            AllocationDomainId::fresh(),
            AllocationId::from_backend_id(13),
        ),
        usize::MAX,
        0,
        1,
    )
    .expect("zero-length maximum-offset extent");
    let end_identity = super::super::RootResourceIdentity::try_new(end_overflow)
        .expect("valid maximum-offset identity");
    let one = crate::TensorLayout::<DynRank>::from_parts(
        vec![1].into_iter().collect(),
        vec![1].into_iter().collect(),
        0,
        1,
    )
    .expect("one-element layout");
    assert!(matches!(
        super::super::group::test_reachable_envelope(end_identity.root_span(), one, 1),
        Err(GroupError::InvalidDescriptor { .. })
    ));
}

#[test]
fn vacant_allocation_entries_are_rejected_without_panicking() {
    let mut group = AllocationGroup::new();
    let (allocation, _, _) = add_owner(&mut group, 14);
    let slot = descriptor::<i32>(&mut group, allocation, 0, 4, vec![1], vec![1], true)
        .expect("descriptor");
    group.test_vacate_allocation(allocation);
    assert!(matches!(
        group.view::<i32, DynRank>(slot),
        Err(GroupError::AllocationSlotVacant { .. })
    ));
    assert!(matches!(
        group.view_mut::<i32, DynRank>(slot),
        Err(GroupError::AllocationSlotVacant { .. })
    ));
    assert!(matches!(
        group.split_mut::<i32, DynRank>(&[slot]),
        Err(DisjointViewError::Group(
            GroupError::AllocationSlotVacant { .. }
        ))
    ));
}

#[test]
fn host_group_constructor_retains_typed_layout_and_prepared_access() {
    let (group, slot) = AllocationGroup::from_host_vec::<i32, DynRank>(
        vec![2, 2].into_iter().collect(),
        vec![1, 2, 3, 4],
    )
    .expect("host group");
    let view = group.view::<i32, DynRank>(slot).expect("read view");
    assert_eq!(view.descriptor().layout().shape(), &[2, 2]);
    let prepared = view.prepare_host_read().expect("prepared read");
    assert_eq!(
        prepared.as_slice().expect("compact host slice"),
        &[1, 2, 3, 4]
    );
}

#[test]
fn set_placement_survives_the_allocation_group_round_trip() {
    let mut tensor =
        crate::TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0; 4]).unwrap();
    let placement = crate::Placement {
        memory_kind: crate::MemoryKind::PinnedHost,
        device: None,
        cpu_affinity: Some(crate::CpuDomainId::new(3)),
    };
    tensor.set_placement(placement.clone());
    let tensor = crate::Tensor::from_typed(tensor);

    let (mut group, slots) = AllocationGroup::from_tensors(vec![tensor]).unwrap();
    assert_eq!(
        group.read_view(slots[0]).unwrap().placement(),
        &placement,
        "read_view must report the placement the tensor entered the group with"
    );
    let extracted = group.take_tensor(slots[0]).unwrap();
    assert_eq!(
        extracted.placement(),
        &placement,
        "take_tensor must restore the placement set before the round trip"
    );
    assert_eq!(extracted.as_slice::<f64>().unwrap(), &[1.0; 4]);
}

#[test]
fn provider_view_host_mapping_uses_derived_layout_and_releases_guard() {
    let (allocation, reads, _) = ByteAllocation::build(901, 16);
    let mut tensor = crate::TypedTensor::<i32>::from_backend_allocation(
        [2, 2],
        allocation,
        crate::Placement {
            memory_kind: crate::MemoryKind::Managed,
            device: None,
            cpu_affinity: None,
        },
    )
    .unwrap();
    tensor
        .with_host_write(|data| data.copy_from_slice(&[1, 2, 3, 4]))
        .unwrap();
    let view = tensor
        .as_view()
        .slice_axis_view(1, crate::StridedSliceSpec::new(1, Some(2), 1))
        .unwrap();
    assert_eq!(
        view.with_host_read(|data| data.to_vec()).unwrap(),
        vec![3, 4]
    );
    assert_eq!(reads.load(Ordering::Relaxed), 1);
    let transposed = tensor.as_view().transpose_view([1, 0]).unwrap();
    assert!(transposed.with_host_read(|_| ()).is_err());
    assert_eq!(
        reads.load(Ordering::Relaxed),
        1,
        "reject unsupported layout before mapping"
    );
    drop((view, transposed));
    tensor.with_host_write(|data| data[0] = 9).unwrap();
    tensor
        .with_host_read(|data| assert_eq!(data, &[9, 2, 3, 4]))
        .unwrap();
    let placement = tensor.placement().clone();
    let (group, slots) =
        AllocationGroup::from_tensors(vec![crate::Tensor::from_typed(tensor)]).unwrap();
    assert_eq!(group.read_view(slots[0]).unwrap().placement(), &placement);
    assert_eq!(group.into_tensor(slots[0]).unwrap().placement(), &placement);
}

/// Allocator used by the descriptor-construction allocation witness.
///
/// It forwards every request to the system allocator and counts only while
/// `count_descriptor_allocations` has the thread-local flag raised, so the rest
/// of this test binary is unaffected. It is `cfg(test)`-only and matches the
/// counting-allocator precedent in `tests/borrowed_view_allocation_tests.rs`;
/// the witness has to live in this crate because `validate_descriptor` is
/// crate-private. No production path takes an allocator of ours.
struct DescriptorCountingAllocator;

thread_local! {
    static DESCRIPTOR_ALLOCATION_COUNTING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static DESCRIPTOR_ALLOCATION_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

unsafe impl std::alloc::GlobalAlloc for DescriptorCountingAllocator {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        if DESCRIPTOR_ALLOCATION_COUNTING.with(std::cell::Cell::get) {
            DESCRIPTOR_ALLOCATION_COUNT.with(|count| count.set(count.get() + 1));
        }
        // SAFETY: the unchanged layout is forwarded to the system allocator.
        unsafe { std::alloc::System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        // SAFETY: `ptr` and `layout` came from the matching system allocation.
        unsafe { std::alloc::System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: std::alloc::Layout, new_size: usize) -> *mut u8 {
        if DESCRIPTOR_ALLOCATION_COUNTING.with(std::cell::Cell::get) {
            DESCRIPTOR_ALLOCATION_COUNT.with(|count| count.set(count.get() + 1));
        }
        // SAFETY: `ptr` and `layout` came from the system allocator and
        // `new_size` is forwarded unchanged.
        unsafe { std::alloc::System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static DESCRIPTOR_COUNTING_ALLOCATOR: DescriptorCountingAllocator = DescriptorCountingAllocator;

fn count_descriptor_allocations(op: impl FnOnce()) -> usize {
    DESCRIPTOR_ALLOCATION_COUNT.with(|count| count.set(0));
    DESCRIPTOR_ALLOCATION_COUNTING.with(|flag| flag.set(true));
    op();
    DESCRIPTOR_ALLOCATION_COUNTING.with(|flag| flag.set(false));
    DESCRIPTOR_ALLOCATION_COUNT.with(std::cell::Cell::get)
}

#[test]
fn descriptor_plan_allocation_count_separates_contiguous_and_strided() {
    let extent = RootResourceExtent::try_new(
        AllocationKey::new(
            AllocationDomainId::fresh(),
            AllocationId::from_backend_id(7),
        ),
        0,
        64,
        8,
    )
    .expect("valid root extent");
    let identity = RootResourceIdentity::try_new(extent).expect("root identity");
    let span = identity.root_span();
    let root = identity.root_resource();

    // Both metadata sets are built outside the counted region, and both fit the
    // inline `SmallVec` capacity, so only the descriptor plan itself is counted.
    let compact_shape = vec![2, 3].into_iter().collect::<ShapeVec>();
    let compact_strides = vec![1, 2].into_iter().collect::<StrideVec>();
    let compact = count_descriptor_allocations(|| {
        let (checked, _) = validate_descriptor::<f64, DynRank>(
            root,
            span,
            compact_shape.clone(),
            compact_strides.clone(),
            0,
            false,
        )
        .expect("compact descriptor");
        assert!(matches!(checked.layout(), CheckedLayout::Contiguous { .. }));
    });

    let strided_shape = vec![2, 3].into_iter().collect::<ShapeVec>();
    let strided_strides = vec![1, 3].into_iter().collect::<StrideVec>();
    let strided = count_descriptor_allocations(|| {
        let (checked, _) = validate_descriptor::<f64, DynRank>(
            root,
            span,
            strided_shape.clone(),
            strided_strides.clone(),
            0,
            false,
        )
        .expect("strided descriptor");
        assert!(matches!(checked.layout(), CheckedLayout::Strided(_)));
    });

    assert_eq!(
        compact, 0,
        "a contiguous descriptor keeps its whole plan inline"
    );
    assert_eq!(
        strided, 1,
        "a strided descriptor owns exactly one boxed plan"
    );
}

/// Extraction must not change values: a strided view-shaped descriptor is
/// refused and the group keeps reading the view (#1938 PR review).
#[test]
fn non_compact_view_descriptors_are_refused_by_extraction_and_keep_their_values() {
    let base = || crate::Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]);
    let cases: Vec<(&str, crate::TensorValue, Vec<f64>)> = vec![
        (
            "transpose",
            crate::TensorValue::from_tensor(base().unwrap())
                .transpose_view([1, 0])
                .unwrap(),
            vec![1.0, 3.0, 2.0, 4.0],
        ),
        (
            "broadcast",
            crate::TensorValue::from_tensor(
                crate::Tensor::from_vec_col_major(vec![2], vec![5.0_f64, 6.0]).unwrap(),
            )
            .broadcast_in_dim_view([2, 2], [0])
            .unwrap(),
            vec![5.0, 6.0, 5.0, 6.0],
        ),
    ];
    for (name, value, expected) in cases {
        let Ok((group, slot, _, _)) = value.try_into_group_parts() else {
            panic!("{name}: a pooled value enters a group");
        };
        let (group, error) = match group.into_tensor(slot) {
            Ok(tensor) => panic!("{name}: extracted {:?}", tensor.as_slice::<f64>()),
            Err(failure) => failure,
        };
        assert_eq!(
            error,
            GroupError::NonCompactDescriptor { slot: slot.index() },
            "{name}"
        );
        // The unchanged group still reads the logical view values.
        let read = group.read_view(slot).unwrap();
        let mut values = Vec::new();
        let crate::TensorView::F64(view) = read.tensor_view() else {
            panic!("{name}: an f64 descriptor");
        };
        let shape = view.shape().to_vec();
        let mut index = vec![0usize; shape.len()];
        for _ in 0..shape.iter().product::<usize>() {
            values.push(*view.get(&index).unwrap());
            for (axis, extent) in shape.iter().enumerate() {
                index[axis] += 1;
                if index[axis] < *extent {
                    break;
                }
                index[axis] = 0;
            }
        }
        assert_eq!(values, expected, "{name}");
    }

    // A compact whole-allocation descriptor still extracts.
    let (group, slots) = AllocationGroup::from_tensors(vec![base().unwrap()]).unwrap();
    let tensor = group.into_tensor(slots[0]).unwrap();
    assert_eq!(tensor.as_slice::<f64>().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
}

/// The extracted owner of the compact `[1..3]` slice of a four-element `f64`
/// root (#2023): a host owner whose descriptor offset is one.
fn offset_slice_owner() -> crate::TypedTensor<f64> {
    let value = crate::TensorValue::from_tensor(
        crate::Tensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap(),
    )
    .slice_view(&crate::SliceConfig {
        starts: vec![1],
        limits: vec![3],
        strides: vec![1],
    })
    .unwrap();
    let (group, slot, _, _) = value
        .try_into_group_parts()
        .expect("a pooled value enters a group");
    let tensor = group
        .into_tensor(slot)
        .expect("a compact root-spanning slice extracts");
    <f64 as crate::TensorScalar>::into_typed(tensor).unwrap_or_else(|_| unreachable!())
}

/// A compact offset slice extracts by preserving its descriptor layout over the
/// whole root allocation: logical reads stay slice-bounded while the root is
/// retained so an in-place consumer can write only the slice (#2023).
#[test]
fn compact_slice_extraction_preserves_offset_extent_and_root() {
    let value = crate::TensorValue::from_tensor(
        crate::Tensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap(),
    )
    .slice_view(&crate::SliceConfig {
        starts: vec![1],
        limits: vec![3],
        strides: vec![1],
    })
    .unwrap();
    // The logical layout keeps the slice offset and extent before extraction.
    assert_eq!(value.shape(), &[2]);
    assert_eq!(value.offset(), 1);
    assert_eq!(value.tensor_read().as_slice::<f64>().unwrap(), &[2.0, 3.0]);
    let (group, slot, _, _) = value
        .try_into_group_parts()
        .expect("a pooled value enters a group");
    let tensor = group
        .into_tensor(slot)
        .expect("a compact root-spanning slice extracts");
    let mut typed =
        <f64 as crate::TensorScalar>::into_typed(tensor).unwrap_or_else(|_| unreachable!());
    // The extracted owner still retains the whole root allocation, reachable
    // through the typed view that carries the descriptor offset.
    assert_eq!(
        typed.as_view().host_storage().unwrap(),
        &[1.0, 2.0, 3.0, 4.0]
    );
    assert_eq!(typed.as_view().offset(), 1);
    // Every logical accessor reads the slice, never the root prefix.
    assert_eq!(typed.as_view().as_slice().unwrap(), &[2.0, 3.0]);
    assert_eq!(typed.as_slice().unwrap(), &[2.0, 3.0]);
    assert_eq!(typed.get(&[0]).unwrap(), &2.0);
    assert_eq!(typed.get(&[1]).unwrap(), &3.0);
    *typed.get_mut(&[1]).unwrap() = 30.0;
    assert_eq!(
        typed.as_view().host_storage().unwrap(),
        &[1.0, 2.0, 30.0, 4.0],
        "logical indexing must write the slice element and nothing outside it"
    );
    // The host mapping guards expose the same logical extent, so a
    // mapping-based consumer cannot read or overwrite the root prefix (#2023).
    assert_eq!(typed.map_read().unwrap().len(), 2);
    assert_eq!(&typed.map_read().unwrap()[..], &[2.0, 30.0]);
    typed
        .map_write()
        .unwrap()
        .copy_from_slice(&[2.0, 40.0])
        .unwrap_or_else(|_| unreachable!());
    assert_eq!(
        typed.as_view().host_storage().unwrap(),
        &[1.0, 2.0, 40.0, 4.0],
        "a mapping write must not touch the root outside the slice"
    );
    // The exclusive view applies the descriptor offset once and stays inside
    // the slice.
    {
        let mut view = typed.as_view_mut();
        assert_eq!(view.offset(), 1);
        *view.get_mut(&[1]).unwrap() = 50.0;
    }
    assert_eq!(
        typed.as_view().host_storage().unwrap(),
        &[1.0, 2.0, 50.0, 4.0]
    );
    assert_eq!(typed.host_data().unwrap(), &[2.0, 50.0]);
    // An explicit copy keeps the logical shape, extent and values.
    let copy = typed.duplicate().unwrap();
    assert_eq!(copy.shape(), &[2]);
    assert_eq!(copy.as_slice().unwrap(), &[2.0, 50.0]);
    // The parts export keeps the logical extent as well.
    let (buffer, layout, _placement) = copy.into_parts().expect("a host owner extracts its parts");
    assert_eq!(layout.shape(), &[2]);
    let crate::StorageBuffer::Host(data) = buffer else {
        unreachable!("an extracted host owner keeps its host root")
    };
    assert_eq!(data, vec![2.0, 50.0]);
    // The vector export returns the slice, not the root allocation.
    let (shape, data) = typed.into_vec_col_major().unwrap();
    assert_eq!(shape, vec![2]);
    assert_eq!(data, vec![2.0, 50.0]);
}

/// #2023: the parts export consumes the descriptor's logical extent too, so an
/// offset owner is not exported as its root.
#[test]
fn offset_owner_exports_its_logical_extent() {
    let typed = offset_slice_owner();
    let (buffer, layout, _placement) = typed.into_parts().expect("a host owner exports its parts");
    assert_eq!(layout.shape(), &[2]);
    let crate::StorageBuffer::Host(data) = buffer else {
        unreachable!("an extracted host owner keeps its host root")
    };
    assert_eq!(data, vec![2.0, 3.0]);
}

/// #2023: an owning reinterpretation rebases the descriptor over the same root,
/// and a value over the result duplicates like the original.
#[test]
fn offset_owner_owning_reinterpretation_round_trips() {
    let value = crate::TensorValue::from_tensor(
        crate::Tensor::from_vec_col_major(
            vec![4],
            (1..=4)
                .map(|x| crate::Complex64::new(x as f64, 0.0))
                .collect(),
        )
        .unwrap(),
    )
    .slice_view(&crate::SliceConfig {
        starts: vec![1],
        limits: vec![3],
        strides: vec![1],
    })
    .unwrap();
    let (group, slot, _, _) = value
        .try_into_group_parts()
        .expect("a pooled value enters a group");
    let tensor = group
        .into_tensor(slot)
        .expect("a compact root-spanning slice extracts");
    let typed = <crate::Complex64 as crate::TensorScalar>::into_typed(tensor)
        .unwrap_or_else(|_| unreachable!());
    let real = typed.into_real().unwrap_or_else(|_| unreachable!());
    let reals = [2.0_f64, 0.0, 3.0, 0.0];
    assert_eq!(real.host_data().unwrap(), &reals);
    let value = crate::TensorValue::from_tensor(crate::Tensor::from_typed(real));
    let duplicated = value
        .duplicate()
        .expect("a reinterpreted root-relative value duplicates");
    assert_eq!(duplicated.tensor_read().as_slice::<f64>().unwrap(), &reals);
}

/// #2023: a backend owner stays whole-root. The provider binding path reads the
/// root handle without a descriptor offset, so a compact slice of a managed or
/// device allocation must not extract - even when its placement metadata says
/// host.
#[test]
fn compact_slice_extraction_refuses_a_backend_owner() {
    let host_label = crate::Placement {
        memory_kind: crate::MemoryKind::UnpinnedHost,
        device: None,
        cpu_affinity: None,
    };
    for (case, relabel_as_host) in [("managed placement", false), ("host-labelled", true)] {
        let (allocation, _reads, _writes) = ByteAllocation::build(902, 32);
        let placement = crate::Placement {
            memory_kind: crate::MemoryKind::Managed,
            device: None,
            cpu_affinity: None,
        };
        let mut tensor =
            crate::TypedTensor::<i64>::from_backend_allocation([4], allocation, placement).unwrap();
        // The label is not authority: relabelling a backend root as host must
        // not make its compact slice extractable.
        if relabel_as_host {
            tensor.set_placement(host_label.clone());
        }
        let value = crate::TensorValue::from_tensor(crate::Tensor::from_typed(tensor))
            .slice_view(&crate::SliceConfig {
                starts: vec![1],
                limits: vec![3],
                strides: vec![1],
            })
            .unwrap();
        let (group, slot, _, _) = value
            .try_into_group_parts()
            .expect("a pooled value enters a group");
        assert!(
            matches!(
                group.into_tensor(slot),
                Err((_, GroupError::NonCompactDescriptor { .. }))
            ),
            "{case} must not extract"
        );
    }
}

/// #2023: a value whose layout is relative to a retained root duplicates that
/// backing root, not the exported prefix the owner's logical accessors expose.
#[test]
fn root_relative_value_duplicates_its_backing_root() {
    let value = crate::TensorValue::from_tensor(
        crate::Tensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap(),
    )
    .slice_view(&crate::SliceConfig {
        starts: vec![1],
        limits: vec![3],
        strides: vec![1],
    })
    .unwrap();
    let (group, slot, _, _) = value
        .try_into_group_parts()
        .expect("a pooled value enters a group");
    let tensor = group
        .into_tensor(slot)
        .expect("a compact root-spanning slice extracts");
    let typed = <f64 as crate::TensorScalar>::into_typed(tensor).unwrap_or_else(|_| unreachable!());
    let value = crate::TensorValue::from_tensor(crate::Tensor::from_typed(typed));
    assert_eq!(value.offset(), 1);
    let duplicated = value
        .duplicate()
        .expect("a root-relative value duplicates its backing storage");
    assert_eq!(duplicated.shape(), &[2]);
    assert_eq!(duplicated.offset(), 1);
    assert_eq!(
        duplicated.tensor_read().as_slice::<f64>().unwrap(),
        &[2.0, 3.0]
    );
    // The duplicate is the same kind of value as the original: both keep the
    // root-relative descriptor, so both convert to an owned tensor.
    assert_eq!(
        value.as_tensor().is_some(),
        duplicated.as_tensor().is_some()
    );
    assert!(duplicated.as_tensor().is_some());
    assert_eq!(
        duplicated.into_tensor().unwrap().as_slice::<f64>().unwrap(),
        &[2.0, 3.0]
    );
}

/// #2023: an exclusive real reinterpretation of a root-preserving complex owner
/// reads the same root region as the borrowed one, so the descriptor offset is
/// applied once.
#[test]
fn compact_slice_extraction_reinterprets_through_the_same_root() {
    let value = crate::TensorValue::from_tensor(
        crate::Tensor::from_vec_col_major(
            vec![4],
            (1..=4)
                .map(|x| crate::Complex64::new(x as f64, 0.0))
                .collect(),
        )
        .unwrap(),
    )
    .slice_view(&crate::SliceConfig {
        starts: vec![1],
        limits: vec![3],
        strides: vec![1],
    })
    .unwrap();
    let (group, slot, _, _) = value
        .try_into_group_parts()
        .expect("a pooled value enters a group");
    let tensor = group
        .into_tensor(slot)
        .expect("a compact root-spanning slice extracts");
    let mut typed = <crate::Complex64 as crate::TensorScalar>::into_typed(tensor)
        .unwrap_or_else(|_| unreachable!());
    let slice_reals = [2.0_f64, 0.0, 3.0, 0.0];
    assert_eq!(
        typed.as_real_view().unwrap().as_slice().unwrap(),
        &slice_reals
    );
    assert_eq!(
        typed
            .as_real_view_mut()
            .unwrap()
            .as_read_only()
            .as_slice()
            .unwrap(),
        &slice_reals,
        "the exclusive reinterpretation must not shift into the root prefix"
    );
}

/// An owning real reinterpretation keeps the complex root, so the descriptor's
/// dtype and the root's scalar type differ from then on. Both read routes must
/// see it: the extracted owner and the group's own read view.
#[test]
fn owning_reinterpretation_of_a_complex_root_reads_back() {
    let value = crate::TensorValue::from_tensor(
        crate::Tensor::from_vec_col_major(
            vec![4],
            (1..=4)
                .map(|x| crate::Complex64::new(x as f64, 0.0))
                .collect(),
        )
        .unwrap(),
    )
    .slice_view(&crate::SliceConfig {
        starts: vec![1],
        limits: vec![3],
        strides: vec![1],
    })
    .unwrap();
    let (group, slot, _, _) = value
        .try_into_group_parts()
        .expect("a pooled value enters a group");
    let tensor = group
        .into_tensor(slot)
        .expect("a compact root-spanning slice extracts");
    let typed = <crate::Complex64 as crate::TensorScalar>::into_typed(tensor)
        .unwrap_or_else(|_| unreachable!());
    let real = typed.into_real().unwrap_or_else(|_| unreachable!());
    let reals = [2.0_f64, 0.0, 3.0, 0.0];
    assert_eq!(real.host_data().unwrap(), &reals);

    // The extracted owner, then the same owner back in a group: the group read
    // resolves the descriptor's `f64` over a `Complex64` root.
    let value = crate::TensorValue::from_tensor(crate::Tensor::from_typed(real));
    assert_eq!(value.tensor_read().as_slice::<f64>().unwrap(), &reals);

    let (group, slot, _, _) = value
        .try_into_group_parts()
        .expect("the reinterpreted value enters a group again");
    let read = group
        .read_view(slot)
        .expect("the reinterpreted descriptor has backing storage");
    let crate::TensorView::F64(view) = read.tensor_view() else {
        panic!("the reinterpreted descriptor is f64");
    };
    assert_eq!(view.shape(), &[2, 2]);
    assert_eq!(
        crate::Tensor::from_typed(
            <f64 as crate::TensorScalar>::into_typed(group.into_tensor(slot).unwrap())
                .unwrap_or_else(|_| unreachable!())
        )
        .as_slice::<f64>()
        .unwrap(),
        &reals
    );
}
