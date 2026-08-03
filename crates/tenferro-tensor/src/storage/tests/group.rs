use std::any::Any;
use std::ops::{Deref, DerefMut, Range};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use crate::{AllocationDomainId, AllocationId, BackendId, DType, DynRank};

use super::super::group::{
    AllocationGroup, AllocationSlot, DescriptorInput, DescriptorSlot, DisjointViewError,
    ExtractError, GroupError,
};
use super::super::prepared::{AccessError, ProviderReadMapping, ProviderWriteMapping};
use super::super::{
    import_unique_root, BackendAllocation, ByteRange, ProviderCapabilities, ProviderKind,
    RootResourceExtent,
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
            shape.into(),
            strides.into(),
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
    assert_eq!(read.descriptor().element_size(), 4);
    assert_eq!(read.descriptor().element_count(), 4);
    assert!(read.descriptor().write_injective());
    assert_eq!(read.descriptor().layout().shape(), &[4]);
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
                vec![3].into(),
                vec![-1].into(),
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
