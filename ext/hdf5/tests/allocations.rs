//! The write path allocates no payload-sized buffer, and a read allocates exactly one.
//!
//! Only allocations made on the measuring thread are counted, and the payload
//! threshold is half the payload, so HDF5's own conversion buffers (allocated
//! inside libhdf5 through libc) are outside the measurement by construction.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use tenferro_hdf5::{hdf5, read_typed, write_typed};
use tenferro_tensor::{DynRank, Host, TypedTensor};

const LEN: usize = 4096;
/// Half of one `f64` payload of `LEN` elements.
const PAYLOAD_HALF: usize = LEN * size_of::<f64>() / 2;

thread_local! {
    static ENABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
static LARGE_ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static LAST_LARGE_POINTER: AtomicUsize = AtomicUsize::new(0);
/// Payload-sized means at least half of one element payload of the measured dtype.
static THRESHOLD: AtomicUsize = AtomicUsize::new(0);

fn record(size: usize, pointer: *mut u8) {
    let enabled = ENABLED.try_with(std::cell::Cell::get).unwrap_or(false);
    if enabled && size >= THRESHOLD.load(Ordering::Relaxed) {
        LARGE_ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        LAST_LARGE_POINTER.store(pointer as usize, Ordering::Relaxed);
    }
}

struct CountingAllocator;

// SAFETY: every allocation is forwarded unchanged to `System`; the counters are
// read-only observations of the layout.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the layout is forwarded unchanged.
        let pointer = unsafe { System.alloc(layout) };
        record(layout.size(), pointer);
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: pointer and layout originate from `System`.
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        // SAFETY: the original allocation and the requested size are forwarded.
        let moved = unsafe { System.realloc(pointer, layout, size) };
        record(size, moved);
        moved
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serialize() -> std::sync::MutexGuard<'static, ()> {
    SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Run `f` while counting allocations of at least `threshold` bytes.
fn measure<T>(threshold: usize, f: impl FnOnce() -> T) -> (T, usize, usize) {
    struct Disable;
    impl Drop for Disable {
        fn drop(&mut self) {
            ENABLED.with(|enabled| enabled.set(false));
        }
    }
    LARGE_ALLOCATIONS.store(0, Ordering::Relaxed);
    LAST_LARGE_POINTER.store(0, Ordering::Relaxed);
    THRESHOLD.store(threshold, Ordering::Relaxed);
    ENABLED.with(|enabled| enabled.set(true));
    let disable = Disable;
    let value = f();
    drop(disable);
    (
        value,
        LARGE_ALLOCATIONS.load(Ordering::Relaxed),
        LAST_LARGE_POINTER.load(Ordering::Relaxed),
    )
}

fn fixture() -> (tempfile::TempDir, hdf5::File, hdf5::Group) {
    let dir = tempfile::tempdir().unwrap();
    let file = hdf5::File::create(dir.path().join("tensors.h5")).unwrap();
    let group = file.create_group("tensors").unwrap();
    (dir, file, group)
}

#[test]
fn the_counter_sees_one_payload_allocation() {
    let _serial = serialize();
    let (buffer, large, _) = measure::<Vec<f64>>(PAYLOAD_HALF, || vec![0.0_f64; LEN]);
    assert_eq!(large, 1);
    drop(buffer);
}

#[test]
fn writes_allocate_no_payload_buffer() {
    let _serial = serialize();
    let (_dir, _file, group) = fixture();
    let tensor = TypedTensor::<f64, DynRank, Host>::from_host_vec_col_major(
        vec![LEN],
        (0..LEN).map(|i| i as f64).collect(),
    )
    .unwrap();
    let (dataset, large, _) = measure::<hdf5::Dataset>(PAYLOAD_HALF, || {
        write_typed(&group, "f64", tensor.as_view()).unwrap()
    });
    assert_eq!(large, 0, "write_typed copied the payload");
    drop(dataset);

    // A bool payload is an eighth of the f64 one, so it needs its own
    // threshold: the f64 threshold would not see a bool-sized staging copy.
    let bool_half = LEN * size_of::<bool>() / 2;
    let (buffer, large, _) = measure::<Vec<bool>>(bool_half, || vec![true; LEN]);
    assert_eq!(
        large, 1,
        "the bool threshold misses a payload-sized allocation"
    );
    drop(buffer);

    let bools =
        TypedTensor::<bool, DynRank, Host>::from_host_vec_col_major(vec![LEN], vec![true; LEN])
            .unwrap();
    let (dataset, large, _) = measure::<hdf5::Dataset>(bool_half, || {
        write_typed(&group, "bool", bools.as_view()).unwrap()
    });
    assert_eq!(large, 0, "write_typed copied the bool payload");
    drop(dataset);
}

#[test]
fn reads_allocate_exactly_the_payload_once() {
    let _serial = serialize();
    let (_dir, _file, group) = fixture();
    let values: Vec<f64> = (0..LEN).map(|i| i as f64 * 0.5).collect();
    let tensor =
        TypedTensor::<f64, DynRank, Host>::from_host_vec_col_major(vec![LEN], values.clone())
            .unwrap();
    write_typed(&group, "f64", tensor.as_view()).unwrap();

    let (read, large, pointer) = measure::<TypedTensor<f64, DynRank, Host>>(PAYLOAD_HALF, || {
        read_typed::<f64, DynRank>(&group, "f64").unwrap()
    });
    assert_eq!(large, 1, "read_typed allocated {large} payload buffers");
    assert_eq!(read.as_slice(), values.as_slice());
    assert_eq!(
        read.as_slice().as_ptr() as usize,
        pointer,
        "the returned tensor does not own the allocation the reader made"
    );

    let bools = TypedTensor::<bool, DynRank, Host>::from_host_vec_col_major(
        vec![LEN],
        (0..LEN).map(|i| i % 2 == 0).collect(),
    )
    .unwrap();
    write_typed(&group, "bool", bools.as_view()).unwrap();
    let (read, large, pointer) = measure::<TypedTensor<bool, DynRank, Host>>(LEN / 2, || {
        read_typed::<bool, DynRank>(&group, "bool").unwrap()
    });
    assert_eq!(
        large, 1,
        "the bool reader allocated {large} payload buffers"
    );
    assert_eq!(read.as_slice(), bools.as_slice());
    assert_eq!(read.as_slice().as_ptr() as usize, pointer);
}
