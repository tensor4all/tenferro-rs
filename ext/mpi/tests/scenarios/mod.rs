//! Two-rank scenarios shared by both backend crates' test binaries.
//!
//! This module is `#[path]`-included by `ext/mpi/tests/mpi_p2p.rs` (crates.io `mpi`)
//! and `ext/mpi-rt/tests/mpi_p2p_rt.rs` (rsmpi-rt), so one scenario list proves both
//! backends. Each rank executes the same sequence in the same order; a scenario that
//! is a send on rank 0 is the matching receive on rank 1, and the roles swap on the
//! return leg.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

use tenferro_mpi::{engine, Error, MpiBackend};
use tenferro_tensor::{
    BackendStorageHandle, Complex32, Complex64, DType, DynRank, Dynamic, Host, Placement, Rank,
    StorageBuffer, Tensor, TypedTensor,
};

/// Both ranks' view of the pair.
pub struct Context<'a, B: MpiBackend> {
    /// The backend under test.
    pub backend: &'a B,
    /// This rank (0 or 1).
    pub rank: i32,
    /// The other rank's handle, usable as both a destination and a source.
    pub other: B::Peer<'a>,
    /// This rank's own handle.
    pub me: B::Peer<'a>,
}

impl<B: MpiBackend> Context<'_, B> {
    fn sends(&self) -> bool {
        self.rank == 0
    }
}

// ---------------------------------------------------------------- allocations

thread_local! {
    static ENABLED: Cell<bool> = const { Cell::new(false) };
}
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
static LARGEST: AtomicUsize = AtomicUsize::new(0);

struct CountingAllocator;

// SAFETY: every allocation, reallocation and deallocation is forwarded unchanged to
// `System`; the counters only observe the layout.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note(layout.size());
        // SAFETY: forwarded unchanged.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: pointer and layout originate from `System`.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        note(size);
        // SAFETY: forwarded unchanged.
        unsafe { System.realloc(ptr, layout, size) }
    }
}

fn note(size: usize) {
    if ENABLED.try_with(Cell::get).unwrap_or(false) {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(size, Ordering::Relaxed);
        LARGEST.fetch_max(size, Ordering::Relaxed);
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// Allocation totals observed while the flag was enabled on this thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AllocationReport {
    /// Number of allocations.
    pub count: usize,
    /// Total bytes allocated.
    pub bytes: usize,
    /// Largest single allocation.
    pub largest: usize,
}

/// Run `body` with the counter enabled and report what it allocated.
pub fn measure<R>(body: impl FnOnce() -> R) -> (R, AllocationReport) {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            ENABLED.with(|enabled| enabled.set(false));
        }
    }
    let _reset = Reset;
    ALLOCATIONS.store(0, Ordering::Relaxed);
    BYTES.store(0, Ordering::Relaxed);
    LARGEST.store(0, Ordering::Relaxed);
    ENABLED.with(|enabled| enabled.set(true));
    let value = body();
    ENABLED.with(|enabled| enabled.set(false));
    let report = AllocationReport {
        count: ALLOCATIONS.load(Ordering::Relaxed),
        bytes: BYTES.load(Ordering::Relaxed),
        largest: LARGEST.load(Ordering::Relaxed),
    };
    (value, report)
}

// ------------------------------------------------------------------ scenarios

/// Run every scenario, in a fixed order, on the pair.
pub fn run<B: MpiBackend>(context: &Context<'_, B>) -> Result<(), Error> {
    // One line per scenario before it runs: a hang names the scenario that hung.
    macro_rules! scenario {
        ($name:literal, $body:expr) => {{
            println!("rank {}: {}", context.rank, $name);
            $body?;
        }};
    }
    scenario!("builtin dtypes", roundtrip_builtin_dtypes(context));
    scenario!("special shapes", roundtrip_special_shapes(context));
    scenario!("crossings", roundtrip_crossings(context));
    scenario!("allocation budget", allocation_budget(context));
    scenario!("receiver rejects", receiver_rejects(context));
    scenario!("sender aborts", sender_aborts(context));
    scenario!("tag isolation", tag_isolation(context));
    scenario!("raw frames", raw_frames(context));
    Ok(())
}

/// Compare payloads by representation, so `NaN` and `-0.0` are still distinguished
/// from other values with the same `Debug` form.
fn same_bits<T: Copy>(left: &[T], right: &[T]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    let bytes = std::mem::size_of_val(left);
    // SAFETY: both slices stay alive for the comparison and are valid for the byte
    // range derived from their own length.
    let (left, right) = unsafe {
        (
            std::slice::from_raw_parts(left.as_ptr().cast::<u8>(), bytes),
            std::slice::from_raw_parts(right.as_ptr().cast::<u8>(), bytes),
        )
    };
    left == right
}

macro_rules! roundtrip {
    ($context:expr, $tag:expr, $ty:ty, $values:expr) => {{
        let context = $context;
        let shape = vec![2, 2];
        let values: Vec<$ty> = $values;
        if context.sends() {
            let value = TypedTensor::<$ty, DynRank, Host>::from_host_vec_col_major(
                shape.clone(),
                values.clone(),
            )?;
            engine::send(context.backend, context.other, $tag, &value)?;
        } else {
            let value: TypedTensor<$ty, DynRank, Host> =
                engine::recv(context.backend, context.other, $tag)?;
            assert!(
                same_bits(value.as_slice(), values.as_slice()),
                "round trip payload"
            );
            assert_eq!(value.shape(), &[2, 2], "round trip shape");
            assert!(
                same_bits(&[*value.get(&[1, 1])?], &[values[3]]),
                "column-major indexing"
            );
        }
    }};
}

fn roundtrip_builtin_dtypes<B: MpiBackend>(context: &Context<'_, B>) -> Result<(), Error> {
    roundtrip!(context, 101, f32, vec![1.5, -0.0, f32::INFINITY, f32::NAN]);
    roundtrip!(context, 102, f64, vec![1.5, -0.0, f64::INFINITY, f64::NAN]);
    roundtrip!(context, 103, i32, vec![1, -2, i32::MIN, i32::MAX]);
    roundtrip!(context, 104, i64, vec![1, -2, i64::MIN, i64::MAX]);
    roundtrip!(context, 105, bool, vec![true, false, true, true]);
    roundtrip!(
        context,
        106,
        Complex32,
        vec![
            Complex32::new(1.0, -2.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(3.5, 4.5),
            Complex32::new(-1.0, 1.0),
        ]
    );
    roundtrip!(
        context,
        107,
        Complex64,
        vec![
            Complex64::new(1.0, -2.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(3.5, 4.5),
            Complex64::new(-1.0, 1.0),
        ]
    );
    Ok(())
}

fn roundtrip_special_shapes<B: MpiBackend>(context: &Context<'_, B>) -> Result<(), Error> {
    // Rank 0: one element, shape [].
    if context.sends() {
        let value = TypedTensor::<f64, DynRank, Host>::from_host_vec_col_major(
            Vec::<usize>::new(),
            vec![7.5],
        )?;
        engine::send(context.backend, context.other, 111, &value)?;
    } else {
        let value: TypedTensor<f64, DynRank, Host> =
            engine::recv(context.backend, context.other, 111)?;
        assert_eq!(value.shape(), &[] as &[usize]);
        assert_eq!(value.as_slice(), &[7.5]);
    }

    // An empty extent carries no payload frame but preserves the shape.
    if context.sends() {
        let value =
            TypedTensor::<i64, DynRank, Host>::from_host_vec_col_major(vec![3, 0, 2], Vec::new())?;
        engine::send(context.backend, context.other, 112, &value)?;
    } else {
        let value: TypedTensor<i64, DynRank, Host> =
            engine::recv(context.backend, context.other, 112)?;
        assert_eq!(value.shape(), &[3, 0, 2]);
        assert!(value.as_slice().is_empty());
    }

    // A statically ranked receiver.
    if context.sends() {
        let value =
            TypedTensor::<f32, Rank<2>, Host>::from_host_vec_col_major([2, 3], vec![1.0; 6])?;
        engine::send(context.backend, context.other, 113, &value)?;
    } else {
        let value: TypedTensor<f32, Rank<2>, Host> =
            engine::recv(context.backend, context.other, 113)?;
        assert_eq!(value.shape(), &[2, 3]);
    }
    Ok(())
}

fn roundtrip_crossings<B: MpiBackend>(context: &Context<'_, B>) -> Result<(), Error> {
    // Typed sender, erased receiver.
    if context.sends() {
        let value =
            TypedTensor::<f64, DynRank, Host>::from_host_vec_col_major(vec![2], vec![1.0, 2.0])?;
        engine::send(context.backend, context.other, 121, &value)?;
    } else {
        let value: Tensor = engine::recv(context.backend, context.other, 121)?;
        assert_eq!(value.dtype(), DType::F64);
        assert_eq!(value.as_slice::<f64>()?, &[1.0, 2.0]);
    }

    // Erased sender, typed receiver.
    if context.sends() {
        let value = Tensor::from_typed(
            TypedTensor::<i32, DynRank, Host>::from_host_vec_col_major(vec![2], vec![3, 4])?
                .into_dynamic(),
        );
        engine::send(context.backend, context.other, 122, &value)?;
    } else {
        let value: TypedTensor<i32, DynRank, Host> =
            engine::recv(context.backend, context.other, 122)?;
        assert_eq!(value.as_slice(), &[3, 4]);
    }

    // Host-resident `Dynamic` send and receive.
    if context.sends() {
        let value: TypedTensor<bool, DynRank, Dynamic> =
            TypedTensor::<bool, DynRank, Host>::from_host_vec_col_major(
                vec![2],
                vec![true, false],
            )?
            .into_dynamic();
        engine::send(context.backend, context.other, 123, &value)?;
    } else {
        let value: TypedTensor<bool, DynRank, Dynamic> =
            engine::recv(context.backend, context.other, 123)?;
        assert_eq!(value.host_data()?, &[true, false]);
    }
    Ok(())
}

fn allocation_budget<B: MpiBackend>(context: &Context<'_, B>) -> Result<(), Error> {
    const LEN: usize = 131_072;
    let shape = vec![LEN];

    if context.sends() {
        let value = TypedTensor::<f64, DynRank, Host>::from_host_vec_col_major(
            shape.clone(),
            vec![1.25; LEN],
        )?;
        let erased = Tensor::from_typed(
            TypedTensor::<f64, DynRank, Host>::from_host_vec_col_major(shape, vec![1.25; LEN])?
                .into_dynamic(),
        );
        for tag in [131, 132] {
            let (result, report) =
                measure(|| engine::send(context.backend, context.other, tag, &value));
            result?;
            assert_eq!(
                report.count, 0,
                "sending a Host payload allocates: {report:?}"
            );
        }
        let (result, report) =
            measure(|| engine::send(context.backend, context.other, 133, &erased));
        result?;
        assert_eq!(
            report.count, 0,
            "sending an erased payload allocates: {report:?}"
        );
        // Drain the peer's three receive legs.
        engine::send(context.backend, context.other, 134, &value)?;
        let _: TypedTensor<f64, DynRank, Host> = engine::recv(context.backend, context.other, 135)?;
        return Ok(());
    }

    let mut out =
        TypedTensor::<f64, DynRank, Host>::from_host_vec_col_major(shape.clone(), vec![0.0; LEN])?;
    let (result, report) =
        measure(|| engine::recv_into(context.backend, context.other, 131, &mut out));
    result?;
    assert_eq!(report.count, 0, "recv_into allocates: {report:?}");
    assert!(out.as_slice().iter().all(|value| *value == 1.25));

    let mut erased = Tensor::from_typed(
        TypedTensor::<f64, DynRank, Host>::from_host_vec_col_major(shape.clone(), vec![0.0; LEN])?
            .into_dynamic(),
    );
    let (result, report) =
        measure(|| engine::recv_into(context.backend, context.other, 132, &mut erased));
    result?;
    assert_eq!(report.count, 0, "erased recv_into allocates: {report:?}");

    let mut dynamic: TypedTensor<f64, DynRank, Dynamic> = out.into_dynamic();
    let (result, report) =
        measure(|| engine::recv_into(context.backend, context.other, 133, &mut dynamic));
    result?;
    assert_eq!(report.count, 0, "Dynamic recv_into allocates: {report:?}");

    let (result, report) = measure(|| {
        engine::recv::<_, TypedTensor<f64, DynRank, Host>>(context.backend, context.other, 134)
    });
    let received = result?;
    assert_eq!(received.shape(), &[LEN]);
    assert!(
        report.largest >= LEN * std::mem::size_of::<f64>(),
        "recv must allocate the payload: {report:?}"
    );

    // Return leg: rank 1 sends one value that rank 0 receives.
    let value = TypedTensor::<f64, DynRank, Host>::from_host_vec_col_major(vec![1], vec![9.0])?;
    engine::send(context.backend, context.other, 135, &value)?;
    Ok(())
}

fn receiver_rejects<B: MpiBackend>(context: &Context<'_, B>) -> Result<(), Error> {
    // dtype mismatch: sender sends f64, receiver wants i32.
    if context.sends() {
        let value =
            TypedTensor::<f64, DynRank, Host>::from_host_vec_col_major(vec![2], vec![1.0, 2.0])?;
        match engine::send(context.backend, context.other, 141, &value) {
            Err(Error::PeerRejected { .. }) => {}
            other => panic!("expected PeerRejected for a dtype mismatch, got {other:?}"),
        }
    } else {
        let mut out =
            TypedTensor::<i32, DynRank, Host>::from_host_vec_col_major(vec![2], vec![7, 7])?;
        match engine::recv_into(context.backend, context.other, 141, &mut out) {
            Err(Error::DTypeMismatch { .. }) => {}
            other => panic!("expected DTypeMismatch, got {other:?}"),
        }
        assert_eq!(out.as_slice(), &[7, 7], "destination must be untouched");
    }

    // rank mismatch.
    if context.sends() {
        let value =
            TypedTensor::<f64, DynRank, Host>::from_host_vec_col_major(vec![2], vec![1.0, 2.0])?;
        match engine::send(context.backend, context.other, 142, &value) {
            Err(Error::PeerRejected { .. }) => {}
            other => panic!("expected PeerRejected for a rank mismatch, got {other:?}"),
        }
    } else {
        let mut out =
            TypedTensor::<f64, Rank<2>, Host>::from_host_vec_col_major([2, 1], vec![0.0, 0.0])?;
        match engine::recv_into(context.backend, context.other, 142, &mut out) {
            Err(Error::RankMismatch { .. }) => {}
            other => panic!("expected RankMismatch, got {other:?}"),
        }
    }

    // A device-backed destination: refused before the ack, so the sender hears it
    // instead of sending a payload nobody will receive.
    if context.sends() {
        let value =
            TypedTensor::<f64, DynRank, Host>::from_host_vec_col_major(vec![2], vec![1.0, 2.0])?;
        match engine::send(context.backend, context.other, 144, &value) {
            Err(Error::PeerRejected { reason }) => {
                assert_eq!(reason, tenferro_mpi::RejectReason::DestinationNotWritable)
            }
            other => panic!("expected DestinationNotWritable, got {other:?}"),
        }
    } else {
        let mut device = Tensor::from_typed(device_only_tensor());
        match engine::recv_into(context.backend, context.other, 144, &mut device) {
            Err(Error::NotHostResident) => {}
            other => panic!("expected NotHostResident, got {other:?}"),
        }
    }

    // shape mismatch, then recovery on the same tag.
    if context.sends() {
        let value =
            TypedTensor::<f64, DynRank, Host>::from_host_vec_col_major(vec![2], vec![1.0, 2.0])?;
        match engine::send(context.backend, context.other, 143, &value) {
            Err(Error::PeerRejected { .. }) => {}
            other => panic!("expected PeerRejected for a shape mismatch, got {other:?}"),
        }
        let good = TypedTensor::<f64, DynRank, Host>::from_host_vec_col_major(
            vec![3],
            vec![4.0, 5.0, 6.0],
        )?;
        engine::send(context.backend, context.other, 143, &good)?;
    } else {
        let mut out =
            TypedTensor::<f64, DynRank, Host>::from_host_vec_col_major(vec![3], vec![0.0; 3])?;
        match engine::recv_into(context.backend, context.other, 143, &mut out) {
            Err(Error::ShapeMismatch { .. }) => {}
            other => panic!("expected ShapeMismatch, got {other:?}"),
        }
        assert_eq!(
            out.as_slice(),
            &[0.0, 0.0, 0.0],
            "destination must be untouched"
        );
        engine::recv_into(context.backend, context.other, 143, &mut out)?;
        assert_eq!(out.as_slice(), &[4.0, 5.0, 6.0], "the tag must still work");
    }
    Ok(())
}

fn device_only_tensor() -> TypedTensor<f64, DynRank, Dynamic> {
    let handle = BackendStorageHandle::<f64>::new_with_len(1, 2);
    TypedTensor::<f64, DynRank, tenferro_tensor::Gpu>::from_backend_buffer_col_major(
        vec![2],
        StorageBuffer::Backend(Box::new(handle)),
        Placement::default(),
    )
    .expect("synthetic backend buffer")
    .into_dynamic()
}

fn sender_aborts<B: MpiBackend>(context: &Context<'_, B>) -> Result<(), Error> {
    if context.sends() {
        let device = device_only_tensor();
        match engine::send(context.backend, context.other, 151, &device) {
            Err(Error::NotHostResident) => {}
            other => panic!("expected NotHostResident, got {other:?}"),
        }
        let external = Tensor::external(tenferro_tensor::ErasedHostTensor::new(
            TypedTensor::<f64, DynRank, Host>::from_host_vec_col_major(vec![1], vec![1.0])
                .expect("host payload"),
        ));
        match engine::send(context.backend, context.other, 152, &external) {
            Err(Error::UnsupportedDType { .. }) => {}
            other => panic!("expected UnsupportedDType, got {other:?}"),
        }
        // Leave a clean message so the peer's second leg completes.
        let value = TypedTensor::<f64, DynRank, Host>::from_host_vec_col_major(vec![1], vec![0.0])?;
        engine::send(context.backend, context.other, 153, &value)?;
    } else {
        match engine::recv::<_, Tensor>(context.backend, context.other, 151) {
            Err(Error::PeerAborted { .. }) => {}
            other => panic!("expected PeerAborted, got {other:?}"),
        }
        match engine::recv::<_, Tensor>(context.backend, context.other, 152) {
            Err(Error::PeerAborted { .. }) => {}
            other => panic!("expected PeerAborted, got {other:?}"),
        }
        let _: Tensor = engine::recv(context.backend, context.other, 153)?;
    }
    Ok(())
}

fn tag_isolation<B: MpiBackend>(context: &Context<'_, B>) -> Result<(), Error> {
    // A self-send and a negative tag are refused before any message.
    let value = TypedTensor::<f64, DynRank, Host>::from_host_vec_col_major(vec![1], vec![1.0])?;
    match engine::send(context.backend, context.me, 161, &value) {
        Err(Error::SelfMessage) => {}
        other => panic!("expected SelfMessage, got {other:?}"),
    }
    match engine::send(context.backend, context.other, -1, &value) {
        Err(Error::InvalidTag { .. }) => {}
        other => panic!("expected InvalidTag, got {other:?}"),
    }

    // Two tags carry independent exchanges. A send blocks until its own tag's ack
    // arrives, so these legs are paired in order: the header/ack protocol cannot
    // overlap two tags in one call, which is why a caller needs a thread per tag.
    for (tag, value) in [(162, 20_i64), (163, 10)] {
        if context.sends() {
            let sent =
                TypedTensor::<i64, DynRank, Host>::from_host_vec_col_major(vec![1], vec![value])?;
            engine::send(context.backend, context.other, tag, &sent)?;
        } else {
            let received: TypedTensor<i64, DynRank, Host> =
                engine::recv(context.backend, context.other, tag)?;
            assert_eq!(received.as_slice(), &[value]);
        }
    }
    if context.sends() {
        let reply: TypedTensor<i64, DynRank, Host> =
            engine::recv(context.backend, context.other, 164)?;
        assert_eq!(reply.as_slice(), &[99]);
    } else {
        let reply = TypedTensor::<i64, DynRank, Host>::from_host_vec_col_major(vec![1], vec![99])?;
        engine::send(context.backend, context.other, 164, &reply)?;
    }
    Ok(())
}

/// A raw header frame, for peer behaviour the adapter itself never produces.
fn raw_header(
    version: u16,
    dtype_code: u8,
    byte_order: u8,
    elements: u64,
    shape: &[u64],
) -> Vec<u8> {
    let mut frame = Vec::new();
    frame.extend_from_slice(b"TFMP");
    frame.extend_from_slice(&version.to_le_bytes());
    frame.push(1);
    frame.push(dtype_code);
    frame.push(0);
    frame.push(byte_order);
    frame.extend_from_slice(&[0, 0]);
    frame.extend_from_slice(&(shape.len() as u32).to_le_bytes());
    frame.extend_from_slice(&elements.to_le_bytes());
    for extent in shape {
        frame.extend_from_slice(&extent.to_le_bytes());
    }
    frame
}

fn read_ack<B: MpiBackend>(context: &Context<'_, B>, tag: i32) -> u8 {
    let (probed, length) = context.backend.probe(context.other, tag);
    assert_eq!(length, 8, "ack length");
    let mut buffer = [0u8; 8];
    context.backend.receive_probed(probed, &mut buffer);
    buffer[7]
}

fn raw_frames<B: MpiBackend>(context: &Context<'_, B>) -> Result<(), Error> {
    let byte_order = if cfg!(target_endian = "little") { 1 } else { 2 };

    // Bad magic: no ack, and the receiver reports a malformed header.
    if context.sends() {
        context
            .backend
            .send_bytes(context.other, 171, b"XXXX-not-a-header");
        // The receiver answers nothing, so synchronize on the next scenario's leg to
        // keep this frame from racing the following one.
        let sync: TypedTensor<f64, DynRank, Host> =
            engine::recv(context.backend, context.other, 172)?;
        assert_eq!(sync.as_slice(), &[0.0]);
    } else {
        match engine::recv::<_, Tensor>(context.backend, context.other, 171) {
            Err(Error::MalformedHeader { .. }) => {}
            other => panic!("expected MalformedHeader, got {other:?}"),
        }
        let value = TypedTensor::<f64, DynRank, Host>::from_host_vec_col_major(vec![1], vec![0.0])?;
        engine::send(context.backend, context.other, 172, &value)?;
    }

    // Unsupported version: the receiver answers with a reject ack.
    if context.sends() {
        context
            .backend
            .send_bytes(context.other, 173, &raw_header(99, 5, byte_order, 1, &[1]));
        assert_eq!(read_ack(context, 173), 4, "UnsupportedVersion ack");
    } else {
        match engine::recv::<_, Tensor>(context.backend, context.other, 173) {
            Err(Error::UnsupportedVersion { received, .. }) => assert_eq!(received, 99),
            other => panic!("expected UnsupportedVersion, got {other:?}"),
        }
    }

    // A foreign byte order is rejected instead of mis-decoded.
    if context.sends() {
        let foreign = if byte_order == 1 { 2 } else { 1 };
        context
            .backend
            .send_bytes(context.other, 174, &raw_header(1, 5, foreign, 1, &[1]));
        assert_eq!(read_ack(context, 174), 8, "ByteOrderMismatch ack");
    } else {
        match engine::recv::<_, Tensor>(context.backend, context.other, 174) {
            Err(Error::ByteOrderMismatch { .. }) => {}
            other => panic!("expected ByteOrderMismatch, got {other:?}"),
        }
    }

    // An element count above one message is rejected, never truncated.
    if context.sends() {
        let too_many = i32::MAX as u64 + 1;
        context.backend.send_bytes(
            context.other,
            175,
            &raw_header(1, 5, byte_order, too_many, &[too_many]),
        );
        assert_eq!(read_ack(context, 175), 6, "TooManyElements ack");
    } else {
        match engine::recv::<_, Tensor>(context.backend, context.other, 175) {
            Err(Error::TooManyElements { .. }) => {}
            other => panic!("expected TooManyElements, got {other:?}"),
        }
    }

    // A bool payload may only contain 0 and 1; the destination keeps its own values
    // and the receiver reports the offending byte instead of storing it.
    if context.sends() {
        context
            .backend
            .send_bytes(context.other, 177, &raw_header(1, 1, byte_order, 2, &[2]));
        assert_eq!(read_ack(context, 177), 0, "a bool header is accepted");
        context.backend.send_bytes(context.other, 177, &[2, 0]);
    } else {
        let mut out =
            TypedTensor::<bool, DynRank, Host>::from_host_vec_col_major(vec![2], vec![true, true])?;
        match engine::recv_into(context.backend, context.other, 177, &mut out) {
            Err(Error::InvalidBoolPayload) => {}
            other => panic!("expected InvalidBoolPayload, got {other:?}"),
        }
        assert_eq!(out.as_slice(), &[true, true], "destination untouched");
    }

    // A header whose element count disagrees with its shape is rejected.
    if context.sends() {
        context
            .backend
            .send_bytes(context.other, 176, &raw_header(1, 5, byte_order, 5, &[2]));
        assert_eq!(read_ack(context, 176), 5, "MalformedHeader ack");
    } else {
        match engine::recv::<_, Tensor>(context.backend, context.other, 176) {
            Err(Error::MalformedHeader { .. }) => {}
            other => panic!("expected MalformedHeader, got {other:?}"),
        }
    }
    Ok(())
}
