//! Two-rank scenarios through the C entry points, with the test acting as the host.
//!
//! The host initializes MPI, gives tenferro a communicator of its own (`Comm_dup`), and
//! passes the raw handle. Run with `scripts/run-mpi.sh capi` (`mpiexec -n 2`); a single
//! process reports the skip. Needs `--features test-hooks`.

use std::ffi::c_void;
use std::os::raw::c_char;

use tenferro_mpi::mpi::ffi;
use tenferro_mpi::mpi::raw::AsRaw;
use tenferro_mpi::mpi::traits::*;
use tenferro_mpi_capi::*;

fn check(what: &str, status: tfe_status_t, expected: tfe_status_t) {
    if status != expected {
        let mut buf = vec![0u8; 4096];
        let mut needed = 0usize;
        unsafe { tfe_last_error_message(buf.as_mut_ptr(), buf.len(), &mut needed) };
        let msg = String::from_utf8_lossy(&buf[..needed.saturating_sub(1).min(4095)]).into_owned();
        panic!("{what}: status {status}, expected {expected}: {msg}");
    }
}

fn last_error() -> String {
    let mut buf = vec![0u8; 4096];
    let mut needed = 0usize;
    unsafe { tfe_last_error_message(buf.as_mut_ptr(), buf.len(), &mut needed) };
    String::from_utf8_lossy(&buf[..needed.saturating_sub(1).min(4095)]).into_owned()
}

fn text(f: impl Fn(*mut c_char, usize, *mut usize, *mut tfe_status_t) -> tfe_status_t) -> String {
    let mut needed = 0usize;
    let mut status = 0;
    let code = f(std::ptr::null_mut(), 0, &mut needed, &mut status);
    assert_eq!(code, TFE_SUCCESS, "size query: {}", last_error());
    let mut buf = vec![0 as c_char; needed];
    let code = f(buf.as_mut_ptr(), buf.len(), &mut needed, &mut status);
    assert_eq!(code, TFE_SUCCESS, "info: {}", last_error());
    let bytes: Vec<u8> = buf[..needed - 1].iter().map(|&c| c as u8).collect();
    String::from_utf8(bytes).unwrap()
}

fn field(info: &str, key: &str) -> String {
    info.lines()
        .find_map(|l| l.strip_prefix(&format!("{key}=")))
        .unwrap_or_else(|| panic!("no {key} in {info}"))
        .to_string()
}

struct Payload {
    code: u8,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

fn payloads() -> Vec<Payload> {
    fn bytes_of<T: Copy>(v: &[T]) -> Vec<u8> {
        unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)).to_vec() }
    }
    vec![
        Payload { code: 1, shape: vec![3], bytes: vec![1, 0, 1] },
        Payload { code: 2, shape: vec![2, 2], bytes: bytes_of(&[1i32, -2, 3, -4]) },
        Payload { code: 3, shape: vec![4], bytes: bytes_of(&[i64::MIN, -1, 0, i64::MAX]) },
        Payload { code: 4, shape: vec![2, 1], bytes: bytes_of(&[1.5f32, f32::NEG_INFINITY]) },
        Payload { code: 5, shape: vec![2, 3], bytes: bytes_of(&[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.5]) },
        Payload { code: 6, shape: vec![1], bytes: bytes_of(&[1.0f32, -2.0]) },
        Payload { code: 7, shape: vec![2], bytes: bytes_of(&[1.0f64, 2.0, 3.0, 4.0]) },
        Payload { code: 5, shape: vec![3, 0], bytes: Vec::new() },
        Payload { code: 3, shape: vec![], bytes: bytes_of(&[42i64]) },
    ]
}

fn main() {
    // D12: before any MPI library is loaded, the info call fails and does not stick.
    let mut status = 0;
    let code = unsafe { tfe_mpi_library_info(std::ptr::null_mut(), 0, std::ptr::null_mut(), &mut status) };
    check("library_info before MPI is loaded", code, TFE_MPI_NO_LIBRARY);
    assert!(last_error().contains("no MPI library is loaded"), "{}", last_error());

    // mpi4py's default is MPI_THREAD_MULTIPLE; `initialize()` would request SINGLE.
    let (universe, _provided) =
        tenferro_mpi::mpi::initialize_with_threading(tenferro_mpi::mpi::Threading::Multiple)
            .expect("MPI_Init_thread");
    let world = universe.world();
    let rank = world.rank();
    if world.size() < 2 {
        println!("rank {rank}: needs two ranks, skipping (run scripts/run-mpi.sh capi)");
        return;
    }
    assert_eq!(world.size(), 2);
    let other = 1 - rank;

    // The failure above was not cached.
    let info = text(|b, l, n, s| unsafe { tfe_mpi_library_info(b, l, n, s) });
    println!("rank {rank}: {}", info.replace('\n', " | "));

    // The host gives tenferro a communicator of its own.
    let dup = world.duplicate();
    let raw = dup.as_raw() as u64;
    let family = field(&info, "family");

    // Encodings refused before any MPI call (D4): null, MPI_COMM_NULL, mixed upper bits.
    let mut comm: *mut tfe_mpi_comm = std::ptr::null_mut();
    check("zero handle", unsafe { tfe_mpi_comm_attach(0, &mut comm, &mut status) }, TFE_INVALID_ARGUMENT);
    assert!(comm.is_null());
    check(
        "MPI_COMM_NULL",
        unsafe { tfe_mpi_comm_attach(ffi::RSMPI_COMM_NULL_fn() as u64, &mut comm, &mut status) },
        TFE_INVALID_ARGUMENT,
    );
    if family == "mpich-abi" {
        check(
            "mixed upper bits",
            unsafe { tfe_mpi_comm_attach(0x1_0000_0000 | raw, &mut comm, &mut status) },
            TFE_INVALID_ARGUMENT,
        );
        // mpi4py's comm.handle sign-extends a negative int handle.
        let sign_extended = (raw as u32 as i32) as i64 as u64;
        let mut c2: *mut tfe_mpi_comm = std::ptr::null_mut();
        check("sign-extended handle", unsafe { tfe_mpi_comm_attach(sign_extended, &mut c2, &mut status) }, TFE_SUCCESS);
        unsafe { tfe_mpi_comm_detach(c2, &mut status) };
    }

    check("attach", unsafe { tfe_mpi_comm_attach(raw, &mut comm, &mut status) }, TFE_SUCCESS);
    let cinfo = text(|b, l, n, s| unsafe { tfe_mpi_comm_info(comm, b, l, n, s) });
    assert_eq!(field(&cinfo, "rank"), rank.to_string());
    assert_eq!(field(&cinfo, "size"), "2");
    println!("rank {rank}: thread_level={} errhandler_fatal={}", field(&cinfo, "thread_level"), field(&cinfo, "errhandler_fatal"));

    // Rejections computed locally, so even a fatal error handler never sees them (D6).
    let one = [0u8; 8];
    let shape = [1usize];
    check("negative tag", unsafe { tfe_mpi_send(comm, other, -1, 5, shape.as_ptr(), 1, one.as_ptr().cast(), 8, &mut status) }, TFE_INVALID_ARGUMENT);
    // Open MPI's MPI_TAG_UB is INT_MAX, so a tag above it exists only on other libraries.
    let tag_ub: i64 = field(&cinfo, "tag_ub").parse().unwrap();
    if tag_ub < i32::MAX as i64 {
        check("tag above TAG_UB", unsafe { tfe_mpi_send(comm, other, (tag_ub + 1) as i32, 5, shape.as_ptr(), 1, one.as_ptr().cast(), 8, &mut status) }, TFE_INVALID_ARGUMENT);
    }
    check("rank out of range", unsafe { tfe_mpi_send(comm, 7, 1, 5, shape.as_ptr(), 1, one.as_ptr().cast(), 8, &mut status) }, TFE_INVALID_ARGUMENT);
    check("self send", unsafe { tfe_mpi_send(comm, rank, 1, 5, shape.as_ptr(), 1, one.as_ptr().cast(), 8, &mut status) }, TFE_INVALID_ARGUMENT);
    check("byte count disagrees with shape", unsafe { tfe_mpi_send(comm, other, 1, 5, shape.as_ptr(), 1, one.as_ptr().cast(), 4, &mut status) }, TFE_SHAPE_MISMATCH);
    let bad_bool = [2u8];
    check("invalid bool", unsafe { tfe_mpi_send(comm, other, 1, 1, shape.as_ptr(), 1, bad_bool.as_ptr().cast(), 1, &mut status) }, TFE_INVALID_ARGUMENT);
    check("unknown dtype", unsafe { tfe_mpi_send(comm, other, 1, 99, shape.as_ptr(), 1, one.as_ptr().cast(), 8, &mut status) }, TFE_INVALID_ARGUMENT);

    // Host traffic on the world communicator and tenferro traffic on the dup, together.
    let host_message: Vec<u8> = (0u8..64).collect();
    if rank == 0 {
        world.process_at_rank(1).send_with_tag(&host_message[..], 5);
        for (i, p) in payloads().iter().enumerate() {
            let data = if p.bytes.is_empty() { std::ptr::null() } else { p.bytes.as_ptr() };
            let shape_ptr = if p.shape.is_empty() { std::ptr::null() } else { p.shape.as_ptr() };
            check(&format!("send payload {i}"), unsafe { tfe_mpi_send(comm, 1, 20 + i as i32, p.code, shape_ptr, p.shape.len(), data.cast(), p.bytes.len(), &mut status) }, TFE_SUCCESS);
        }
    } else {
        let mut got = vec![0u8; 64];
        world.process_at_rank(0).receive_into_with_tag(&mut got[..], 5);
        assert_eq!(got, host_message, "host message on MPI_COMM_WORLD");
        for (i, p) in payloads().iter().enumerate() {
            let (mut code, mut rk, mut nbytes) = (0u8, 0usize, 0usize);
            let mut dims = [0usize; 8];
            check(&format!("recv payload {i}"), unsafe { tfe_mpi_recv(comm, 0, 20 + i as i32, &mut code, dims.as_mut_ptr(), dims.len(), &mut rk, &mut nbytes, &mut status) }, TFE_SUCCESS);
            assert_eq!((code, &dims[..rk], nbytes), (p.code, &p.shape[..], p.bytes.len()), "payload {i} header");
            let mut out = vec![0u8; nbytes];
            check(&format!("take payload {i}"), unsafe { tfe_mpi_recv_take(comm, out.as_mut_ptr().cast::<c_void>(), out.len(), &mut status) }, TFE_SUCCESS);
            assert_eq!(out, p.bytes, "payload {i} bytes");
        }
        // No tensor is held now.
        check("take with nothing held", unsafe { tfe_mpi_recv_take(comm, std::ptr::null_mut(), 0, &mut status) }, TFE_INVALID_ARGUMENT);
    }

    // A shape buffer that is too small loses nothing.
    if rank == 0 {
        let data: Vec<u8> = [1.0f64, 2.0, 3.0, 4.0].iter().flat_map(|x| x.to_ne_bytes()).collect();
        let shape = [1usize, 2, 2];
        check("send rank-3", unsafe { tfe_mpi_send(comm, 1, 40, 5, shape.as_ptr(), 3, data.as_ptr().cast(), 32, &mut status) }, TFE_SUCCESS);
    } else {
        let (mut code, mut rk, mut nbytes) = (0u8, 0usize, 0usize);
        let mut dims = [0usize; 2];
        check("recv with a small shape buffer", unsafe { tfe_mpi_recv(comm, 0, 40, &mut code, dims.as_mut_ptr(), 2, &mut rk, &mut nbytes, &mut status) }, TFE_BUFFER_TOO_SMALL);
        assert_eq!((rk, nbytes), (3, 32));
        let mut small = [0u8; 8];
        check("take into a small buffer", unsafe { tfe_mpi_recv_take(comm, small.as_mut_ptr().cast(), 8, &mut status) }, TFE_BUFFER_TOO_SMALL);
        let mut out = [0u8; 32];
        check("take", unsafe { tfe_mpi_recv_take(comm, out.as_mut_ptr().cast(), 32, &mut status) }, TFE_SUCCESS);
        assert_eq!(f64::from_ne_bytes(out[24..32].try_into().unwrap()), 4.0);
    }
    world.barrier();

    // D10: a panic stub reached through the entry is caught, reported, and poisons the object.
    let mut victim: *mut tfe_mpi_comm = std::ptr::null_mut();
    check("attach victim", unsafe { tfe_mpi_comm_attach(raw, &mut victim, &mut status) }, TFE_SUCCESS);
    // The default panic hook would print the expected panic to stderr.
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let hit = unsafe { tfe_mpi_test_hit_stub(victim, &mut status) };
    std::panic::set_hook(hook);
    check("stub hit", hit, TFE_INTERNAL_ERROR);
    assert!(last_error().contains("outside tenferro's native MPI surface"), "{}", last_error());
    check("poisoned", unsafe { tfe_mpi_comm_info(victim, std::ptr::null_mut(), 0, std::ptr::null_mut(), &mut status) }, TFE_INTERNAL_ERROR);
    assert!(last_error().contains("poisoned"), "{}", last_error());
    unsafe { tfe_mpi_comm_detach(victim, &mut status) };
    // The healthy object is unaffected, and the forced hit is the only one.
    check("healthy after the poisoned one", unsafe { tfe_mpi_comm_info(comm, std::ptr::null_mut(), 0, std::ptr::null_mut(), &mut status) }, TFE_SUCCESS);
    assert_eq!(ffi::stub_hits(), 1, "only the forced stub was reached");
    world.barrier();

    // Host finalizes first: calls report it, detach and info stay safe.
    drop(dup);
    drop(world);
    drop(universe);
    check("send after finalize", unsafe { tfe_mpi_send(comm, other, 1, 5, shape.as_ptr(), 1, one.as_ptr().cast(), 8, &mut status) }, TFE_MPI_FINALIZED);
    assert!(last_error().contains("host finalized MPI"), "{}", last_error());
    check("comm_info after finalize", unsafe { tfe_mpi_comm_info(comm, std::ptr::null_mut(), 0, std::ptr::null_mut(), &mut status) }, TFE_MPI_FINALIZED);
    check("detach after finalize", unsafe { tfe_mpi_comm_detach(comm, &mut status) }, TFE_SUCCESS);
    check("library_info after finalize", unsafe { tfe_mpi_library_info(std::ptr::null_mut(), 0, std::ptr::null_mut(), &mut status) }, TFE_SUCCESS);
    println!("rank {rank}: all scenarios passed");
}
