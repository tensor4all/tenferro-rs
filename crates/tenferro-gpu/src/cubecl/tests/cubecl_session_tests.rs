//! CubeCL-session contract tests (issue #1597).
//!
//! These tests require CUDA hardware and are therefore ignored by default,
//! matching the regular CUDA test convention in this crate.

use crate::cuda::{gpu_available, CudaBackend};

use super::*;

fn first_cuda_backend() -> Option<CudaBackend> {
    let devices = crate::cuda::cuda_devices().ok()?;
    let device = devices.first()?;
    CudaBackend::new(device.id()).ok()
}

#[test]
#[ignore = "requires CUDA"]
fn cubecl_session_exposes_client_and_launch_helpers() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let mut backend = first_cuda_backend().expect("CUDA backend should initialize");
    with_cuda_exec(&mut backend, |session| {
        session
            .with_cubecl("test.cubecl_helpers", |cubecl| {
                // A small one-dimensional problem gives the standard cube dim
                // and a bounded cube count.
                let count = cubecl.cube_count_1d(256)?;
                let cubecl::prelude::CubeCount::Static(x, _y, _z) = count else {
                    panic!("expected static cube count")
                };
                assert!(x >= 1);
                let _dim = cubecl.cube_dim_1d();
                Ok(())
            })
            .unwrap();
    });
}

#[test]
#[ignore = "requires CUDA"]
fn cubecl_session_allocates_and_binds_output() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let mut backend = first_cuda_backend().expect("CUDA backend should initialize");
    with_cuda_exec(&mut backend, |session| {
        session
            .with_cubecl("test.cubecl_alloc", |cubecl| {
                let output = cubecl.alloc_output::<f32>(&[8])?;
                // Binding a freshly allocated tensor must be valid.
                // SAFETY: every binding here is created for a tensor this call owns, and the
                // kernel writes only into those buffers.
                let _binding = unsafe { cubecl.tensor_binding(&output, "test.cubecl_alloc") }?;
                Ok(())
            })
            .unwrap();
    });
}

#[test]
#[ignore = "requires CUDA"]
fn cubecl_session_allocates_zero_filled_output() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let mut backend = first_cuda_backend().expect("CUDA backend should initialize");
    with_cuda_exec(&mut backend, |session| {
        let output = session
            .with_cubecl("test.cubecl_alloc_zero", |cubecl| {
                cubecl.alloc_zero_output::<f32>(&[16])
            })
            .unwrap();
        // The fill kernel must produce semantic zeros on the device.
        let result = session
            .with_raw("test.cubecl_alloc_zero_raw", |raw| {
                raw.download_tensor::<f32>(&output, "test.cubecl_alloc_zero_raw")
            })
            .unwrap();
        let values = result.host_data().unwrap();
        assert_eq!(values.len(), 16);
        assert!(values.iter().all(|&v| v == 0.0));
    });
}

#[test]
#[ignore = "requires CUDA"]
fn cubecl_session_fills_an_existing_output_with_zero() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let mut backend = first_cuda_backend().expect("CUDA backend should initialize");
    with_cuda_exec(&mut backend, |session| {
        // Seed a destination the session owns with a value that `0 * y` would
        // preserve, then reset it through the public session entry.
        let output = session
            .with_raw("test.cubecl_fill_zero_raw", |raw| {
                let mut output = raw.alloc_output::<f32>(&[4])?;
                let seed = [f32::NAN, -0.0, f32::INFINITY, 2.5];
                let seed_bytes = unsafe {
                    std::slice::from_raw_parts(seed.as_ptr().cast::<u8>(), seed.len() * 4)
                };
                let uploaded = raw.upload_bytes(seed_bytes, "test.cubecl_fill_zero_seed")?;
                let dst = raw.tensor_mut(&mut output)?;
                let dst_ptr = unsafe { dst.raw_ptr() };
                let mut copy_result = Ok(());
                // SAFETY: `uploaded` is an uploaded workspace of the same byte
                // size as `dst`, both on this runtime's stream; `dst` uniquely
                // owns the destination span.
                unsafe {
                    uploaded.with_ptr(|src_ptr| {
                        copy_result = raw.copy_bytes(
                            dst_ptr,
                            src_ptr,
                            seed_bytes.len(),
                            "test.cubecl_fill_zero_copy",
                        );
                    });
                }
                copy_result?;
                Ok(output)
            })
            .unwrap();
        let mut output_enum = tenferro_tensor::Tensor::from_typed::<f32>(output);
        session
            .with_cubecl("test.cubecl_fill_zero", |cubecl| {
                cubecl.fill_zero_write(tenferro_tensor::TensorWrite::from_tensor(&mut output_enum))
            })
            .unwrap();
        let typed = output_enum
            .into_typed::<f32>()
            .expect("the fill keeps the destination dtype");
        let result = session
            .with_raw("test.cubecl_fill_zero_raw2", |raw| {
                raw.download_tensor::<f32>(&typed, "test.cubecl_fill_zero_raw2")
            })
            .unwrap();
        let values = result.host_data().unwrap();
        assert_eq!(values.len(), 4);
        for value in values {
            assert_eq!(value.to_bits(), 0.0_f32.to_bits(), "expected +0.0 bits");
        }
    });
}

#[test]
#[ignore = "requires CUDA"]
fn cubecl_session_scales_output_in_place() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let mut backend = first_cuda_backend().expect("CUDA backend should initialize");
    with_cuda_exec(&mut backend, |session| {
        // Allocate and seed [1.0, 2.0, 3.0, 4.0] on device, then scale by 3
        // and read back, returning the typed tensor from the raw entrance.
        let output = session
            .with_raw("test.cubecl_scale_raw", |raw| {
                let mut output = raw.alloc_output::<f32>(&[4])?;
                let seed = [1.0f32, 2.0, 3.0, 4.0];
                let seed_bytes = unsafe {
                    std::slice::from_raw_parts(seed.as_ptr().cast::<u8>(), seed.len() * 4)
                };
                let uploaded = raw.upload_bytes(seed_bytes, "test.cubecl_scale_seed")?;
                let dst = raw.tensor_mut(&mut output)?;
                let dst_ptr = unsafe { dst.raw_ptr() };
                let mut copy_result = Ok(());
                // SAFETY: `uploaded` is an uploaded workspace of the same
                // byte size as `dst`, both on this runtime's stream; `dst`
                // uniquely owns the destination span.
                unsafe {
                    uploaded.with_ptr(|src_ptr| {
                        copy_result = raw.copy_bytes(
                            dst_ptr,
                            src_ptr,
                            seed_bytes.len(),
                            "test.cubecl_scale_copy",
                        );
                    });
                }
                copy_result?;
                Ok(output)
            })
            .unwrap();
        let mut output_enum = tenferro_tensor::Tensor::from_typed::<f32>(output);
        session
            .with_cubecl("test.cubecl_scale", |cubecl| {
                cubecl.scale_tensor_write(
                    tenferro_tensor::TensorWrite::from_tensor(&mut output_enum),
                    3.0,
                )
            })
            .unwrap();
        let typed = output_enum
            .into_typed::<f32>()
            .expect("the scale output keeps its dtype");
        let bytes = session
            .with_raw("test.cubecl_scale_raw2", |raw| {
                raw.download_tensor::<f32>(&typed, "test.cubecl_scale_raw2")
            })
            .unwrap();
        let values = bytes.host_data().unwrap();
        assert_eq!(values[0], 3.0);
        assert_eq!(values[1], 6.0);
        assert_eq!(values[2], 9.0);
        assert_eq!(values[3], 12.0);
    });
}

#[test]
#[ignore = "requires CUDA"]
fn cubecl_session_flushes_on_exit_so_raw_sees_work() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let mut backend = first_cuda_backend().expect("CUDA backend should initialize");
    with_cuda_exec(&mut backend, |session| {
        // Enqueue a trivial cubecl interaction, then immediately enter the raw
        // session. The cross-session flush must make the captured stream usable.
        session
            .with_cubecl("test.cubecl_then_raw", |cubecl| {
                let _ = cubecl.cube_count_1d(16)?;
                Ok(())
            })
            .unwrap();
        session
            .with_raw("test.cubecl_then_raw", |raw| {
                let _stream = raw.stream();
                Ok(())
            })
            .unwrap();
    });
}

#[test]
#[ignore = "requires CUDA"]
fn cubecl_session_flushes_after_error_callback() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let mut backend = first_cuda_backend().expect("CUDA backend should initialize");
    with_cuda_exec(&mut backend, |session| {
        let result: tenferro_tensor::Result<()> =
            session.with_cubecl("test.cubecl_error", |cubecl| {
                let _ = cubecl.cube_count_1d(4)?;
                Err(tenferro_tensor::Error::runtime_state(
                    "test.cubecl_error",
                    "intentional failure",
                ))
            });
        assert!(result.is_err());
        // A subsequent session still works after the error path flushed.
        session
            .with_cubecl("test.cubecl_after_error", |cubecl| {
                let _ = cubecl.cube_count_1d(8)?;
                Ok(())
            })
            .unwrap();
    });
}

#[test]
#[ignore = "requires CUDA"]
fn cuda_with_backend_session_rejects_nested_entry() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let mut backend = first_cuda_backend().expect("CUDA backend should initialize");
    let nested = backend
        .with_backend_session(|_session| {
            // The backend override wraps its closure in the portable in-session
            // guard, so re-entering any session-entry point on this thread
            // (here the shared helper directly) is rejected before its callback
            // runs (issue #1680 Phase 3).
            tenferro_tensor::with_session_entry_guard("nested", || ())
        })
        .unwrap();
    assert!(
        matches!(
            nested,
            Err(tenferro_tensor::SessionEntryError::Reentered { backend: "nested" })
        ),
        "nested session entry must be rejected with a typed error: {nested:?}"
    );
}

#[cfg(debug_assertions)]
#[test]
#[ignore = "requires CUDA"]
fn cuda_with_backend_session_restores_the_in_session_flag_after_panic() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let mut backend = first_cuda_backend().expect("CUDA backend should initialize");
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        backend
            .with_backend_session(|_session| panic!("boom"))
            .unwrap()
    }));
    assert!(outcome.is_err());
    // The flag is usable again on the same thread.
    let value = backend.with_backend_session(|_session| 7usize).unwrap();
    assert_eq!(value, 7);
}

/// An allocation the device refuses must reach the caller as a typed error on
/// the buffer: never a panic on CubeCL's device thread, and never an `Ok` that
/// hands back bytes nothing wrote (#2081).
///
/// On the 0.11 pins tenferro's contract is CubeCL's. `Client::do_empty` queues
/// the reservation and `cubecl-cuda/src/compute/server.rs` records a failure on
/// the buffer (`fail_unallocated`), so `alloc_output` returns `Ok` meaning "the
/// request was accepted", and the failure is reported at that buffer's next use:
/// `check`, a read, or `download`. The older behaviour - the device thread
/// panicking while the caller is told the allocation succeeded - is what this
/// test forbids, and it is also why a per-allocation `check` is not the fix: it
/// costs a device-thread round trip (measured at 8.19 us on the A100, against a
/// 1-2 us per-operation budget in #2079).
///
/// A 200 GiB request is refused by the device itself, so nothing is allocated.
#[test]
#[ignore = "requires CUDA"]
fn cubecl_refused_allocation_is_reported_on_the_buffer() {
    use cubecl_runtime::server::Handle;
    use tenferro_tensor::{backend::BackendSessionHost, TensorRead};

    assert!(gpu_available(), "CUDA test requires an available device");
    let mut backend = first_cuda_backend().expect("CUDA backend should initialize");
    let bytes = 200usize << 30;

    // The raw boundary: the refusal is readable, and it does not poison the device.
    with_cuda_exec(&mut backend, |session| {
        session
            .with_cubecl("test.refused_alloc", |cubecl| {
                let refused: Handle = cubecl.client().empty(bytes);
                assert!(
                    cubecl.client().check([&refused]).is_err(),
                    "a refused allocation must be reported on its buffer"
                );

                let later: Handle = cubecl.client().empty(4096);
                assert!(
                    cubecl.client().check([&later]).is_ok(),
                    "a later allocation must still succeed after a refused one"
                );
                Ok(())
            })
            .unwrap();
    });

    // The tenferro boundary: `alloc_output` reports acceptance, and reading that
    // output reports the failure instead of returning unwritten bytes.
    with_cuda_exec(&mut backend, |session| {
        let output = session
            .with_cubecl("test.refused_alloc_output", |cubecl| {
                cubecl.alloc_output::<f32>(&[bytes / 4])
            })
            .expect("the request is accepted; the failure belongs to the buffer");
        let read = session.with_raw("test.refused_alloc_read", |raw| {
            raw.download_tensor::<f32>(&output, "test.refused_alloc_read")
        });
        assert!(
            read.is_err(),
            "reading an output whose allocation was refused must fail"
        );
    });

    // The session still works afterwards.
    let input = tensor_f32(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
    let uploaded = upload(&backend, &input);
    let difference = backend
        .with_backend_session(|session| {
            session.sub_read(
                TensorRead::from_tensor(&uploaded),
                TensorRead::from_tensor(&uploaded),
            )
        })
        .unwrap()
        .unwrap();
    let values = download(&backend, &difference);
    assert_eq!(values.as_slice::<f32>().unwrap(), &[0.0, 0.0, 0.0, 0.0]);
}

/// #1887: a `with_cubecl` callback must not pay a blocking entry flush.
///
/// The compact zero fill drains the CubeCL host queue itself (one
/// `resolve_and_flush` hand-off) before its raw memset, so the wrapper's entry
/// `client.flush()` is a second device-thread round trip for no ordering gain.
/// This reads the counter at the top of the callback, before the fill runs, so
/// it fails if the entry flush comes back independently of machine timing, and
/// prints the host cost of an empty session, `alloc_output`, and
/// `alloc_zero_output` for the record.
#[test]
#[ignore = "requires CUDA"]
fn alloc_zero_output_avoids_the_with_cubecl_entry_flush() {
    use crate::cubecl::runtime::{blocking_flush_handoffs, reset_blocking_flush_handoffs};
    use std::time::Instant;

    assert!(gpu_available(), "CUDA test requires an available device");
    let mut backend = first_cuda_backend().expect("CUDA backend should initialize");
    with_cuda_exec(&mut backend, |session| {
        // Warm the raw-stream memo and the output-handle pool.
        for _ in 0..3 {
            let _ = session
                .with_cubecl("test.alloc_zero_warm", |c| {
                    c.alloc_zero_output::<f32>(&[16])
                })
                .unwrap();
        }

        // An entry flush, if the wrapper has one, runs before the callback body.
        // Read the counter at the top of the callback, before the fill runs, so a
        // cross-stream synchronize that `alloc_zero_output` takes inside the fill
        // cannot be mistaken for a wrapper entry flush.
        reset_blocking_flush_handoffs();
        let entry_flush_before_callback = std::cell::Cell::new(usize::MAX);
        let _ = session
            .with_cubecl("test.alloc_zero", |c| {
                entry_flush_before_callback.set(blocking_flush_handoffs());
                c.alloc_zero_output::<f32>(&[16])
            })
            .unwrap();

        let iters = 2000usize;
        let per_iter = |t: Instant| t.elapsed().as_secs_f64() * 1e6 / iters as f64;

        let started = Instant::now();
        for _ in 0..iters {
            session
                .with_cubecl("test.empty_session", |_c| -> tenferro_tensor::Result<()> {
                    Ok(())
                })
                .unwrap();
        }
        let empty_us = per_iter(started);

        let started = Instant::now();
        for _ in 0..iters {
            let _ = session
                .with_cubecl("test.alloc_only", |c| c.alloc_output::<f32>(&[16]))
                .unwrap();
        }
        let alloc_us = per_iter(started);

        let mut samples = Vec::new();
        for _ in 0..5 {
            let started = Instant::now();
            for _ in 0..iters {
                let _ = session
                    .with_cubecl("test.alloc_zero", |c| c.alloc_zero_output::<f32>(&[16]))
                    .unwrap();
            }
            samples.push(per_iter(started));
        }
        let best = samples.iter().cloned().fold(f64::INFINITY, f64::min);
        eprintln!(
            "[1887] empty {empty_us:.2} us/op; alloc {alloc_us:.2} us/op; zero {samples:?} best {best:.2} us/op (entry flushes before callback: {})",
            entry_flush_before_callback.get()
        );

        // Assert last so the host-cost reading is printed before a regression
        // failure surfaces the count.
        assert_eq!(
            entry_flush_before_callback.get(),
            0,
            "alloc_zero_output already drains the CubeCL queue; the with_cubecl \
             wrapper must not add a blocking entry flush (#1887)"
        );
    });
}
