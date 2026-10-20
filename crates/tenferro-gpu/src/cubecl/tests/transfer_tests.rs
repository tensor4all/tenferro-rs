//! Host <-> device transfer correctness (#2009).
//!
//! Uploads stage a borrowed host buffer once (or hand an owned buffer to CubeCL
//! without a copy) and downloads copy straight into the host tensor's vector.
//! These tests pin the observable contract around those paths: exact bits for
//! every dtype and size class, allocations that do not start at offset zero,
//! the caller's buffer being free to change as soon as the upload returns, and
//! downloads observing earlier device writes without an explicit barrier.
use crate::cubecl::tests::EngineRun;

use num_complex::{Complex32, Complex64};
use tenferro_tensor::{ErrorKind, TensorRead, ValidationKind};

use super::super::interop::upload_typed_tensor;
use super::{download, gpu_backend, tensor_bool, tensor_c32, tensor_c64, tensor_f32, tensor_f64};
use super::{tensor_i32, tensor_i64, upload};
use crate::config::SliceConfig;
use crate::cuda::{download_tensor_into, CudaRuntime, DownloadIntoError};
use crate::{DType, Tensor};

/// Element counts around the small-download fast path (16 bytes), page and
/// pool boundaries, and one payload above CubeCL's 100 MB pinned staging limit.
const SIZES: [usize; 9] = [0, 1, 2, 3, 17, 4_097, 65_537, 1 << 20, (1 << 24) + 3];

fn assert_same_bits(actual: &Tensor, expected: &Tensor) {
    assert_eq!(actual.shape(), expected.shape());
    assert_eq!(actual.dtype(), expected.dtype());
    let len = expected.shape().iter().product::<usize>();
    macro_rules! bits {
        ($ty:ty, $to:expr) => {{
            let a = actual.as_slice::<$ty>().unwrap();
            let e = expected.as_slice::<$ty>().unwrap();
            assert_eq!(a.len(), len);
            for (index, (x, y)) in a.iter().zip(e).enumerate() {
                assert_eq!($to(x), $to(y), "element {index} of {len}");
            }
        }};
    }
    match expected.dtype() {
        DType::F64 => bits!(f64, |v: &f64| v.to_bits()),
        DType::F32 => bits!(f32, |v: &f32| v.to_bits()),
        DType::I64 => bits!(i64, |v: &i64| *v),
        DType::I32 => bits!(i32, |v: &i32| *v),
        DType::Bool => bits!(bool, |v: &bool| *v),
        DType::C64 => bits!(Complex64, |v: &Complex64| (v.re.to_bits(), v.im.to_bits())),
        DType::C32 => bits!(Complex32, |v: &Complex32| (v.re.to_bits(), v.im.to_bits())),
        other => panic!("unexpected dtype {other:?}"),
    }
}

fn payloads(len: usize) -> Vec<Tensor> {
    let shape = vec![len];
    // Signed zeros, NaN payloads and infinities must survive bit for bit.
    let special = [
        0.0,
        -0.0,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        1.0e-310,
    ];
    let f64s: Vec<f64> = (0..len)
        .map(|i| {
            special
                .get(i % 11)
                .copied()
                .unwrap_or(i as f64 * 0.5 - 3.25)
        })
        .collect();
    vec![
        tensor_f64(shape.clone(), f64s.clone()),
        tensor_f32(shape.clone(), f64s.iter().map(|&v| v as f32).collect()),
        tensor_i64(shape.clone(), (0..len).map(|i| i as i64 * -7 + 3).collect()),
        tensor_i32(shape.clone(), (0..len).map(|i| i as i32 * 5 - 11).collect()),
        tensor_bool(shape.clone(), (0..len).map(|i| i % 3 == 1).collect()),
        tensor_c64(
            shape.clone(),
            f64s.iter().map(|&v| Complex64::new(v, -v - 1.0)).collect(),
        ),
        tensor_c32(
            shape,
            f64s.iter()
                .map(|&v| Complex32::new(v as f32, 2.0 - v as f32))
                .collect(),
        ),
    ]
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn transfers_round_trip_every_dtype_and_size_class_bit_exactly() {
    let gpu = gpu_backend();
    for len in SIZES {
        // The largest size class is checked for one dtype only to bound the
        // test's memory and time.
        let all = payloads(len);
        let selected: Vec<&Tensor> = if len > (1 << 20) {
            all.iter().take(1).collect()
        } else {
            all.iter().collect()
        };
        for host in selected {
            let device = upload(&gpu, host);
            let back = download(&gpu, &device);
            assert_same_bits(&back, host);
        }
    }
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn owned_upload_and_offset_allocations_round_trip() {
    let gpu = gpu_backend();
    // `upload_typed_tensor` takes the vector by value and hands its allocation
    // to CubeCL without staging a copy.
    let data: Vec<f64> = (0..1_000).map(|i| i as f64 - 0.125).collect();
    let device = upload_typed_tensor(gpu.runtime(), vec![1_000], data.clone()).unwrap();
    let back = download(&gpu, &Tensor::from_typed(device));
    assert_eq!(back.as_slice::<f64>().unwrap(), data.as_slice());

    // Many small uploads share CubeCL pool pages, so later ones start at a
    // nonzero offset inside their page; downloads must read from that offset.
    let tensors: Vec<(Tensor, Tensor)> = (0..64)
        .map(|k| {
            let host = tensor_f64(
                vec![5 + k],
                (0..5 + k).map(|i| (k * 100 + i) as f64).collect(),
            );
            let device = upload(&gpu, &host);
            (host, device)
        })
        .collect();
    for (host, device) in &tensors {
        assert_same_bits(&download(&gpu, device), host);
    }

    // A device-produced allocation (a slice result) downloads exactly.
    let host = tensor_f64(vec![40, 30], (0..1_200).map(|i| i as f64).collect());
    let device = upload(&gpu, &host);
    let config = SliceConfig {
        starts: vec![3, 4],
        limits: vec![33, 24],
        strides: vec![2, 1],
    };
    let mut gpu = gpu;
    let sliced = gpu
        .run(|session| session.slice(&device, &config))
        .unwrap()
        .unwrap();
    let expected = tenferro_cpu::CpuEngine::serial()
        .run(|session| session.slice(&host, &config))
        .unwrap()
        .unwrap();
    assert_same_bits(&download(&gpu, &sliced), &expected);
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn caller_may_overwrite_the_source_as_soon_as_upload_returns() {
    let gpu = gpu_backend();
    for len in [3usize, 4_097, 1 << 22] {
        let original: Vec<f64> = (0..len).map(|i| i as f64 + 0.5).collect();
        let mut host = tensor_f64(vec![len], original.clone());
        let device = upload(&gpu, &host);
        // No barrier between the upload and the overwrite: the device write may
        // still be queued, so it must read CubeCL's staged copy, not `host`.
        host.as_typed_mut::<f64>()
            .unwrap()
            .host_data_mut()
            .unwrap()
            .fill(-1.0);
        let back = download(&gpu, &device);
        assert_eq!(
            back.as_slice::<f64>().unwrap(),
            original.as_slice(),
            "len {len}"
        );
    }
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn download_observes_queued_device_writes_on_this_and_other_streams() {
    let len = 1 << 20;
    let host = tensor_f64(vec![len], (0..len).map(|i| i as f64).collect());
    let expected: Vec<f64> = (0..len).map(|i| 2.0 * i as f64).collect();
    let mut gpu = gpu_backend();
    let device = upload(&gpu, &host);

    // Same thread (same CubeCL stream): the add is only queued when the
    // download starts.
    let doubled = gpu
        .run(|session| {
            session.add_read(
                TensorRead::from_tensor(&device),
                TensorRead::from_tensor(&device),
            )
        })
        .unwrap()
        .unwrap();
    assert_eq!(
        download(&gpu, &doubled).as_slice::<f64>().unwrap(),
        expected.as_slice()
    );

    // Another thread drives the same engine (and so its stream): produce there,
    // download here without any explicit synchronization in between.
    let worker = &mut gpu;
    let produced = std::thread::scope(|scope| {
        scope
            .spawn(|| {
                worker
                    .run(|session| {
                        session.add_read(
                            TensorRead::from_tensor(&device),
                            TensorRead::from_tensor(&device),
                        )
                    })
                    .unwrap()
                    .unwrap()
            })
            .join()
            .unwrap()
    });
    assert_eq!(
        download(&gpu, &produced).as_slice::<f64>().unwrap(),
        expected.as_slice()
    );
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn download_tensor_into_round_trips_and_reuses_one_allocation() {
    let gpu = gpu_backend();
    let runtime = gpu.runtime();

    // f64 round trip and pointer stability across repeated readbacks.
    let host_f64 = tensor_f64(
        vec![4096],
        (0..4096).map(|i| i as f64 * 0.5 - 1.0).collect(),
    );
    let device_f64 = upload(&gpu, &host_f64);
    let filled =
        download_tensor_into(runtime, &device_f64, Vec::<f64>::with_capacity(4096)).unwrap();
    assert_eq!(filled.as_slice(), host_f64.as_slice::<f64>().unwrap());
    let pointer = filled.as_ptr();
    let mut filled = filled;
    for _ in 0..2 {
        filled = download_tensor_into(runtime, &device_f64, filled).unwrap();
        assert_eq!(filled.as_slice(), host_f64.as_slice::<f64>().unwrap());
        assert_eq!(
            filled.as_ptr(),
            pointer,
            "a sufficient capacity must not be reallocated"
        );
    }

    // Small uploads share CubeCL pool pages, so later handles start at a nonzero offset inside
    // their page; the readback must honor that offset.
    let mut offset_buffer = Vec::<f64>::new();
    for k in 0..64usize {
        let host = tensor_f64(
            vec![5 + k],
            (0..5 + k).map(|i| (k * 100 + i) as f64).collect(),
        );
        let device = upload(&gpu, &host);
        offset_buffer = download_tensor_into(runtime, &device, offset_buffer).unwrap();
        assert_eq!(
            offset_buffer.as_slice(),
            host.as_slice::<f64>().unwrap(),
            "offset tensor {k}"
        );
    }

    // f32 and c64 through a fresh buffer.
    let host_f32 = tensor_f32(vec![17], (0..17).map(|i| i as f32 + 0.25).collect());
    let device_f32 = upload(&gpu, &host_f32);
    let filled_f32 = download_tensor_into(runtime, &device_f32, Vec::<f32>::new()).unwrap();
    assert_eq!(filled_f32, host_f32.as_slice::<f32>().unwrap());

    let data_c64: Vec<Complex64> = (0..32)
        .map(|i| Complex64::new(i as f64, -(i as f64)))
        .collect();
    let host_c64 = tensor_c64(vec![32], data_c64.clone());
    let device_c64 = upload(&gpu, &host_c64);
    let filled_c64 = download_tensor_into(runtime, &device_c64, Vec::<Complex64>::new()).unwrap();
    assert_eq!(filled_c64, data_c64);
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn download_tensor_into_grows_a_short_buffer_and_handles_empty() {
    let gpu = gpu_backend();
    let runtime = gpu.runtime();

    let small = tensor_f64(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
    let small_device = upload(&gpu, &small);
    let buffer = download_tensor_into(runtime, &small_device, Vec::<f64>::new()).unwrap();
    assert_eq!(buffer, [1.0, 2.0, 3.0, 4.0]);

    // A later, larger readback must grow the reused allocation and land correctly.
    let large = tensor_f64(vec![64], (0..64).map(|i| i as f64).collect());
    let large_device = upload(&gpu, &large);
    let buffer = download_tensor_into(runtime, &large_device, buffer).unwrap();
    assert_eq!(buffer.as_slice(), large.as_slice::<f64>().unwrap());
    assert!(buffer.capacity() >= 64);

    // An empty tensor clears the buffer and hands it back.
    let empty = tensor_f64(vec![0], Vec::new());
    let empty_device = upload(&gpu, &empty);
    let buffer = download_tensor_into(runtime, &empty_device, buffer).unwrap();
    assert!(buffer.is_empty());
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn download_tensor_into_rejects_dtype_mismatch_and_returns_the_buffer() {
    let gpu = gpu_backend();
    let runtime = gpu.runtime();
    let host = tensor_f64(vec![8], (0..8).map(|i| i as f64).collect());
    let device = upload(&gpu, &host);

    let buffer = Vec::<f32>::with_capacity(8);
    let error = download_tensor_into(runtime, &device, buffer).unwrap_err();
    match error {
        DownloadIntoError::Rejected { error, buffer } => {
            assert_eq!(
                error.kind(),
                ErrorKind::Validation(ValidationKind::DTypeMismatch)
            );
            assert!(buffer.is_empty());
            assert_eq!(buffer.capacity(), 8, "the submitted buffer is handed back");
        }
        other => panic!("expected a pre-enqueue rejection, got {other:?}"),
    }
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn download_tensor_into_rejects_bool_destination() {
    let gpu = gpu_backend();
    let runtime = gpu.runtime();
    let host = tensor_bool(vec![4], vec![true, false, true, false]);
    let device = upload(&gpu, &host);

    let error = download_tensor_into::<bool>(runtime, &device, Vec::new()).unwrap_err();
    match error {
        DownloadIntoError::Rejected { error, buffer } => {
            assert_eq!(error.kind(), ErrorKind::Unsupported);
            assert!(buffer.is_empty());
        }
        other => panic!("expected a pre-enqueue rejection, got {other:?}"),
    }
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn download_tensor_into_rejects_host_and_foreign_runtime_tensors() {
    let gpu = gpu_backend();
    let runtime = gpu.runtime();
    let host = tensor_f64(vec![4], vec![1.0, 2.0, 3.0, 4.0]);

    // A host tensor has no device allocation to read.
    let error = download_tensor_into(runtime, &host, Vec::<f64>::new()).unwrap_err();
    assert!(matches!(
        error,
        DownloadIntoError::Rejected { error, .. } if error.kind() == ErrorKind::RuntimeState
    ));

    // A same-ordinal but distinct runtime owns another allocation domain.
    let device = upload(&gpu, &host);
    let foreign = CudaRuntime::new(crate::cuda::CudaDeviceId::from_ordinal(0)).unwrap();
    let error = download_tensor_into(&foreign, &device, Vec::<f64>::new()).unwrap_err();
    assert!(matches!(
        error,
        DownloadIntoError::Rejected { error, .. } if error.kind() == ErrorKind::RuntimeState
    ));
}
