#![cfg(feature = "cuda")]

//! The backend's capability descriptor against what the engine actually does (issues #2208,
//! #2477).
//!
//! The table is written by hand, so every entry is held to the engine's observed behaviour, and
//! every served pair is held to the CPU backend's numbers on the same fixture. The cases come
//! from `tenferro-cpu`'s shared dispatcher so both lanes run the identical fixture and the
//! identical rule; the operation list comes from the tables, so a new entry is covered without
//! editing this file.

use tenferro_core_ops::PrimitiveOpKind;
use tenferro_gpu::cuda::{
    cuda_capabilities, download_tensor, gpu_available, upload_tensor, CudaDeviceId, CudaEngine,
};
use tenferro_tensor::{
    BackendId, CapabilityQuery, DType, SupportLevel, Tensor, TensorBackendCapability,
};

#[path = "../../tenferro-cpu/tests/support/capability_cases.rs"]
mod capability_cases;

use capability_cases::{assert_entry_matches, catalog_admits, run_case, DTYPES};

fn engine() -> Option<CudaEngine> {
    if !gpu_available() {
        return None;
    }
    Some(CudaEngine::open(CudaDeviceId::from_ordinal(0)).expect("CUDA engine"))
}

/// Both tables' operations in table order, each once.
fn table_ops() -> Vec<PrimitiveOpKind> {
    let mut ops: Vec<PrimitiveOpKind> = Vec::new();
    for entry in cuda_capabilities()
        .iter()
        .chain(tenferro_cpu::cpu_capabilities())
    {
        if !ops.contains(&entry.op) {
            ops.push(entry.op);
        }
    }
    ops
}

#[test]
#[ignore = "requires CUDA"]
fn every_claimed_entry_matches_what_the_engine_does() {
    let Some(mut engine) = engine() else { return };
    let mut cpu = tenferro_cpu::CpuEngine::serial();
    let mut upload = |engine: &mut CudaEngine, tensor: &Tensor| {
        upload_tensor(engine.runtime(), tensor).expect("upload")
    };
    let mut duplicate = |_engine: &mut tenferro_cpu::CpuEngine, tensor: &Tensor| {
        tensor.duplicate().expect("host copy")
    };
    let ops = table_ops();
    let mut examined = 0_usize;
    let mut compared = 0_usize;

    for op in &ops {
        for dtype in DTYPES {
            examined += 1;
            let entry = engine.capability(CapabilityQuery::new(*op, dtype));
            if !catalog_admits(*op, dtype) {
                assert!(
                    entry.is_none(),
                    "CUDA {op:?}/{dtype:?} is outside the catalog's dtype policy but the \
                     descriptor names it"
                );
                continue;
            }
            let outcome = run_case(&mut engine, *op, dtype, &mut upload);
            if !entry.is_some_and(|entry| entry.result.is_supported()) {
                assert_entry_matches(BackendId::Cuda, *op, dtype, entry, outcome);
                continue;
            }
            let device = match outcome {
                Ok(device) => device,
                Err(error) => {
                    assert_entry_matches(BackendId::Cuda, *op, dtype, entry, Err(error));
                    continue;
                }
            };
            // The result dtype is the entry's, not the input's: `abs` on complex returns real
            // and `compare` returns `Bool`.
            let declared = entry.expect("a served pair has a descriptor entry");
            assert_eq!(
                device.dtype(),
                declared.output_dtype,
                "CUDA {} /{dtype:?} is declared with output dtype {:?} but returned {:?}",
                tenferro_core_ops::descriptor(*op).name,
                declared.output_dtype,
                device.dtype()
            );
            let reference =
                run_case(&mut cpu, *op, dtype, &mut duplicate).unwrap_or_else(|error| {
                    panic!(
                        "CUDA declares {} native but the CPU reference run failed: {error}",
                        tenferro_core_ops::descriptor(*op).name
                    )
                });
            assert_same_values(&engine, *op, device.dtype(), &reference, &device);
            compared += 1;
        }
    }

    // Seven dtypes times the catalogue operations the tables name: a run that visited fewer
    // has stopped enumerating the table rather than found it consistent.
    assert_eq!(
        examined,
        DTYPES.len() * ops.len(),
        "the run visited {examined} operation/dtype pairs"
    );
    assert_eq!(
        compared,
        served_pairs(),
        "every served pair needs a parity check"
    );
}

/// Number of table entries whose result axis declares support.
fn served_pairs() -> usize {
    cuda_capabilities()
        .iter()
        .filter(|entry| entry.result.is_supported())
        .count()
}

/// How many ulps of the dtype a kernel may differ by.
///
/// The tolerance is relative, scaled by `max(|expected|, 1.0)`, so near zero it is that many
/// ulps of 1.0 rather than of the value. Arithmetic (add/sub/mul/div/rem, reductions,
/// `dot_general`) gets 8 ulps. The transcendental kernels are different implementations on each
/// side (host `libm`/SIMD math versus the CUDA math library), so last-bit agreement is not part
/// of the contract and they get 64.
fn tolerance_ulps(op: PrimitiveOpKind) -> f64 {
    match op {
        PrimitiveOpKind::Erf
        | PrimitiveOpKind::Exp
        | PrimitiveOpKind::Expm1
        | PrimitiveOpKind::Log
        | PrimitiveOpKind::Log1p
        | PrimitiveOpKind::Pow
        | PrimitiveOpKind::Rsqrt
        | PrimitiveOpKind::Sin
        | PrimitiveOpKind::Cos
        | PrimitiveOpKind::Tanh
        | PrimitiveOpKind::Sqrt => 64.0,
        _ => 8.0,
    }
}

/// `(real, imaginary)` components of a host-resident tensor. Integers and `Bool` land on exact
/// small values here (the fixtures hold `1` and `2`), so the same comparison serves them
/// exactly; `i64 as f64` loses precision only above 2^53, which no fixture reaches.
fn components(dtype: DType, tensor: &Tensor) -> Vec<(f64, f64)> {
    match dtype {
        DType::F32 => tensor
            .as_slice::<f32>()
            .expect("f32 payload")
            .iter()
            .map(|value| (f64::from(*value), 0.0))
            .collect(),
        DType::F64 => tensor
            .as_slice::<f64>()
            .expect("f64 payload")
            .iter()
            .map(|value| (*value, 0.0))
            .collect(),
        DType::C32 => tensor
            .as_slice::<tenferro_tensor::Complex32>()
            .expect("c32 payload")
            .iter()
            .map(|value| (f64::from(value.re), f64::from(value.im)))
            .collect(),
        DType::C64 => tensor
            .as_slice::<tenferro_tensor::Complex64>()
            .expect("c64 payload")
            .iter()
            .map(|value| (value.re, value.im))
            .collect(),
        DType::I32 => tensor
            .as_slice::<i32>()
            .expect("i32 payload")
            .iter()
            .map(|value| (f64::from(*value), 0.0))
            .collect(),
        DType::I64 => tensor
            .as_slice::<i64>()
            .expect("i64 payload")
            .iter()
            .map(|value| (*value as f64, 0.0))
            .collect(),
        DType::Bool => tensor
            .as_slice::<bool>()
            .expect("bool payload")
            .iter()
            .map(|value| (f64::from(u8::from(*value)), 0.0))
            .collect(),
        other => panic!("no expected values for {other:?}"),
    }
}

fn assert_same_values(
    engine: &CudaEngine,
    op: PrimitiveOpKind,
    dtype: DType,
    reference: &Tensor,
    device: &Tensor,
) {
    let name = tenferro_core_ops::descriptor(op).name;
    let host = download_tensor(engine.runtime(), device).expect("download");
    assert_eq!(
        reference.shape(),
        host.shape(),
        "{name}/{dtype:?} has a different shape on CUDA"
    );
    let expected = components(dtype, reference);
    let actual = components(dtype, &host);
    let tolerance = match dtype {
        DType::F32 | DType::C32 => tolerance_ulps(op) * f64::from(f32::EPSILON),
        _ => tolerance_ulps(op) * f64::EPSILON,
    };
    for (index, (expected, actual)) in expected.iter().zip(actual.iter()).enumerate() {
        for (part, (expected, actual)) in [
            ("re", (expected.0, actual.0)),
            ("im", (expected.1, actual.1)),
        ]
        .into_iter()
        {
            let scale = expected.abs().max(1.0);
            assert!(
                (actual - expected).abs() <= tolerance * scale,
                "{name}/{dtype:?} element {index} {part}: CUDA {actual} differs from CPU \
                 {expected} by more than {tolerance} relative"
            );
        }
    }
}

#[test]
#[ignore = "requires CUDA"]
fn the_descriptor_names_this_backend_and_its_dtype_pairs() {
    // This one needs an engine, and a host without a device skips it like its siblings (the GPU
    // lane runs it with `TENFERRO_REQUIRE_GPU=1`, where a missing device fails).
    let Some(engine) = engine() else { return };
    assert_eq!(
        engine.backend_id(),
        BackendId::Cuda,
        "the descriptor claims another backend"
    );
    // Seven dtypes times the served operations: the table must at least name every dtype the
    // engine accepts, and must not silently drop one.
    for dtype in DTYPES {
        assert!(
            engine
                .capabilities()
                .iter()
                .any(|entry| entry.dtype == dtype),
            "{dtype:?} has no descriptor entry at all"
        );
    }
    // The table is a hand-written constant, not a view of the kernel catalogue.
    assert!(cuda_capabilities().len() > DTYPES.len());
    assert!(cuda_capabilities()
        .iter()
        .any(|entry| entry.result == SupportLevel::Unsupported));
}
