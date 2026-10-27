#![cfg(feature = "cuda")]

//! The backend's capability descriptor against what the engine actually does (issue #2208).
//!
//! The table is written by hand, so every entry is held to the engine's observed behaviour: a
//! `Native` entry must run, and every other entry must be refused. Seven dtypes times the
//! catalogue operations the CUDA backend claims to serve.

use tenferro_core_ops::PrimitiveOpKind;
use tenferro_gpu::cuda::{
    cuda_capabilities, gpu_available, upload_tensor, CudaDeviceId, CudaEngine,
};
use tenferro_tensor::{
    CapabilityQuery, Complex32, Complex64, DType, DotGeneralConfig, SupportLevel, Tensor,
    TensorAnalytic, TensorBackendCapability, TensorDot, TensorElementwise, TensorRead,
    TensorReduction,
};
use tinyvec::TinyVec;

const DTYPES: [DType; 7] = [
    DType::F32,
    DType::F64,
    DType::C32,
    DType::C64,
    DType::I32,
    DType::I64,
    DType::Bool,
];

const OPS: [PrimitiveOpKind; 7] = [
    PrimitiveOpKind::Add,
    PrimitiveOpKind::Mul,
    PrimitiveOpKind::Abs,
    PrimitiveOpKind::Exp,
    PrimitiveOpKind::Compare,
    PrimitiveOpKind::ReduceSum,
    PrimitiveOpKind::DotGeneral,
];

fn engine() -> Option<CudaEngine> {
    if !gpu_available() {
        return None;
    }
    Some(CudaEngine::open(CudaDeviceId::from_ordinal(0)).expect("CUDA engine"))
}

/// A two-element column vector of `dtype`, holding `[1, 2]` (or `[true, false]`).
fn host(dtype: DType) -> Tensor {
    match dtype {
        DType::F32 => Tensor::from_vec_col_major(vec![2], vec![1.0_f32, 2.0]).expect("host"),
        DType::F64 => Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).expect("host"),
        DType::C32 => Tensor::from_vec_col_major(
            vec![2],
            vec![Complex32::new(1.0, 1.0), Complex32::new(2.0, -1.0)],
        )
        .expect("host"),
        DType::C64 => Tensor::from_vec_col_major(
            vec![2],
            vec![Complex64::new(1.0, 1.0), Complex64::new(2.0, -1.0)],
        )
        .expect("host"),
        DType::I32 => Tensor::from_vec_col_major(vec![2], vec![1_i32, 2]).expect("host"),
        DType::I64 => Tensor::from_vec_col_major(vec![2], vec![1_i64, 2]).expect("host"),
        DType::Bool => Tensor::from_vec_col_major(vec![2], vec![true, false]).expect("host"),
        other => panic!("no fixture for {other:?}"),
    }
}

/// Run one catalogue operation on a two-element vector of `dtype`.
fn run(
    engine: &mut CudaEngine,
    op: PrimitiveOpKind,
    dtype: DType,
) -> tenferro_tensor::Result<Tensor> {
    let left = upload_tensor(engine.runtime(), &host(dtype)).expect("upload");
    let read = TensorRead::from_tensor(&left);
    match op {
        PrimitiveOpKind::Add => engine.add_read(read, TensorRead::from_tensor(&left)),
        PrimitiveOpKind::Mul => engine.mul_read(read, TensorRead::from_tensor(&left)),
        PrimitiveOpKind::Abs => engine.abs_read(read),
        PrimitiveOpKind::Exp => engine.exp_read(read),
        PrimitiveOpKind::Compare => engine.compare_read(
            read,
            TensorRead::from_tensor(&left),
            &tenferro_tensor::CompareDir::Eq,
        ),
        PrimitiveOpKind::ReduceSum => engine.reduce_sum_read(read, &[0]),
        PrimitiveOpKind::DotGeneral => engine.dot_general_read(
            read,
            TensorRead::from_tensor(&left),
            &DotGeneralConfig {
                lhs_contracting_dims: [0].into_iter().collect(),
                rhs_contracting_dims: [0].into_iter().collect(),
                lhs_batch_dims: TinyVec::new(),
                rhs_batch_dims: TinyVec::new(),
            },
        ),
        other => panic!("no call for {other:?}"),
    }
}

#[test]
fn every_claimed_entry_matches_what_the_engine_does() {
    let Some(mut engine) = engine() else { return };
    let mut checked = 0_usize;
    for dtype in DTYPES {
        for op in OPS {
            let Some(entry) = engine.capability(CapabilityQuery::new(op, dtype)) else {
                // Unlisted pairs must be refused: the descriptor is the authority, and a pair the
                // table does not mention is one this backend does not offer.
                let outcome = run(&mut engine, op, dtype);
                assert!(
                    outcome.is_err(),
                    "{op:?}/{dtype:?} has no descriptor entry but the engine served it"
                );
                continue;
            };
            let outcome = run(&mut engine, op, dtype);
            match entry.result {
                SupportLevel::Native => {
                    assert!(
                        outcome.is_ok(),
                        "{op:?}/{dtype:?} is declared Native but the engine failed: {:?}",
                        outcome.err()
                    );
                }
                _ => {
                    assert!(
                        outcome.is_err(),
                        "{op:?}/{dtype:?} is declared {:?} but the engine served it",
                        entry.result
                    );
                }
            }
            checked += 1;
        }
    }
    assert!(checked >= 7, "the descriptor covered only {checked} pairs");
}

#[test]
fn the_descriptor_names_this_backend_and_its_dtype_pairs() {
    // This one needs an engine, and a host without a device skips it like its siblings (the GPU
    // lane runs it with `TENFERRO_REQUIRE_GPU=1`, where a missing device fails).
    let Some(engine) = engine() else { return };
    assert_eq!(
        engine.backend_id(),
        tenferro_tensor::BackendId::Cuda,
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
