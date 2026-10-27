#![cfg(feature = "cuda")]

//! U3: a fused region is emitted, compiled once per runtime, bound and launched, and a
//! warm run performs no NVRTC compilation (issue #2208 U3).

use t4a_cuda_kernels::fusion::{FusionInstruction, FusionOp, FusionOperand, FusionPlan};
use t4a_cuda_kernels::DType;
use tenferro_gpu::cuda::{gpu_available, CudaDeviceId, CudaEngine, PreparedFusion};

fn runtime() -> Option<tenferro_gpu::cuda::CudaRuntime> {
    if !gpu_available() {
        return None;
    }
    let engine = CudaEngine::open(CudaDeviceId::from_ordinal(0)).expect("CUDA engine");
    Some(engine.runtime().clone())
}

fn f64_bytes(values: &[f64]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn from_f64_bytes(bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks_exact(8)
        .map(|word| f64::from_le_bytes(word.try_into().expect("8 bytes")))
        .collect()
}

/// `a * b + a` as two instructions over two inputs.
fn multiply_then_add() -> FusionPlan {
    FusionPlan {
        dtype: DType::F64,
        inputs: 2,
        instructions: vec![
            FusionInstruction {
                op: FusionOp::Mul,
                operands: vec![FusionOperand::Input(0), FusionOperand::Input(1)],
            },
            FusionInstruction {
                op: FusionOp::Add,
                operands: vec![FusionOperand::Value(0), FusionOperand::Input(0)],
            },
        ],
        outputs: vec![FusionOperand::Value(1)],
    }
}

#[test]
fn a_fused_region_matches_the_host_and_prepares_once() {
    let Some(runtime) = runtime() else { return };
    let len = 1024i64;
    let a: Vec<f64> = (0..len).map(|index| index as f64 * 0.5 - 3.0).collect();
    let b: Vec<f64> = (0..len).map(|index| 1.0 + (index % 7) as f64).collect();
    let input_a = runtime.upload_bytes(&f64_bytes(&a)).expect("a");
    let input_b = runtime.upload_bytes(&f64_bytes(&b)).expect("b");
    let out = runtime
        .upload_bytes(&vec![0u8; len as usize * 8])
        .expect("out");

    let prepared = PreparedFusion::prepare(&runtime, &multiply_then_add()).expect("prepare");
    let cold = prepared.counters();
    assert_eq!(
        cold.compile_attempts, 1,
        "a cold prepare compiles the emitted source exactly once"
    );
    assert_eq!(cold.module_loads, 1, "the emitted entry loads one module");

    prepared
        .launch(len, &[&out, &input_a, &input_b])
        .expect("launch");
    let observed = from_f64_bytes(&runtime.download_bytes(&out).expect("download"));
    let expected: Vec<f64> = a
        .iter()
        .zip(b.iter())
        .map(|(left, right)| left * right + left)
        .collect();
    assert_eq!(observed, expected);

    // A second preparation of the same plan serves the artifact cache: no NVRTC.
    let warm_prepared = PreparedFusion::prepare(&runtime, &multiply_then_add()).expect("prepare");
    let warm = warm_prepared.counters();
    assert_eq!(
        warm.compile_attempts, cold.compile_attempts,
        "a warm prepare performs zero NVRTC compilations"
    );
    assert!(
        warm.artifact_hits >= 1,
        "the artifact cache served the source"
    );
    assert_eq!(
        warm.module_loads, cold.module_loads,
        "the runtime's module cache serves the entry, so no second load"
    );
    assert_eq!(prepared.key(), warm_prepared.key());

    // A warm run of the prepared region also compiles nothing.
    warm_prepared
        .launch(len, &[&out, &input_a, &input_b])
        .expect("warm launch");
    assert_eq!(
        warm_prepared.counters().compile_attempts,
        warm.compile_attempts,
        "a warm run performs zero NVRTC compilations"
    );
}

#[test]
fn a_region_with_the_wrong_operand_count_is_refused() {
    let Some(runtime) = runtime() else { return };
    let prepared = PreparedFusion::prepare(&runtime, &multiply_then_add()).expect("prepare");
    let a = runtime.upload_bytes(&f64_bytes(&[1.0, 2.0])).expect("a");
    let out = runtime.upload_bytes(&[0u8; 16]).expect("out");
    // The plan needs one output and two inputs.
    assert!(prepared.launch(2, &[&out, &a]).is_err());
}

#[test]
fn an_ineligible_plan_is_refused_before_any_compile() {
    let Some(runtime) = runtime() else { return };
    let before = runtime.compile_attempts();
    // Bool is not an eligible fusion dtype.
    let plan = FusionPlan {
        dtype: DType::Bool,
        inputs: 1,
        instructions: vec![],
        outputs: vec![FusionOperand::Input(0)],
    };
    assert!(PreparedFusion::prepare(&runtime, &plan).is_err());
    assert_eq!(
        runtime.compile_attempts(),
        before,
        "a refused plan never reaches NVRTC"
    );
}
