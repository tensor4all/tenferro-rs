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
#[ignore = "requires CUDA"]
fn a_region_with_the_wrong_operand_count_is_refused() {
    let Some(runtime) = runtime() else { return };
    let prepared = PreparedFusion::prepare(&runtime, &multiply_then_add()).expect("prepare");
    let a = runtime.upload_bytes(&f64_bytes(&[1.0, 2.0])).expect("a");
    let out = runtime.upload_bytes(&[0u8; 16]).expect("out");
    // The plan needs one output and two inputs.
    assert!(prepared.launch(2, &[&out, &a]).is_err());
}

#[test]
#[ignore = "requires CUDA"]
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
