//! #1793's bf16 einsum row: f32 accumulation with a single output rounding.
//!
//! The row asks for a contraction whose result distinguishes f32 accumulation from repeated
//! bfloat16 rounding, compared against an independent reference with the specified output rounding.
//! The first test is that comparison: summing three hundred stored ones by contraction gives three
//! hundred through the promised accumulation, while the same sum rounded at every step stalls where
//! bfloat16 spacing above one becomes two.

use std::sync::Arc;

use tenferro_bf16_proof::einsum::{module, Bf16Einsum};
use tenferro_bf16_proof::Bf16;
use tenferro_cpu::CpuBackend;
use tenferro_runtime::extension::apply;
use tenferro_runtime::{GraphCompiler, Runtime, TracedTensor};
use tenferro_tensor::{DType, Tensor};
use tenferro_tensor_core::{ErasedHostTensor, HostTensor};

fn runtime_with_module() -> Runtime {
    let backend = CpuBackend::new();
    let mut builder = Runtime::builder();
    builder
        .register_engine(tenferro_cpu::runtime_engine_registration(&backend).expect("engine"))
        .expect("register the CPU engine");
    builder
        .install_extension_module(module().expect("module"))
        .expect("install the bf16 module");
    builder.build().expect("runtime with the module")
}

fn external(values: &[f32], shape: Vec<usize>) -> Tensor {
    Tensor::external(ErasedHostTensor::new(
        HostTensor::from_vec_col_major(shape, values.iter().copied().map(Bf16::from_f32).collect())
            .expect("shape matches data"),
    ))
}

fn values(tensor: &Tensor) -> Vec<f32> {
    match tensor {
        Tensor::External(payload, _) => payload
            .downcast_ref::<Bf16>()
            .expect("bfloat16 payload")
            .as_slice()
            .iter()
            .map(|value| value.to_f32())
            .collect(),
        other => panic!("expected an external payload, found {:?}", other.dtype()),
    }
}

/// Contract two bfloat16 operands through the runtime and return the result's elements.
fn contract(lhs: &[f32], lhs_shape: [usize; 2], rhs: &[f32], rhs_shape: [usize; 2]) -> Vec<f32> {
    let op = Bf16Einsum::new(&[0, 1], &[1, 2], &[0, 2]).expect("a contraction");
    let lhs_leaf = TracedTensor::from_tensor_concrete_shape_declaring_scalar(
        external(lhs, lhs_shape.to_vec()),
        tenferro_bf16_proof::einsum::BF16_SCALAR_IDENTITY,
    )
    .expect("traced operand");
    let rhs_leaf = TracedTensor::from_tensor_concrete_shape_declaring_scalar(
        external(rhs, rhs_shape.to_vec()),
        tenferro_bf16_proof::einsum::BF16_SCALAR_IDENTITY,
    )
    .expect("traced operand");
    let output = apply(Arc::new(op), &[&lhs_leaf, &rhs_leaf]).expect("traced contraction");

    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&output[0]).expect("compiled contraction");
    let results = runtime_with_module()
        .run_compiled(
            &program,
            &[
                &external(lhs, lhs_shape.to_vec()),
                &external(rhs, rhs_shape.to_vec()),
            ],
        )
        .expect("contraction execution");
    values(&results[0])
}

#[test]
fn the_contraction_accumulates_in_f32_rather_than_rounding_at_every_step() {
    let count = 300usize;
    let lhs = vec![1.0_f32; count];
    let rhs = vec![1.0_f32; count];

    let result = contract(&lhs, [1, count], &rhs, [count, 1]);
    assert_eq!(result.len(), 1);
    assert_eq!(
        result[0], 300.0,
        "f32 accumulation must carry the sum; rounding at every step would stall at 256"
    );

    // The independent reference: the same sum with the storage type's own addition, which rounds
    // once per step. It is a different number, which is what makes the promise testable.
    let per_step = (0..count).fold(Bf16::from_f32(0.0), |total, _| total + Bf16::from_f32(1.0));
    assert_eq!(per_step.to_f32(), 256.0);
    assert_ne!(result[0], per_step.to_f32());
}

#[test]
fn the_ordinary_matrix_contraction_matches_the_expected_product() {
    // A = [[1, 2], [3, 4]] and B = [[5, 6], [7, 8]] in column-major order contract to
    // [[19, 22], [43, 50]], whose entries are exactly representable in bfloat16.
    let result = contract(&[1.0, 3.0, 2.0, 4.0], [2, 2], &[5.0, 7.0, 6.0, 8.0], [2, 2]);
    assert_eq!(result, vec![19.0, 43.0, 22.0, 50.0]);
}

#[test]
fn the_output_is_rounded_once_to_the_storage_type() {
    // Ten stored values of 1 + 2^-7 — the spacing of bfloat16 on [1, 2), so it survives storage —
    // sum to ten times that in f32 and round once on the way out.
    let stored = Bf16::from_f32(1.0 + 2f32.powi(-7)).to_f32();
    assert_eq!(stored, 1.0 + 2f32.powi(-7));
    let lhs = vec![stored; 10];
    let rhs = vec![1.0_f32; 10];
    let result = contract(&lhs, [1, 10], &rhs, [10, 1]);
    let exact = stored * 10.0;
    assert_eq!(result.len(), 1);
    assert_eq!(
        result[0],
        Bf16::from_f32(exact).to_f32(),
        "the output is the rounded accumulation, not an accumulation of rounded values"
    );
}

#[test]
fn the_operation_refuses_a_preset_scalar() {
    // The body is defined for the externally defined scalar, so a preset dtype fails explicitly.
    let op = Bf16Einsum::new(&[0, 1], &[1, 2], &[0, 2]).expect("a contraction");
    let lhs = TracedTensor::input_concrete_shape(DType::F32, &[1, 2]).expect("traced operand");
    let rhs = TracedTensor::input_concrete_shape(DType::F32, &[2, 1]).expect("traced operand");
    let error = apply(Arc::new(op), &[&lhs, &rhs])
        .expect_err("the body is defined for the externally defined scalar only");
    assert!(
        error.to_string().contains("externally defined"),
        "the failure must say why the dtype is unsupported: {error}"
    );
}

#[test]
fn the_pattern_validator_refuses_a_repeated_label() {
    assert!(
        Bf16Einsum::new(&[0, 0], &[0, 2], &[0, 2]).is_err(),
        "a trace is not part of this module's pattern surface"
    );
    assert!(Bf16Einsum::new(&[], &[1, 2], &[0, 2]).is_err());
}
