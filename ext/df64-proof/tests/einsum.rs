//! The matrix contraction #1793 asks for, in the external scalar.
//!
//! #1793's example is `einsum("ik,kj->ij", A, B)` evaluated in Df64, and its precision row is the
//! contraction of the row `[1, 1]` with the column `[1, 2^-80]`, which has to keep `2^-80` after
//! subtracting one. Both run here through the runtime's extension module, using the same public
//! boundary the other contribution operations use. Patterns other than a matrix contraction are
//! refused with a typed error, which `extension_qr.rs`-style boundary tests assert below.

use std::sync::Arc;

use tenferro_ad::semantic_extension::SemanticExtensionRuleSet;
use tenferro_ad::AdContext;
use tenferro_cpu::CpuBackend;
use tenferro_df64_proof::ad::{Df64LinearizeRule, Df64VjpRule};
use tenferro_df64_proof::extension::{module, Df64Einsum, DF64_SCALAR_IDENTITY};
use tenferro_df64_proof::Df64;
use tenferro_runtime::extension::apply;
use tenferro_runtime::{GraphCompiler, Runtime, TracedTensor};
use tenferro_tensor::{DType, Tensor};
use tenferro_tensor_core::{ErasedHostTensor, HostTensor};

fn external(values: Vec<Df64>, shape: Vec<usize>) -> Tensor {
    Tensor::external(ErasedHostTensor::new(
        HostTensor::from_vec_col_major(shape, values).expect("shape matches data"),
    ))
}

fn payload(tensor: &Tensor) -> Vec<Df64> {
    match tensor {
        Tensor::External(value, _) => value
            .downcast_ref::<Df64>()
            .expect("external element type")
            .as_slice()
            .to_vec(),
        other => panic!(
            "expected an externally defined payload, found {:?}",
            other.dtype()
        ),
    }
}

fn runtime_with_module() -> Runtime {
    let backend = CpuBackend::new();
    let mut builder = Runtime::builder();
    builder
        .register_engine(tenferro_cpu::runtime_engine_registration(&backend).expect("engine"))
        .expect("register the CPU engine");
    builder
        .install_extension_module(module().expect("module"))
        .expect("install the Df64 module");
    builder.build().expect("runtime with the module")
}

fn leaf(values: Vec<Df64>, shape: Vec<usize>) -> TracedTensor {
    TracedTensor::from_tensor_concrete_shape_declaring_scalar(
        external(values, shape),
        DF64_SCALAR_IDENTITY,
    )
    .expect("traced leaf")
}

/// Contract two matrices through the runtime and return the result's elements.
fn contract(
    lhs: Vec<Df64>,
    lhs_shape: [usize; 2],
    rhs: Vec<Df64>,
    rhs_shape: [usize; 2],
) -> Vec<Df64> {
    let op = Df64Einsum::new(&[0, 1], &[1, 2], &[0, 2]).expect("a matrix contraction");
    let lhs_leaf = leaf(lhs.clone(), lhs_shape.to_vec());
    let rhs_leaf = leaf(rhs.clone(), rhs_shape.to_vec());
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
    payload(&results[0])
}

#[test]
fn the_ordinary_matrix_contraction_matches_the_expected_product() {
    // #1793's table: einsum("ik,kj->ij", A, B) with A = [[1, 2], [3, 4]] and B = [[5, 6], [7, 8]]
    // is [[19, 22], [43, 50]].
    // The tensor is column-major, so the flat vector is the first column followed by the second.
    let values = |rows: [[f64; 2]; 2]| -> Vec<Df64> {
        (0..2)
            .flat_map(|column| (0..2).map(move |row| Df64::from_f64(rows[row][column])))
            .collect()
    };

    let results = contract(
        values([[1.0, 2.0], [3.0, 4.0]]),
        [2, 2],
        values([[5.0, 6.0], [7.0, 8.0]]),
        [2, 2],
    );

    assert_eq!(
        results,
        values([[19.0, 22.0], [43.0, 50.0]]),
        "the contraction must match the ordinary result"
    );
}

#[test]
fn the_contraction_keeps_the_low_component_an_f64_accumulator_would_drop() {
    let low = 2f64.powi(-80);
    // The row [1, 1] contracted with the column [1, 2^-80] is 1 + 2^-80.
    let results = contract(
        vec![Df64::from_f64(1.0), Df64::from_f64(1.0)],
        [1, 2],
        vec![Df64::from_f64(1.0), Df64 { hi: low, lo: 0.0 }],
        [2, 1],
    );

    assert_eq!(results.len(), 1);
    assert_eq!(results[0], Df64 { hi: 1.0, lo: low });
    assert_eq!(
        results[0] - Df64::from_f64(1.0),
        Df64 { hi: low, lo: 0.0 },
        "subtracting one must retain 2^-80 in the external scalar"
    );
    assert_eq!(results[0].narrow_to_f64(), 1.0);
    // Control: the same contraction in f64 loses the low component entirely.
    assert_eq!((1.0_f64 * 1.0 + 1.0 * low) - 1.0, 0.0);
}

#[test]
fn the_pattern_validator_refuses_anything_but_a_matrix_contraction() {
    // A trace repeats a label inside one input.
    assert!(Df64Einsum::new(&[0, 0], &[0, 2], &[0, 2]).is_err());
    // A rank-one input is not a matrix.
    assert!(Df64Einsum::new(&[0], &[0, 2], &[2]).is_err());
    // The output must be the free labels in the order the inputs name them.
    assert!(Df64Einsum::new(&[0, 1], &[1, 2], &[2, 0]).is_err());
    // A valid pattern reports the labels it was built from.
    let pattern = Df64Einsum::new(&[0, 1], &[1, 2], &[0, 2]).expect("a matrix contraction");
    assert_eq!(pattern.labels(), (&[0, 1][..], &[1, 2][..], &[0, 2][..]));
}

#[test]
fn the_body_refuses_disagreeing_contracted_dimensions() {
    let op = Df64Einsum::new(&[0, 1], &[1, 2], &[0, 2]).expect("a matrix contraction");
    let lhs = leaf(vec![Df64::from_f64(1.0); 2], vec![1, 2]);
    let rhs = leaf(vec![Df64::from_f64(1.0); 3], vec![3, 1]);
    let output = apply(Arc::new(op), &[&lhs, &rhs]).expect("traced contraction");

    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&output[0]).expect("compiled contraction");
    let error = runtime_with_module()
        .run_compiled(
            &program,
            &[
                &external(vec![Df64::from_f64(1.0); 2], vec![1, 2]),
                &external(vec![Df64::from_f64(1.0); 3], vec![3, 1]),
            ],
        )
        .expect_err("a 1x2 contraction with a 3x1 input has no shape");
    assert!(
        error.to_string().contains("contracted"),
        "the failure must name the contracted dimension: {error}"
    );
}

#[test]
fn the_operation_refuses_a_preset_scalar() {
    let op = Df64Einsum::new(&[0, 1], &[1, 2], &[0, 2]).expect("a matrix contraction");
    let lhs = TracedTensor::input_concrete_shape(DType::F64, &[1, 2]).expect("traced input");
    let rhs = TracedTensor::input_concrete_shape(DType::F64, &[2, 1]).expect("traced input");
    let error = apply(Arc::new(op), &[&lhs, &rhs])
        .expect_err("the body is defined for the externally defined scalar only");
    assert!(
        error.to_string().contains("externally defined"),
        "the failure must say why the dtype is unsupported: {error}"
    );
}

#[test]
fn differentiating_the_contraction_fails_explicitly() {
    // #1793 and #1788 require an unsupported AD mode to fail explicitly rather than as a zero
    // gradient, so this asserts the refusal instead of leaving it to the rule set's internals.
    let rules = SemanticExtensionRuleSet::new()
        .with_linearize(std::sync::Arc::new(Df64LinearizeRule))
        .expect("one linearize rule per family");
    let context = AdContext::builder()
        .with_semantic_extension_rules(rules)
        .expect("extension rules")
        .build()
        .expect("ad context");

    let op = Df64Einsum::new(&[0, 1], &[1, 2], &[0, 2]).expect("a matrix contraction");
    let lhs = leaf(vec![Df64::from_f64(1.0); 2], vec![1, 2]);
    let rhs = leaf(vec![Df64::from_f64(1.0); 2], vec![2, 1]);
    let output = apply(std::sync::Arc::new(op), &[&lhs, &rhs]).expect("traced contraction");
    let tangent = leaf(vec![Df64::from_f64(1.0); 2], vec![1, 2]);

    let error = context
        .jvp(&output[0], &lhs, &tangent)
        .expect_err("the contribution has no rule for the contraction");
    let message = error.to_string();
    println!("contraction AD refusal: {message}");
    assert!(
        message.contains("df64") || message.contains("unsupported") || message.contains("rule"),
        "the refusal must say which rule is missing: {message}"
    );
}
