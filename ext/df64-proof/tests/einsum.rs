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
    match tensor.external_payload() {
        Some(value) => value
            .downcast_ref::<Df64>()
            .expect("external element type")
            .as_slice()
            .to_vec(),
        None => panic!("expected an externally defined payload"),
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
fn the_pattern_validator_accepts_any_two_input_contraction_and_reports_its_labels() {
    // The caller may write any labels; the validator checks structure, not the matrix case.
    let pattern = Df64Einsum::new(&[0, 1], &[1, 2], &[0, 2]).expect("a contraction");
    assert_eq!(
        pattern.labels(),
        Some((&[0, 1][..], &[1, 2][..], &[0, 2][..]))
    );
    assert_eq!(pattern.input_labels(), &[vec![0, 1], vec![1, 2]]);
    assert_eq!(pattern.out_labels(), &[0, 2]);
    assert!(Df64Einsum::new(&[7, 9], &[9, 4], &[7, 4]).is_ok());
    // An input with no labels is not a tensor.
    assert!(Df64Einsum::new(&[], &[0, 2], &[2]).is_err());
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
        error.to_string().contains("shared label"),
        "the failure must say the inputs disagree on a shared label: {error}"
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
fn differentiating_the_contraction_is_supported_in_both_modes() {
    // The forward tangent used to be refused here. It is implemented now, and its values and its
    // duality with the adjoint are checked in `einsum_ad.rs`; this test keeps the rule set up
    // honest by building one and asserting that the refusal is gone rather than silently stale.
    let rules = SemanticExtensionRuleSet::new()
        .with_linearize(std::sync::Arc::new(Df64LinearizeRule))
        .expect("one linearize rule per family")
        .with_primal_vjp(std::sync::Arc::new(Df64VjpRule))
        .expect("one adjoint rule per family");
    let context = AdContext::builder()
        .with_semantic_extension_rules(rules)
        .expect("extension rules")
        .build()
        .expect("ad context");

    let op = Df64Einsum::new(&[0, 1], &[1, 2], &[0, 2]).expect("a contraction");
    let lhs = leaf(vec![Df64::from_f64(1.0); 2], vec![1, 2]);
    let rhs = leaf(vec![Df64::from_f64(1.0); 2], vec![2, 1]);
    let output = apply(std::sync::Arc::new(op), &[&lhs, &rhs]).expect("traced contraction");
    let tangent = leaf(vec![Df64::from_f64(1.0); 2], vec![1, 2]);
    let cotangent = leaf(vec![Df64::from_f64(1.0)], vec![1, 1]);

    assert!(
        context.jvp(&output[0], &lhs, &tangent).is_ok(),
        "the forward tangent of the contraction is implemented"
    );
    assert!(
        context.vjp(&output[0], &lhs, &cotangent).is_ok(),
        "the adjoint of the contraction is implemented"
    );
}

/// Contract with an explicit pattern, returning the result's elements.
fn contract_with(
    lhs: Vec<Df64>,
    lhs_shape: Vec<usize>,
    rhs: Vec<Df64>,
    rhs_shape: Vec<usize>,
    pattern: (&[u32], &[u32], &[u32]),
) -> Vec<Df64> {
    let op = Df64Einsum::new(pattern.0, pattern.1, pattern.2).expect("a valid pattern");
    let lhs_leaf = leaf(lhs.clone(), lhs_shape.clone());
    let rhs_leaf = leaf(rhs.clone(), rhs_shape.clone());
    let output =
        apply(std::sync::Arc::new(op), &[&lhs_leaf, &rhs_leaf]).expect("traced contraction");
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&output[0]).expect("compiled contraction");
    let results = runtime_with_module()
        .run_compiled(
            &program,
            &[&external(lhs, lhs_shape), &external(rhs, rhs_shape)],
        )
        .expect("contraction execution");
    payload(&results[0])
}

fn numbers(values: &[f64]) -> Vec<Df64> {
    values.iter().copied().map(Df64::from_f64).collect()
}

#[test]
fn a_batched_contraction_with_a_free_batch_label_runs() {
    // "bij,bjk->bik" with two batches of a 1x1 matrix: each output entry is that batch's product,
    // so the batch label is free rather than contracted.
    let results = contract_with(
        numbers(&[1.0, 2.0]),
        vec![2, 1, 1],
        numbers(&[3.0, 4.0]),
        vec![2, 1, 1],
        (&[0, 1, 2], &[0, 2, 3], &[0, 1, 3]),
    );
    assert_eq!(results, numbers(&[3.0, 8.0]));
}

#[test]
fn an_outer_product_has_no_contracted_label() {
    // "i,j->ij" has no shared label, so the contraction is a product.
    let results = contract_with(
        numbers(&[2.0, 3.0]),
        vec![2],
        numbers(&[5.0, 7.0]),
        vec![2],
        (&[0], &[1], &[0, 1]),
    );
    // [[10, 14], [15, 21]] in column-major order.
    assert_eq!(results, numbers(&[10.0, 15.0, 14.0, 21.0]));
}

#[test]
fn a_label_only_one_input_names_and_the_output_omits_is_summed() {
    // "ij,kl->ik" sums each input over its omitted label before multiplying, which is what the
    // notation means by a label the output does not name.
    let results = contract_with(
        numbers(&[1.0, 3.0, 2.0, 4.0]),
        vec![2, 2],
        numbers(&[5.0, 7.0, 6.0, 8.0]),
        vec![2, 2],
        (&[0, 1], &[2, 3], &[0, 2]),
    );
    // Rows of A sum to [3, 7] and rows of B to [11, 15], so the product is [[33, 45], [77, 105]].
    assert_eq!(results, numbers(&[33.0, 77.0, 45.0, 105.0]));
}

#[test]
fn a_repeated_label_extracts_the_diagonal_before_contracting() {
    // "iij,jk->ik" takes the diagonal of the first operand's first two axes. With A[i, i, 0] =
    // [1, 2] and B = [[3, 4]] the result is the diagonal placed against B's row: [[3, 4], [6, 8]],
    // which is [3, 6, 4, 8] in column-major order.
    let results = contract_with(
        numbers(&[1.0, 0.0, 0.0, 2.0]),
        vec![2, 2, 1],
        numbers(&[3.0, 4.0]),
        vec![1, 2],
        (&[0, 0, 1], &[1, 2], &[0, 2]),
    );
    assert_eq!(results, numbers(&[3.0, 6.0, 4.0, 8.0]));
}

#[test]
fn a_repeated_label_the_output_omits_is_a_trace() {
    // "ii,j->j" sums the repeated label's diagonal: with A[i, i] = [1, 2] the trace is 3, and the
    // scalar operand leaves it unchanged.
    let results = contract_with(
        numbers(&[1.0, 0.0, 0.0, 2.0]),
        vec![2, 2],
        numbers(&[1.0]),
        vec![1],
        (&[0, 0], &[1], &[1]),
    );
    assert_eq!(results, numbers(&[3.0]));
}

/// Contract any number of operands with an explicit pattern, returning the result's elements.
fn contract_nary(
    operands: Vec<(Vec<Df64>, Vec<usize>)>,
    pattern: (&[&[u32]], &[u32]),
) -> Vec<Df64> {
    let op = Df64Einsum::new_nary(pattern.0, pattern.1).expect("a valid pattern");
    let leaves: Vec<TracedTensor> = operands
        .iter()
        .map(|(values, shape)| leaf(values.clone(), shape.clone()))
        .collect();
    let refs: Vec<&TracedTensor> = leaves.iter().collect();
    let output = apply(std::sync::Arc::new(op), &refs).expect("traced contraction");

    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&output[0]).expect("compiled contraction");
    let values: Vec<Tensor> = operands
        .iter()
        .map(|(values, shape)| external(values.clone(), shape.clone()))
        .collect();
    let borrowed: Vec<&Tensor> = values.iter().collect();
    let results = runtime_with_module()
        .run_compiled(&program, &borrowed)
        .expect("contraction execution");
    payload(&results[0])
}

#[test]
fn a_three_operand_contraction_folds_from_the_left() {
    // "ij,jk,kl->il" with A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]] and C = I is A times B. The
    // intermediate keeps the contracted label `k` because the third operand still needs it, and
    // sums `j` because nothing after the first step does.
    let results = contract_nary(
        vec![
            (numbers(&[1.0, 3.0, 2.0, 4.0]), vec![2, 2]),
            (numbers(&[5.0, 7.0, 6.0, 8.0]), vec![2, 2]),
            (numbers(&[1.0, 0.0, 0.0, 1.0]), vec![2, 2]),
        ],
        (&[&[0, 1], &[1, 2], &[2, 3]], &[0, 3]),
    );
    assert_eq!(results, numbers(&[19.0, 43.0, 22.0, 50.0]));
}

#[test]
fn a_three_operand_contraction_with_a_repeated_final_operand_runs() {
    // "iij,jk,kl->il" repeats a label in the first operand, which is the diagonal, and the fold
    // carries that through the remaining steps.
    let results = contract_nary(
        vec![
            (numbers(&[1.0, 0.0, 0.0, 2.0]), vec![2, 2, 1]),
            (numbers(&[3.0, 4.0]), vec![1, 2]),
            (numbers(&[1.0, 0.0, 0.0, 1.0]), vec![2, 2]),
        ],
        (&[&[0, 0, 1], &[1, 2], &[2, 3]], &[0, 3]),
    );
    // The diagonal [1, 2] times B = [[3, 4]] gives [[3, 4], [6, 8]].
    assert_eq!(results, numbers(&[3.0, 6.0, 4.0, 8.0]));
}

#[test]
fn the_validator_refuses_a_single_operand() {
    assert!(
        Df64Einsum::new_nary(&[&[0, 1]], &[0]).is_err(),
        "a contraction takes at least two operands"
    );
}

#[test]
fn a_label_the_output_names_is_not_contracted() {
    // "ij,ij->ij" is the Hadamard product, which #1793 lists as a reached operation. The labels are
    // shared but the output names them, so the body multiplies elementwise rather than contracting,
    // which is what makes the row above a real case rather than a restatement of the matrix one.
    let results = contract_with(
        numbers(&[1.0, 3.0, 2.0, 4.0]),
        vec![2, 2],
        numbers(&[5.0, 7.0, 6.0, 8.0]),
        vec![2, 2],
        (&[0, 1], &[0, 1], &[0, 1]),
    );
    // [[5, 12], [21, 32]] in column-major order.
    assert_eq!(results, numbers(&[5.0, 21.0, 12.0, 32.0]));
}
