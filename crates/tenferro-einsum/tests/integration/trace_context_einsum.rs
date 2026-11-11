use tenferro_einsum::{
    parse_einsum_subscripts, EinsumAxis, EinsumNotation, EinsumOptimize, EinsumOptions,
    TracedEinsumOps, EINSUM_EXTENSION_FAMILY_ID,
};
use tenferro_ops::dim_expr::DimExpr;
use tenferro_runtime::program::{ProgramInputSpec, SemanticOpRef};
use tenferro_runtime::{CompiledGraph, Error as RuntimeError, GraphCompiler, TraceContext};
use tenferro_tensor::{DType, Error as TensorError, ShapeMismatch, Tensor, ValidationError};

use super::support;

#[test]
fn trace_context_einsum_ext_emits_one_ordered_semantic_extension() {
    let matrix = || ProgramInputSpec::new(DType::F64, [DimExpr::Const(2), DimExpr::Const(2)]);
    let trace = TraceContext::new();
    let lhs = trace.derive_tensor(trace.input(matrix()).unwrap()).unwrap();
    let rhs = trace.derive_tensor(trace.input(matrix()).unwrap()).unwrap();

    let output = trace.einsum(&[&lhs, &rhs], "ij,jk->ik").unwrap();
    let graph = trace.finish(&[output.trace_value()]).unwrap();
    let operations = graph.program().operations().collect::<Vec<_>>();

    assert_eq!(operations.len(), 1);
    assert_eq!(operations[0].inputs(), graph.program().inputs());
    assert_eq!(operations[0].outputs(), graph.program().outputs());
    assert_eq!(
        graph
            .program()
            .value_metadata(graph.program().outputs()[0])
            .unwrap()
            .shape()
            .iter()
            .map(|extent| extent.as_exact())
            .collect::<Vec<_>>(),
        [Some(&DimExpr::Const(2)), Some(&DimExpr::Const(2))]
    );
}

#[test]
fn trace_context_einsum_supports_ellipsis_broadcast_and_programmatic_notation() {
    let batch_matrix = || {
        ProgramInputSpec::new(
            DType::F64,
            [DimExpr::Const(2), DimExpr::Const(2), DimExpr::Const(3)],
        )
    };
    let single_batch_matrix = || {
        ProgramInputSpec::new(
            DType::F64,
            [DimExpr::Const(1), DimExpr::Const(3), DimExpr::Const(2)],
        )
    };
    let notation = EinsumNotation::new(
        &[
            &[
                EinsumAxis::Ellipsis,
                EinsumAxis::Label(0),
                EinsumAxis::Label(1),
            ],
            &[
                EinsumAxis::Ellipsis,
                EinsumAxis::Label(1),
                EinsumAxis::Label(2),
            ],
        ],
        &[
            EinsumAxis::Ellipsis,
            EinsumAxis::Label(0),
            EinsumAxis::Label(2),
        ],
    );

    let trace = TraceContext::new();
    let lhs = trace
        .derive_tensor(trace.input(batch_matrix()).unwrap())
        .unwrap();
    let rhs = trace
        .derive_tensor(trace.input(single_batch_matrix()).unwrap())
        .unwrap();
    let string_output = trace.einsum(&[&lhs, &rhs], "...ij,...jk->...ik").unwrap();
    let string_graph = trace.finish(&[string_output.trace_value()]).unwrap();
    let compiled = GraphCompiler::new()
        .compile_traced_graph(&string_graph)
        .unwrap();
    let lhs_tensor = Tensor::from_vec_col_major(vec![2, 2, 3], vec![1.0_f64; 12]).unwrap();
    let rhs_tensor = Tensor::from_vec_col_major(vec![1, 3, 2], vec![1.0_f64; 6]).unwrap();
    let result = support::run_one(&compiled, &[&lhs_tensor, &rhs_tensor]).unwrap();
    assert_eq!(result.shape(), &[2, 2, 2]);
    assert_eq!(result.as_slice::<f64>().unwrap(), &[3.0; 8]);
    assert_eq!(
        string_graph
            .program()
            .value_metadata(string_graph.program().outputs()[0])
            .unwrap()
            .shape()
            .iter()
            .map(|extent| extent.as_exact())
            .collect::<Vec<_>>(),
        vec![
            Some(&DimExpr::Const(2)),
            Some(&DimExpr::Const(2)),
            Some(&DimExpr::Const(2)),
        ]
    );

    let trace = TraceContext::new();
    let lhs = trace
        .derive_tensor(trace.input(batch_matrix()).unwrap())
        .unwrap();
    let rhs = trace
        .derive_tensor(trace.input(single_batch_matrix()).unwrap())
        .unwrap();
    let output = trace.einsum(&[&lhs, &rhs], &notation).unwrap();
    let graph = trace.finish(&[output.trace_value()]).unwrap();
    assert_eq!(
        graph.program().semantic_fingerprint(),
        string_graph.program().semantic_fingerprint()
    );
}

#[test]
fn trace_context_einsum_ext_exposes_all_four_entry_points() {
    let parsed = parse_einsum_subscripts("ij,jk->ik").unwrap();
    let mut fingerprints = Vec::new();
    for entry in 0..4 {
        let matrix = || ProgramInputSpec::new(DType::F64, [DimExpr::Const(2), DimExpr::Const(2)]);
        let trace = TraceContext::new();
        let lhs = trace.derive_tensor(trace.input(matrix()).unwrap()).unwrap();
        let rhs = trace.derive_tensor(trace.input(matrix()).unwrap()).unwrap();
        let inputs = [&lhs, &rhs];
        let output = match entry {
            0 => trace.einsum(&inputs, "ij,jk->ik").unwrap(),
            1 => trace.einsum(&inputs, &parsed).unwrap(),
            2 => trace
                .einsum_with(
                    &inputs,
                    "ij,jk->ik",
                    &EinsumOptions::default().optimize(EinsumOptimize::False),
                )
                .unwrap(),
            3 => trace
                .einsum_with(
                    &inputs,
                    &parsed,
                    &EinsumOptions::default().optimize(EinsumOptimize::Path(vec![(0, 1)])),
                )
                .unwrap(),
            _ => unreachable!(),
        };
        let graph = trace.finish(&[output.trace_value()]).unwrap();
        fingerprints.push(graph.program().semantic_fingerprint());
    }

    assert_eq!(fingerprints[0], fingerprints[1]);
    assert_ne!(fingerprints[1], fingerprints[2]);
    assert_ne!(fingerprints[2], fingerprints[3]);
}

#[test]
fn trace_context_einsum_rejects_foreign_values_without_mutating_the_trace() {
    let vector = || ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]);
    let owner = TraceContext::new();
    let foreign = owner.derive_tensor(owner.input(vector()).unwrap()).unwrap();
    let trace = TraceContext::new();
    let local = trace.derive_tensor(trace.input(vector()).unwrap()).unwrap();

    let error = trace.einsum(&[&local, &foreign], "i,i->").unwrap_err();
    assert!(matches!(
        error.kind(),
        tenferro_tensor::ErrorKind::Validation(_)
    ));

    let graph = trace.finish(&[local.trace_value()]).unwrap();
    assert!(graph.program().operations().next().is_none());
}

#[test]
fn traced_semantic_einsum_compiles_and_executes_with_ordered_inputs() {
    let matrix = || ProgramInputSpec::new(DType::F64, [DimExpr::Const(2), DimExpr::Const(2)]);
    let trace = TraceContext::new();
    let lhs = trace.derive_tensor(trace.input(matrix()).unwrap()).unwrap();
    let rhs = trace.derive_tensor(trace.input(matrix()).unwrap()).unwrap();
    let output = trace.einsum(&[&lhs, &rhs], "ij,jk->ik").unwrap();
    let traced = trace.finish(&[output.trace_value()]).unwrap();
    let compiled = GraphCompiler::new().compile_traced_graph(&traced).unwrap();

    let lhs = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 3.0, 2.0, 4.0]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![2, 2], vec![5.0_f64, 7.0, 6.0, 8.0]).unwrap();
    let result = support::run_one(&compiled, &[&lhs, &rhs]).unwrap();

    let operation = compiled.program().operations().next().unwrap();
    let SemanticOpRef::Extension(extension) = operation.op() else {
        panic!("expected semantic einsum extension");
    };
    assert_eq!(extension.family_id(), EINSUM_EXTENSION_FAMILY_ID);
    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result.as_slice::<f64>().unwrap(), &[19.0, 43.0, 22.0, 50.0]);
}

#[test]
fn traced_einsum_broadcasts_singleton_contracting_extent() {
    let lhs_spec = || ProgramInputSpec::new(DType::F64, [DimExpr::Const(2), DimExpr::Const(1)]);
    let rhs_spec = || {
        ProgramInputSpec::new(
            DType::F64,
            [DimExpr::Const(4), DimExpr::Const(2), DimExpr::Const(3)],
        )
    };
    let parsed = parse_einsum_subscripts("ij,pqj->pqi").unwrap();
    let lhs_tensor = Tensor::from_vec_col_major(vec![2, 1], vec![2.0_f64; 2]).unwrap();
    let rhs_tensor = Tensor::from_vec_col_major(vec![4, 2, 3], vec![1.0_f64; 24]).unwrap();

    for entry in 0..3 {
        let trace = TraceContext::new();
        let lhs = trace
            .derive_tensor(trace.input(lhs_spec()).unwrap())
            .unwrap();
        let rhs = trace
            .derive_tensor(trace.input(rhs_spec()).unwrap())
            .unwrap();
        let output = match entry {
            0 => trace.einsum(&[&lhs, &rhs], "ij,pqj->pqi").unwrap(),
            1 => trace.einsum(&[&lhs, &rhs], &parsed).unwrap(),
            _ => trace
                .einsum_with(
                    &[&lhs, &rhs],
                    &parsed,
                    &tenferro_einsum::EinsumOptions::default()
                        .optimize(EinsumOptimize::Path(vec![(0, 1)])),
                )
                .unwrap(),
        };
        let graph = trace.finish(&[output.trace_value()]).unwrap();
        assert_eq!(
            graph
                .program()
                .value_metadata(graph.program().outputs()[0])
                .unwrap()
                .shape()
                .iter()
                .map(|extent| extent.as_exact())
                .collect::<Vec<_>>(),
            [
                Some(&DimExpr::Const(4)),
                Some(&DimExpr::Const(2)),
                Some(&DimExpr::Const(2)),
            ]
        );
        let compiled = GraphCompiler::new().compile_traced_graph(&graph).unwrap();
        let result = support::run_one(&compiled, &[&lhs_tensor, &rhs_tensor]).unwrap();
        assert_eq!(result.shape(), &[4, 2, 2]);
        assert_eq!(result.as_slice::<f64>().unwrap(), &[6.0; 16]);
    }
}

#[test]
fn traced_einsum_broadcasts_singleton_extent_with_symbolic_axes() {
    let trace = TraceContext::new();
    let lhs = trace
        .derive_tensor(
            trace
                .input(ProgramInputSpec::new(
                    DType::F64,
                    [
                        DimExpr::InputDim {
                            input_idx: 0,
                            axis: 0,
                        },
                        DimExpr::Const(1),
                    ],
                ))
                .unwrap(),
        )
        .unwrap();
    let rhs = trace
        .derive_tensor(
            trace
                .input(ProgramInputSpec::new(
                    DType::F64,
                    [DimExpr::Const(4), DimExpr::Const(2), DimExpr::Const(3)],
                ))
                .unwrap(),
        )
        .unwrap();
    let output = trace.einsum(&[&lhs, &rhs], "ij,pqj->pqi").unwrap();
    let graph = trace.finish(&[output.trace_value()]).unwrap();
    let compiled = GraphCompiler::new().compile_traced_graph(&graph).unwrap();
    let lhs_tensor = Tensor::from_vec_col_major(vec![4, 1], vec![2.0_f64; 4]).unwrap();
    let rhs_tensor = Tensor::from_vec_col_major(vec![4, 2, 3], vec![1.0_f64; 24]).unwrap();

    let result = support::run_one(&compiled, &[&lhs_tensor, &rhs_tensor]).unwrap();

    assert_eq!(result.shape(), &[4, 2, 4]);
    assert_eq!(result.as_slice::<f64>().unwrap(), &[6.0; 32]);
}

/// A symbolic extent shared with a constant non-1 extent must merge to that
/// constant once an unrelated label turns broadcasting on: `j` is `1` vs `2`,
/// so the op broadcasts, and the lhs `i` (bound to `1`) broadcasts against the
/// rhs `i = 3`. Both the compiled output metadata and the executed output are
/// `[3]`.
#[test]
fn traced_einsum_merges_symbolic_extent_with_constant_under_unrelated_broadcast() {
    let trace = TraceContext::new();
    let lhs = trace
        .derive_tensor(
            trace
                .input(ProgramInputSpec::new(
                    DType::F64,
                    [
                        DimExpr::InputDim {
                            input_idx: 0,
                            axis: 0,
                        },
                        DimExpr::Const(1),
                    ],
                ))
                .unwrap(),
        )
        .unwrap();
    let rhs = trace
        .derive_tensor(
            trace
                .input(ProgramInputSpec::new(
                    DType::F64,
                    [DimExpr::Const(3), DimExpr::Const(2)],
                ))
                .unwrap(),
        )
        .unwrap();
    let output = trace.einsum(&[&lhs, &rhs], "ij,ij->i").unwrap();
    let graph = trace.finish(&[output.trace_value()]).unwrap();
    let compiled = GraphCompiler::new().compile_traced_graph(&graph).unwrap();

    let output_extents: Vec<_> = compiled
        .program()
        .value_metadata(compiled.program().outputs()[0])
        .unwrap()
        .shape()
        .iter()
        .map(|extent| extent.as_exact().cloned())
        .collect();
    assert_eq!(output_extents, [Some(DimExpr::Const(3))]);

    let lhs_tensor = Tensor::from_vec_col_major(vec![1, 1], vec![5.0_f64]).unwrap();
    // rhs[i, j] column-major: rhs[:, 0] = [1, 2, 3], rhs[:, 1] = [4, 5, 6].
    let rhs_tensor =
        Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let result = support::run_one(&compiled, &[&lhs_tensor, &rhs_tensor]).unwrap();

    assert_eq!(result.shape(), &[3]);
    assert_eq!(result.as_slice::<f64>().unwrap(), &[25.0, 35.0, 45.0]);
}

/// `"ij,ij->i"` with a separate symbolic `i` per operand and constant `j`
/// (`1` vs `2`), so the known `j` mismatch turns broadcasting on.
fn compile_ij_ij_to_i_with_separate_symbolic_i() -> CompiledGraph {
    let symbolic_i = |input_idx| {
        ProgramInputSpec::new(
            DType::F64,
            [
                DimExpr::InputDim { input_idx, axis: 0 },
                DimExpr::Const(if input_idx == 0 { 1 } else { 2 }),
            ],
        )
    };
    let trace = TraceContext::new();
    let lhs = trace
        .derive_tensor(trace.input(symbolic_i(0)).unwrap())
        .unwrap();
    let rhs = trace
        .derive_tensor(trace.input(symbolic_i(1)).unwrap())
        .unwrap();
    let output = trace.einsum(&[&lhs, &rhs], "ij,ij->i").unwrap();
    let graph = trace.finish(&[output.trace_value()]).unwrap();
    GraphCompiler::new().compile_traced_graph(&graph).unwrap()
}

/// Two different symbolic extents of `i` under a broadcast turned on by the
/// constant `j` (`1` vs `2`), with `i` bound to `0` and `1`. Singleton
/// broadcasting of an empty axis yields an empty axis, so the merged extent is
/// `0`, not `max(0, 1) = 1`. `zero_operand` picks which operand carries the
/// empty axis, so both fold orders of the merge are exercised. The compiled
/// output metadata (evaluated at the bound shapes) and the executed output
/// must both be `[0]`.
fn assert_empty_axis_broadcast_merges_to_zero(zero_operand: usize) {
    let compiled = compile_ij_ij_to_i_with_separate_symbolic_i();

    let (lhs_rows, rhs_rows) = if zero_operand == 0 { (0, 1) } else { (1, 0) };
    let lhs_tensor =
        Tensor::from_vec_col_major(vec![lhs_rows, 1], vec![2.0_f64; lhs_rows]).unwrap();
    let rhs_tensor =
        Tensor::from_vec_col_major(vec![rhs_rows, 2], vec![3.0_f64; rhs_rows * 2]).unwrap();
    let bound: [&[usize]; 2] = [&[lhs_rows, 1], &[rhs_rows, 2]];

    let output_extents: Vec<usize> = compiled
        .program()
        .value_metadata(compiled.program().outputs()[0])
        .unwrap()
        .shape()
        .iter()
        .map(|extent| extent.as_exact().unwrap().eval(&bound).unwrap())
        .collect();
    assert_eq!(output_extents, [0], "zero operand {zero_operand}");

    let result = support::run_one(&compiled, &[&lhs_tensor, &rhs_tensor]).unwrap();
    assert_eq!(result.shape(), &[0], "zero operand {zero_operand}");
}

#[test]
fn traced_einsum_broadcasts_an_empty_symbolic_axis_in_the_first_operand_to_zero() {
    assert_empty_axis_broadcast_merges_to_zero(0);
}

#[test]
fn traced_einsum_broadcasts_an_empty_symbolic_axis_in_the_second_operand_to_zero() {
    assert_empty_axis_broadcast_merges_to_zero(1);
}

/// The same graph as [`assert_empty_axis_broadcast_merges_to_zero`] bound to
/// an invalid pair: `i` is `2` in one operand and `3` in the other, neither a
/// broadcast singleton. The symbolic merge cannot see this at compile time, so
/// execution with the bound shapes must reject it as an einsum shape mismatch
/// instead of producing an output of extent `max(2, 3) = 3`. Both operand
/// orders are checked.
#[test]
fn traced_einsum_rejects_an_invalid_symbolic_binding_as_a_shape_mismatch() {
    for (lhs_rows, rhs_rows) in [(2, 3), (3, 2)] {
        let compiled = compile_ij_ij_to_i_with_separate_symbolic_i();
        let lhs_tensor =
            Tensor::from_vec_col_major(vec![lhs_rows, 1], vec![2.0_f64; lhs_rows]).unwrap();
        let rhs_tensor =
            Tensor::from_vec_col_major(vec![rhs_rows, 2], vec![3.0_f64; rhs_rows * 2]).unwrap();

        let error = support::run_one(&compiled, &[&lhs_tensor, &rhs_tensor])
            .expect_err("i bound to 2 and 3 is not a singleton broadcast");

        let RuntimeError::TensorRuntime(TensorError::Validation {
            op: "einsum",
            source: ValidationError::ShapeMismatch(mismatch),
        }) = &error
        else {
            panic!("expected an einsum shape mismatch for ({lhs_rows}, {rhs_rows}), got {error:?}");
        };
        let ShapeMismatch::ExpectedActual { expected, actual } = mismatch.as_ref() else {
            panic!("expected an expected/actual shape mismatch, got {mismatch:?}");
        };
        let mut extents = [expected[0], actual[0]];
        extents.sort_unstable();
        assert_eq!(extents, [2, 3], "({lhs_rows}, {rhs_rows}): {error}");
    }
}

#[test]
fn traced_einsum_rejects_a_diagonal_mismatch_while_another_label_broadcasts() {
    let trace = TraceContext::new();
    let diagonal = trace
        .derive_tensor(
            trace
                .input(ProgramInputSpec::new(
                    DType::F64,
                    [DimExpr::Const(1), DimExpr::Const(3)],
                ))
                .unwrap(),
        )
        .unwrap();
    let vector = trace
        .derive_tensor(
            trace
                .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(1)]))
                .unwrap(),
        )
        .unwrap();
    let output = trace.einsum(&[&diagonal, &vector], "ii,i->i").unwrap();
    let graph = trace.finish(&[output.trace_value()]).unwrap();

    assert!(GraphCompiler::new().compile_traced_graph(&graph).is_err());
}
