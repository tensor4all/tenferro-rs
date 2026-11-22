use std::sync::Arc;

use tenferro_ops::dim_expr::DimExpr;
use tenferro_tensor::DType;

use super::{Subprogram, SubprogramCall};
use crate::program::{
    CoreSemanticOp, FrozenProgram, ProgramImport, ProgramInputSpec, ProgramValueMetadata,
    SemanticProgramBuilder,
};
use crate::TraceContext;

fn square() -> Subprogram {
    TraceContext::record(&[(DType::F64, 1)], |_, p| Ok(vec![p[0].mul(&p[0])?]))
        .unwrap()
        .subprogram()
        .clone()
}

fn cube() -> Subprogram {
    TraceContext::record(&[(DType::F64, 1)], |_, p| {
        Ok(vec![p[0].mul(&p[0])?.mul(&p[0])?])
    })
    .unwrap()
    .subprogram()
    .clone()
}

/// Callee input metadata naming the carrying operation's own input 0.
fn local_meta() -> Box<[ProgramValueMetadata]> {
    Box::new([ProgramValueMetadata::new(
        DType::F64,
        [DimExpr::InputDim {
            input_idx: 0,
            axis: 0,
        }],
    )])
}

/// `neg(x)` over a length-3 input, the `neg` carrying `calls`.
fn outer_with(calls: Vec<SubprogramCall>) -> FrozenProgram {
    let mut builder = SemanticProgramBuilder::new();
    let x = builder
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(3)]))
        .unwrap();
    let y = builder.add_op(CoreSemanticOp::Neg, &[x]).unwrap()[0];
    builder.attach_subprograms_to_last_operation_for_test(calls);
    builder.finish(&[y]).unwrap()
}

fn call(program: Subprogram) -> SubprogramCall {
    SubprogramCall::new(program, local_meta())
}

fn assert_same(left: &FrozenProgram, right: &FrozenProgram) {
    assert_eq!(
        left.program.semantic_fingerprint(),
        right.program.semantic_fingerprint()
    );
    assert!(left.program.semantic_eq(&right.program));
}

fn assert_distinct(left: &FrozenProgram, right: &FrozenProgram) {
    assert_ne!(
        left.program.semantic_fingerprint(),
        right.program.semantic_fingerprint()
    );
    assert!(!left.program.semantic_eq(&right.program));
}

#[test]
fn outer_programs_that_differ_only_in_a_callee_are_distinct() {
    assert_distinct(
        &outer_with(vec![call(square())]),
        &outer_with(vec![call(cube())]),
    );
}

#[test]
fn equal_callees_from_separate_recordings_give_equal_outer_programs() {
    let first = square();
    let second = square();
    assert!(!std::ptr::eq(first.program(), second.program()));
    assert_same(
        &outer_with(vec![call(first)]),
        &outer_with(vec![call(second)]),
    );
}

#[test]
fn call_count_order_and_call_site_metadata_are_part_of_the_identity() {
    let none = outer_with(Vec::new());
    let one = outer_with(vec![call(square())]);
    assert_distinct(&none, &one);

    let square_then_cube = outer_with(vec![call(square()), call(cube())]);
    let cube_then_square = outer_with(vec![call(cube()), call(square())]);
    assert_distinct(&square_then_cube, &cube_then_square);

    let concrete_meta = outer_with(vec![SubprogramCall::new(
        square(),
        Box::new([ProgramValueMetadata::new(DType::F64, [DimExpr::Const(3)])]),
    )]);
    assert_distinct(&one, &concrete_meta);
}

#[test]
fn the_parameter_split_is_part_of_subprogram_identity() {
    let recorded = TraceContext::record(&[(DType::F64, 1)], |trace, p| {
        let w = trace.from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
        Ok(vec![p[0].mul(&w)?])
    })
    .unwrap();
    let as_recorded = recorded.subprogram().clone();
    let all_parameters = Subprogram::new(Arc::clone(&as_recorded.program), 2);
    assert_eq!(as_recorded.capture_count(), 1);
    assert_eq!(all_parameters.capture_count(), 0);
    assert_ne!(as_recorded, all_parameters);
    assert_distinct(
        &outer_with(vec![call(as_recorded)]),
        &outer_with(vec![call(all_parameters)]),
    );
}

#[test]
fn calls_survive_snapshot_and_import() {
    let mut builder = SemanticProgramBuilder::new();
    let x = builder
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(3)]))
        .unwrap();
    // A dead operation forces the snapshot to rebuild rather than copy.
    builder.add_op(CoreSemanticOp::Exp, &[x]).unwrap();
    let y = builder.add_op(CoreSemanticOp::Neg, &[x]).unwrap()[0];
    builder.attach_subprograms_to_last_operation_for_test(vec![call(square())]);
    let (snapshot, _) = builder.snapshot_pruned(&[y]).unwrap();
    let expected = outer_with(vec![call(square())]);
    assert_same(&snapshot, &expected);

    let mut destination = SemanticProgramBuilder::new();
    let imported = destination
        .import(ProgramImport {
            program: snapshot.program.as_ref(),
            bindings: None,
            roots: snapshot.program.outputs(),
            input_map: &[],
        })
        .unwrap();
    let roundtrip = destination.finish(imported.roots()).unwrap();
    assert_same(&roundtrip, &expected);
}

#[test]
fn retained_bytes_account_for_call_site_metadata() {
    let none = outer_with(Vec::new())
        .program
        .logical_retained_bytes()
        .unwrap();
    let one = outer_with(vec![call(square())])
        .program
        .logical_retained_bytes()
        .unwrap();
    assert!(one > none);
}

#[test]
fn subprogram_debug_is_a_summary() {
    let rendered = format!("{:?}", square());
    assert!(rendered.starts_with("Subprogram"));
    assert!(rendered.contains("parameters: 1"));
    assert!(rendered.contains("captures: 0"));
    assert!(rendered.contains("operations: 1"));
}
