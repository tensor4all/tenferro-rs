use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use tenferro_cpu::CpuEngine;
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::ShapeExtent;

use crate::compiler::semantic_staging::stage_semantic_program;
use crate::program::{ProgramImport, Subprogram};
use crate::{
    DType, Error, ErrorPhase, GraphCompiler, RecordedProgram, Tensor, TraceContext, TracedTensor,
};

fn hash_of(program: &Subprogram) -> u64 {
    let mut hasher = DefaultHasher::new();
    program.hash(&mut hasher);
    hasher.finish()
}

/// `x * w + x` with a captured weight `w` of length 2.
fn scaled_shift(weight: [f64; 2]) -> RecordedProgram {
    TraceContext::record(&[(DType::F64, 1)], |trace, p| {
        let w = trace.from_vec_col_major(vec![2], weight.to_vec())?;
        Ok(vec![p[0].mul(&w)?.add(&p[0])?])
    })
    .unwrap()
}

/// Inline a recording into a fresh trace on concrete arguments and run it.
///
/// Execution of a carried sub-program arrives with #2691 step 2; inlining runs
/// the recorded structure through today's compile/prepare/run path, which is
/// what proves the recording is a correct shape-generic program.
fn inline_and_run(recorded: &RecordedProgram, arguments: Vec<Tensor>) -> Vec<f64> {
    let program = recorded.subprogram().program();
    let captures = recorded.captures().map(|capture| {
        let data = capture
            .tensor_read()
            .unwrap()
            .as_slice::<f64>()
            .unwrap()
            .to_vec();
        Tensor::from_vec_col_major(capture.shape().to_vec(), data).unwrap()
    });
    let operands: Vec<Tensor> = arguments.into_iter().chain(captures).collect();
    assert_eq!(operands.len(), program.inputs().len());

    // Every operand enters as a concrete-shape placeholder, in program input order.
    let trace = TraceContext::new();
    let mut input_map = Vec::new();
    for (input, operand) in program.inputs().iter().zip(&operands) {
        let leaf = trace
            .input_concrete_shape(operand.dtype(), operand.shape())
            .unwrap();
        input_map.push((*input, leaf.trace_value().program_value()));
    }
    let roots = trace
        .import_values(ProgramImport {
            program,
            bindings: None,
            roots: program.outputs(),
            input_map: &input_map,
        })
        .unwrap();
    let output: TracedTensor = trace.derive_tensor(roots[0]).unwrap();

    let compiled = GraphCompiler::new().compile(&output).unwrap();
    let staged = stage_semantic_program(compiled.program(), compiled.compiler_options()).unwrap();
    let mut engine = CpuEngine::serial();
    let tensor = crate::exec::run_exec_program_for_test(&mut engine, &staged, operands)
        .unwrap()
        .pop()
        .unwrap();
    tensor.as_slice::<f64>().unwrap().to_vec()
}

#[test]
fn recordings_of_one_closure_are_equal_and_others_are_not() {
    let square =
        || TraceContext::record(&[(DType::F64, 1)], |_, p| Ok(vec![p[0].mul(&p[0])?])).unwrap();
    let first = square();
    let second = square();
    assert!(!std::ptr::eq(
        first.subprogram().program(),
        second.subprogram().program()
    ));
    assert_eq!(first.subprogram(), second.subprogram());
    assert_eq!(
        first.subprogram().semantic_fingerprint(),
        second.subprogram().semantic_fingerprint()
    );
    assert_eq!(hash_of(first.subprogram()), hash_of(second.subprogram()));

    let cube = TraceContext::record(&[(DType::F64, 1)], |_, p| {
        Ok(vec![p[0].mul(&p[0])?.mul(&p[0])?])
    })
    .unwrap();
    assert_ne!(first.subprogram(), cube.subprogram());
    assert_ne!(
        first.subprogram().semantic_fingerprint(),
        cube.subprogram().semantic_fingerprint()
    );

    // The same body over a different parameter declaration is a different program.
    let square_f32 =
        TraceContext::record(&[(DType::F32, 1)], |_, p| Ok(vec![p[0].mul(&p[0])?])).unwrap();
    assert_ne!(first.subprogram(), square_f32.subprogram());
}

#[test]
fn captured_data_is_hoisted_out_of_the_identity() {
    let first = scaled_shift([2.0, 3.0]);
    let second = scaled_shift([5.0, 7.0]);

    assert_eq!(first.subprogram().parameter_count(), 1);
    assert_eq!(first.subprogram().capture_count(), 1);
    assert_eq!(first.subprogram().program().inputs().len(), 2);
    assert_eq!(first.subprogram(), second.subprogram());

    let captured: Vec<Vec<f64>> = [&first, &second]
        .iter()
        .map(|recorded| {
            let capture = recorded.captures().next().unwrap();
            capture
                .tensor_read()
                .unwrap()
                .as_slice::<f64>()
                .unwrap()
                .to_vec()
        })
        .collect();
    assert_eq!(captured, [vec![2.0, 3.0], vec![5.0, 7.0]]);

    let argument = || Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 10.0]).unwrap();
    assert_eq!(inline_and_run(&first, vec![argument()]), [3.0, 40.0]);
    assert_eq!(inline_and_run(&second, vec![argument()]), [6.0, 80.0]);
}

#[test]
fn a_capture_with_another_shape_is_another_program() {
    let wide = TraceContext::record(&[(DType::F64, 1)], |trace, p| {
        let w = trace.from_vec_col_major(vec![3], vec![1.0_f64; 3])?;
        Ok(vec![p[0].mul(&w)?.add(&p[0])?])
    })
    .unwrap();
    assert_ne!(wide.subprogram(), scaled_shift([1.0, 1.0]).subprogram());
}

#[test]
fn one_recording_runs_at_two_sizes() {
    let recorded = TraceContext::record(&[(DType::F64, 1), (DType::F64, 1)], |_, p| {
        Ok(vec![p[0].mul(&p[1])?.add(&p[0])?.exp()?])
    })
    .unwrap();
    for n in [2_usize, 5] {
        let x: Vec<f64> = (0..n).map(|i| 0.1 * i as f64).collect();
        let y: Vec<f64> = (0..n).map(|i| 1.0 - 0.2 * i as f64).collect();
        let expected: Vec<f64> = x.iter().zip(&y).map(|(x, y)| (x * y + x).exp()).collect();
        let actual = inline_and_run(
            &recorded,
            vec![
                Tensor::from_vec_col_major(vec![n], x).unwrap(),
                Tensor::from_vec_col_major(vec![n], y).unwrap(),
            ],
        );
        assert_eq!(actual.len(), n);
        for (actual, expected) in actual.iter().zip(&expected) {
            assert!((actual - expected).abs() <= 1e-15 * expected.abs());
        }
    }
}

#[test]
fn parameters_are_symbolic_inside_the_closure() {
    let mut observed = None;
    TraceContext::record(&[(DType::F64, 2)], |_, p| {
        observed = Some((p[0].is_concrete_shape(), p[0].concrete_shape().is_err()));
        Ok(vec![p[0].clone()])
    })
    .unwrap();
    assert_eq!(observed, Some((false, true)));
}

#[test]
fn every_declared_parameter_is_kept_in_order() {
    let recorded = TraceContext::record(
        &[(DType::F64, 1), (DType::F32, 2), (DType::I64, 0)],
        |_, p| Ok(vec![p[1].neg()?]),
    )
    .unwrap();
    let program = recorded.subprogram().program();
    let declared: Vec<(DType, usize)> = program
        .inputs()
        .iter()
        .map(|input| {
            let metadata = program.value_metadata(*input).unwrap();
            (metadata.dtype(), metadata.shape().len())
        })
        .collect();
    assert_eq!(
        declared,
        [(DType::F64, 1), (DType::F32, 2), (DType::I64, 0)]
    );
    // Each parameter's extents name its own input position, so a later
    // specialization resolves every axis against the right argument.
    for (position, input) in program.inputs().iter().enumerate() {
        let metadata = program.value_metadata(*input).unwrap();
        for (axis, extent) in metadata.shape().iter().enumerate() {
            assert_eq!(
                *extent,
                ShapeExtent::Exact(DimExpr::InputDim {
                    input_idx: position,
                    axis
                })
            );
        }
    }
    // The output of `neg(p[1])` keeps p[1]'s symbolic extents (input 1), not
    // input 0's: the unused leading parameter must not shift the numbering.
    let output = program.value_metadata(program.outputs()[0]).unwrap();
    assert_eq!(
        output.shape(),
        [
            ShapeExtent::Exact(DimExpr::InputDim {
                input_idx: 1,
                axis: 0
            }),
            ShapeExtent::Exact(DimExpr::InputDim {
                input_idx: 1,
                axis: 1
            }),
        ]
    );
    assert_eq!(recorded.subprogram().parameter_count(), 3);
    assert_eq!(recorded.subprogram().capture_count(), 0);
}

#[test]
fn a_parameter_may_be_returned_directly() {
    let recorded = TraceContext::record(&[(DType::F64, 1)], |_, p| {
        Ok(vec![p[0].clone(), p[0].clone()])
    })
    .unwrap();
    let program = recorded.subprogram().program();
    assert_eq!(program.operations().count(), 0);
    assert_eq!(
        program.outputs(),
        [program.inputs()[0], program.inputs()[0]]
    );
}

fn assert_record_invalid_argument(error: Error, expected_argument: &str) {
    match error {
        Error::Validation {
            phase: ErrorPhase::GraphBuild,
            source: tenferro_tensor::ValidationError::InvalidArgument { argument, .. },
            ..
        } => assert_eq!(argument, expected_argument),
        other => panic!("expected an InvalidArgument for {expected_argument}, got {other:?}"),
    }
}

#[test]
fn an_outer_trace_value_cannot_enter_a_recording() {
    let outer = TraceContext::new();
    let captured = outer.from_vec_col_major(vec![1], vec![1.0_f64]).unwrap();

    let combined = TraceContext::record(&[(DType::F64, 1)], |_, p| Ok(vec![p[0].add(&captured)?]))
        .unwrap_err();
    assert_record_invalid_argument(combined, "context");

    let returned =
        TraceContext::record(&[(DType::F64, 1)], |_, _| Ok(vec![captured.clone()])).unwrap_err();
    assert_record_invalid_argument(returned, "outputs");
}

#[test]
fn an_unused_undeclared_placeholder_is_pruned() {
    let recorded = TraceContext::record(&[(DType::F64, 1)], |trace, p| {
        trace.input_symbolic_shape(DType::F64, 1)?;
        Ok(vec![p[0].neg()?])
    })
    .unwrap();
    assert_eq!(recorded.subprogram().program().inputs().len(), 1);
    assert_eq!(recorded.subprogram().capture_count(), 0);
}

#[test]
fn an_undeclared_placeholder_is_rejected() {
    let error = TraceContext::record(&[(DType::F64, 1)], |trace, p| {
        let extra = trace.input_symbolic_shape(DType::F64, 1)?;
        Ok(vec![p[0].add(&extra)?])
    })
    .unwrap_err();
    assert_record_invalid_argument(error, "f");
}

#[test]
fn a_closure_error_is_returned_unchanged() {
    let error = TraceContext::record(&[(DType::F64, 1)], |_, _| {
        Err(Error::invalid_argument(
            "user",
            ErrorPhase::GraphBuild,
            "marker",
            "closure failed",
        ))
    })
    .unwrap_err();
    assert_record_invalid_argument(error, "marker");
}

#[test]
fn debug_output_is_a_summary() {
    let recorded = scaled_shift([1.0, 2.0]);
    let rendered = format!("{recorded:?}");
    assert!(rendered.starts_with("RecordedProgram"));
    assert!(rendered.contains("parameters: 1"));
    assert!(rendered.contains("captures: 1"));
    assert!(!rendered.contains("2.0"));
}
