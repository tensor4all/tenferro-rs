#![cfg(feature = "autodiff")]

use tenferro_ad::AdContext;
use tenferro_cpu::CpuBackend;
use tenferro_ext_sparse::{
    register_runtime, sparse_matmul, sparse_matmul_eager, sparse_semantic_ad_rules, SparseCooTensor,
    SparseCooTracedTensor,
};
use tenferro_runtime::{Error, GraphCompiler, GraphExecutor, TracedTensor};
use tenferro_tensor::Tensor;

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn assert_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (idx, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let error = (actual - expected).abs();
        assert!(
            error < 1.0e-12,
            "idx {idx}: actual={actual}, expected={expected}, error={error}"
        );
    }
}

fn coordinates(entries: &[[i64; 2]]) -> Tensor {
    let mut data = Vec::with_capacity(entries.len() * 2);
    for [row, col] in entries {
        data.push(*row);
        data.push(*col);
    }
    Tensor::from_vec_col_major(vec![2, entries.len()], data).unwrap()
}

fn left_sparse_values(values: &[f64]) -> Tensor {
    Tensor::from_vec_col_major(vec![values.len()], values.to_vec()).unwrap()
}

fn left_sparse(values: &[f64]) -> SparseCooTensor {
    SparseCooTensor::from_parts(
        vec![2, 2],
        coordinates(&[[0, 0], [0, 1], [1, 0]]),
        left_sparse_values(values),
    )
    .unwrap()
}

fn right_sparse(values: &[f64]) -> SparseCooTensor {
    SparseCooTensor::from_parts(
        vec![2, 2],
        coordinates(&[[0, 0], [1, 0], [0, 1]]),
        Tensor::from_vec_col_major(vec![values.len()], values.to_vec()).unwrap(),
    )
    .unwrap()
}

fn left_traced(values: &[f64]) -> SparseCooTracedTensor {
    SparseCooTracedTensor::from_parts(
        vec![2, 2],
        coordinates(&[[0, 0], [0, 1], [1, 0]]),
        TracedTensor::from_vec_col_major(vec![values.len()], values.to_vec()).unwrap(),
    )
    .unwrap()
}

fn right_traced(values: &[f64]) -> SparseCooTracedTensor {
    SparseCooTracedTensor::from_parts(
        vec![2, 2],
        coordinates(&[[0, 0], [1, 0], [0, 1]]),
        TracedTensor::from_vec_col_major(vec![values.len()], values.to_vec()).unwrap(),
    )
    .unwrap()
}

fn left_symbolic(values: TracedTensor) -> SparseCooTracedTensor {
    SparseCooTracedTensor::from_parts(vec![2, 2], coordinates(&[[0, 0], [0, 1], [1, 0]]), values)
        .unwrap()
}

fn right_symbolic(values: TracedTensor) -> SparseCooTracedTensor {
    SparseCooTracedTensor::from_parts(vec![2, 2], coordinates(&[[0, 0], [1, 0], [0, 1]]), values)
        .unwrap()
}

fn symbolic_values(len: usize) -> TracedTensor {
    TracedTensor::from_tensor_symbolic_shape(
        Tensor::from_vec_col_major(vec![len], vec![1.0_f64; len]).unwrap(),
    )
    .unwrap()
}

fn run_values(values: &TracedTensor) -> Tensor {
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(values).expect("compile sparse graph");
    let mut executor = GraphExecutor::new(CpuBackend::new());
    executor
        .register_extension(register_runtime)
        .expect("register sparse runtime");
    executor.run(&program).expect("run sparse graph")
}

fn sparse_ad() -> AdContext {
    AdContext::builder()
        .with_semantic_extension_rules(
            sparse_semantic_ad_rules().expect("sparse semantic AD rules"),
        )
        .unwrap()
        .build()
        .expect("AD context")
}

#[test]
fn eager_sparse_matmul_matches_dense_reference() -> TestResult {
    let lhs = left_sparse(&[2.0, 1.0, 3.0]);
    let rhs = right_sparse(&[10.0, 70.0, 20.0]);

    let out = sparse_matmul_eager(&lhs, &rhs)?;

    assert_eq!(out.shape(), &[2, 2]);
    assert_eq!(
        out.coordinates().as_slice::<i64>()?,
        &[0, 0, 1, 0, 0, 1, 1, 1]
    );
    assert_close(out.values().as_slice::<f64>()?, &[90.0, 30.0, 40.0, 60.0]);
    Ok(())
}

#[test]
fn traced_sparse_matmul_executes_through_extension_runtime() -> TestResult {
    let lhs = left_traced(&[2.0, 1.0, 3.0]);
    let rhs = right_traced(&[10.0, 70.0, 20.0]);

    let out = sparse_matmul(&lhs, &rhs)?;
    let values = run_values(out.values());

    assert_close(values.as_slice::<f64>()?, &[90.0, 30.0, 40.0, 60.0]);
    Ok(())
}

#[test]
fn sparse_matmul_sum_gradients_match_dense_reference() -> TestResult {
    let lhs = left_traced(&[2.0, 1.0, 3.0]);
    let rhs = right_traced(&[10.0, 70.0, 20.0]);
    let out = sparse_matmul(&lhs, &rhs)?;
    let loss = out.values().reduce_sum(Some(&[0]))?;
    let ad = sparse_ad();

    let grad_lhs = ad.grad(&loss, lhs.values())?;
    let grad_rhs = ad.grad(&loss, rhs.values())?;

    assert_close(
        run_values(&grad_lhs).as_slice::<f64>()?,
        &[30.0, 70.0, 30.0],
    );
    assert_close(run_values(&grad_rhs).as_slice::<f64>()?, &[5.0, 1.0, 5.0]);
    Ok(())
}

#[test]
fn tangent_shape_constraint_rejects_independent_sparse_tangent_mismatch() -> TestResult {
    let lhs_values = symbolic_values(3);
    let rhs_values = symbolic_values(3);
    let tangent = symbolic_values(3);
    assert_ne!(
        lhs_values.axis_sym_dim(0)?,
        tangent.axis_sym_dim(0)?,
        "primal and tangent must have independent symbolic origins"
    );
    let lhs = left_symbolic(lhs_values.clone());
    let rhs = right_symbolic(rhs_values.clone());
    let output = sparse_matmul(&lhs, &rhs)?;
    let jvp = sparse_ad().jvp(output.values(), lhs.values(), &tangent)?;
    let mut compiler = GraphCompiler::new();
    compiler.compile(&jvp)?;

    let mismatched_tangent = symbolic_values(4);
    let mismatched_jvp = sparse_ad().jvp(output.values(), lhs.values(), &mismatched_tangent)?;
    let error = GraphCompiler::new()
        .compile(&mismatched_jvp)
        .expect_err("mismatched sparse tangent axis must fail during compilation");
    assert!(matches!(
        error,
        Error::ShapeConstraintViolation {
            family: "tenferro-ext-sparse.matmul_jvp.v1",
            ..
        }
    ));
    Ok(())
}

#[test]
fn primal_payload_shape_constraint_rejects_symbolic_nnz_mismatch() -> TestResult {
    let lhs_values = symbolic_values(4);
    let rhs_values = symbolic_values(3);
    let lhs = left_symbolic(lhs_values);
    let rhs = right_symbolic(rhs_values);
    let output = sparse_matmul(&lhs, &rhs)?;

    let error = GraphCompiler::new()
        .compile(output.values())
        .expect_err("payload nnz mismatch must fail during compilation");
    assert!(matches!(
        error,
        Error::ShapeConstraintViolation {
            family: "tenferro-ext-sparse.matmul.v1",
            ..
        }
    ));
    Ok(())
}

#[test]
fn cotangent_shape_constraint_rejects_sparse_vjp_output_nnz_mismatch() -> TestResult {
    let lhs_values = symbolic_values(3);
    let rhs_values = symbolic_values(3);
    let lhs = left_symbolic(lhs_values);
    let rhs = right_symbolic(rhs_values);
    let output = sparse_matmul(&lhs, &rhs)?;
    let cotangent = symbolic_values(5);
    assert_ne!(
        output.values().axis_sym_dim(0)?,
        cotangent.axis_sym_dim(0)?,
        "output and cotangent must have independent symbolic origins"
    );
    let vjp = sparse_ad().vjp(output.values(), lhs.values(), &cotangent)?;

    let error = GraphCompiler::new()
        .compile(&vjp)
        .expect_err("mismatched sparse cotangent axis must fail during compilation");
    assert!(matches!(
        error,
        Error::ShapeConstraintViolation {
            family: "tenferro-ext-sparse.matmul_vjp.v1",
            ..
        }
    ));
    Ok(())
}
