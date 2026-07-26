use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use tenferro_ad::AdContext;
use tenferro_cpu::CpuBackend;
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::SymDim;
use tenferro_runtime::program::{ProgramInputSpec, SemanticProgramBuilder};
use tenferro_runtime::{GraphCompiler, Runtime};
use tenferro_tensor::{
    core::DType as CoreDType, DType, Error, ErrorKind, ShapeMismatch, Tensor, ValidationError,
    ValidationKind,
};

use super::{
    execute_sparse_reference_payload, extension_modules, sparse_semantic_ad_rules,
    SparseMatmulJvpOp, SparseMatmulOp, SparseMatmulPlan, SparseMatmulVjpOp, JVP_FAMILY_ID,
    VJP_FAMILY_ID,
};

fn plan() -> SparseMatmulPlan {
    SparseMatmulPlan::new(
        &[2, 2],
        &[[0, 0], [0, 1], [1, 0]],
        &[2, 2],
        &[[0, 0], [1, 0], [0, 1]],
    )
    .unwrap()
}

fn f64_values(len: usize) -> Tensor {
    Tensor::from_vec_col_major(vec![len], vec![1.0_f64; len]).unwrap()
}

fn semantic_sparse_program() -> tenferro_runtime::program::FrozenProgram {
    let mut builder = SemanticProgramBuilder::new();
    let lhs = builder
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(3)]))
        .unwrap();
    let rhs = builder
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(3)]))
        .unwrap();
    let output = builder
        .add_extension(Arc::new(SparseMatmulOp { plan: plan() }), &[lhs, rhs])
        .unwrap()[0];
    builder.finish(&[output]).unwrap()
}

fn cpu_runtime_with_sparse() -> Runtime {
    let backend = CpuBackend::new();
    let mut builder = Runtime::builder();
    builder
        .register_engine(tenferro_cpu::runtime_engine_registration(&backend).unwrap())
        .unwrap();
    for module in
        extension_modules::<CpuBackend>(tenferro_cpu::runtime_engine_id().unwrap()).unwrap()
    {
        builder.install_extension_module(module).unwrap();
    }
    builder.build().unwrap()
}

fn run_one(
    runtime: &Runtime,
    program: &tenferro_runtime::CompiledGraph,
    inputs: &[&Tensor],
) -> tenferro_runtime::Result<Tensor> {
    let mut outputs = runtime.run_compiled(program, inputs)?;
    assert_eq!(outputs.len(), 1);
    Ok(outputs.remove(0))
}

fn host_error(result: std::thread::Result<tenferro_tensor::Result<Vec<Tensor>>>) -> Error {
    result
        .expect("host-reference validation must not panic")
        .expect_err("malformed host-reference inputs must be rejected")
}

fn assert_invalid_argument(error: &Error) {
    assert_eq!(
        error.kind(),
        ErrorKind::Validation(ValidationKind::InvalidArgument)
    );
    assert!(matches!(
        error,
        Error::Validation {
            op: "tenferro-ext-sparse",
            source: ValidationError::InvalidArgument {
                argument: "configuration",
                ..
            },
        }
    ));
    assert!(std::error::Error::source(error).is_some());
}

fn assert_dtype_mismatch(error: &Error, expected: CoreDType, actual: CoreDType) {
    assert_eq!(
        error.kind(),
        ErrorKind::Validation(ValidationKind::DTypeMismatch)
    );
    assert!(matches!(
        error,
        Error::Validation {
            op: "tenferro-ext-sparse",
            source: ValidationError::DTypeMismatch {
                expected: actual_expected,
                actual: actual_actual,
            },
        } if *actual_expected == expected && *actual_actual == actual
    ));
    assert!(std::error::Error::source(error).is_some());
}

fn assert_shape_mismatch(error: &Error, expected: &[usize], actual: &[usize]) {
    assert_eq!(
        error.kind(),
        ErrorKind::Validation(ValidationKind::ShapeMismatch)
    );
    assert!(matches!(
        error,
        Error::Validation {
            op: "tenferro-ext-sparse",
            source: ValidationError::ShapeMismatch(payload),
        } if matches!(
            payload.as_ref(),
            ShapeMismatch::IncompatibleShapes { lhs, rhs }
                if lhs.as_slice() == expected && rhs.as_slice() == actual
        )
    ));
    let validation_source =
        std::error::Error::source(error).expect("shape mismatch must retain its validation source");
    assert!(
        std::error::Error::source(validation_source).is_some(),
        "shape mismatch must retain its typed payload source"
    );
}

#[test]
fn sparse_jvp_host_boundary_rejects_count_dtype_and_exact_shape() {
    let plan = plan();
    let op = SparseMatmulJvpOp {
        plan,
        active_inputs: vec![0, 1],
    };
    let lhs = f64_values(3);
    let rhs = f64_values(3);
    let valid_lhs_tangent = f64_values(3);
    let wrong_dtype = Tensor::from_vec_col_major(vec![3], vec![1_i64; 3]).unwrap();
    let wrong_shape = f64_values(4);

    let error = host_error(catch_unwind(AssertUnwindSafe(|| {
        execute_sparse_reference_payload(JVP_FAMILY_ID, &op, &[&lhs])
    })));
    assert_invalid_argument(&error);

    let error = host_error(catch_unwind(AssertUnwindSafe(|| {
        execute_sparse_reference_payload(
            JVP_FAMILY_ID,
            &op,
            &[&lhs, &rhs, &valid_lhs_tangent, &wrong_dtype],
        )
    })));
    assert_dtype_mismatch(&error, CoreDType::F64, CoreDType::I64);

    let error = host_error(catch_unwind(AssertUnwindSafe(|| {
        execute_sparse_reference_payload(
            JVP_FAMILY_ID,
            &op,
            &[&lhs, &rhs, &valid_lhs_tangent, &wrong_shape],
        )
    })));
    assert_shape_mismatch(&error, &[3], &[4]);
}

#[test]
fn sparse_vjp_host_boundary_rejects_count_dtype_and_exact_shape() {
    let plan = plan();
    let output_nnz = plan.output_nnz();
    let op = SparseMatmulVjpOp {
        plan,
        active_input: 0,
    };
    let lhs = f64_values(3);
    let rhs = f64_values(3);
    let wrong_dtype =
        Tensor::from_vec_col_major(vec![output_nnz], vec![1_i64; output_nnz]).unwrap();
    let wrong_shape = f64_values(output_nnz + 1);

    let error = host_error(catch_unwind(AssertUnwindSafe(|| {
        execute_sparse_reference_payload(VJP_FAMILY_ID, &op, &[&lhs, &rhs])
    })));
    assert_invalid_argument(&error);

    let error = host_error(catch_unwind(AssertUnwindSafe(|| {
        execute_sparse_reference_payload(VJP_FAMILY_ID, &op, &[&lhs, &rhs, &wrong_dtype])
    })));
    assert_dtype_mismatch(&error, CoreDType::F64, CoreDType::I64);

    let error = host_error(catch_unwind(AssertUnwindSafe(|| {
        execute_sparse_reference_payload(VJP_FAMILY_ID, &op, &[&lhs, &rhs, &wrong_shape])
    })));
    assert_shape_mismatch(&error, &[output_nnz], &[output_nnz + 1]);
}

#[test]
fn sparse_metadata_validation_preserves_dtype_and_rank_payloads() {
    let shape = [SymDim::from(3_usize)];
    let dtype_error =
        super::validate_primal_meta(&[DType::I64, DType::F64], &[&shape[..], &shape[..]])
            .unwrap_err();
    assert_dtype_mismatch(&dtype_error, CoreDType::F64, CoreDType::I64);

    let rank_shape = [SymDim::from(3_usize), SymDim::from(1_usize)];
    let rank_error =
        super::validate_primal_meta(&[DType::F64, DType::F64], &[&rank_shape[..], &shape[..]])
            .unwrap_err();
    assert_eq!(
        rank_error.kind(),
        ErrorKind::Validation(ValidationKind::RankMismatch)
    );
    assert!(matches!(
        rank_error,
        Error::Validation {
            op: "tenferro-ext-sparse",
            source: ValidationError::RankMismatch {
                expected: 1,
                actual: 2,
            },
        }
    ));
}

#[test]
fn sparse_semantic_rules_execute_jvp_and_vjp_numerically() {
    let source = semantic_sparse_program();
    let ad = AdContext::builder()
        .with_semantic_extension_rules(
            sparse_semantic_ad_rules().expect("sparse semantic AD rules"),
        )
        .unwrap()
        .build()
        .unwrap();
    let lhs = Tensor::from_vec_col_major(vec![3], vec![2.0_f64, 1.0, 3.0]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![3], vec![10.0_f64, 70.0, 20.0]).unwrap();
    let lhs_tangent = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 0.0, 0.0]).unwrap();
    let rhs_tangent = Tensor::from_vec_col_major(vec![3], vec![0.0_f64, 0.0, 1.0]).unwrap();

    let jvp = ad.jvp_program(&source, &[true, true]).unwrap();
    let compiled = GraphCompiler::new()
        .compile_frozen_program(jvp.frozen())
        .unwrap();
    let runtime = cpu_runtime_with_sparse();
    let tangent = run_one(
        &runtime,
        &compiled,
        &[&lhs, &rhs, &lhs_tangent, &rhs_tangent],
    )
    .unwrap();
    assert_eq!(tangent.as_slice::<f64>().unwrap(), &[10.0, 0.0, 22.0, 3.0]);

    let output_cotangent =
        Tensor::from_vec_col_major(vec![4], vec![1.0_f64, 1.0, 1.0, 1.0]).unwrap();
    let vjp = ad.vjp_program(&source, &[true, true], &[true]).unwrap();
    let compiled = GraphCompiler::new()
        .compile_frozen_program(vjp.frozen())
        .unwrap();
    let cotangents = runtime
        .run_compiled(&compiled, &[&lhs, &rhs, &output_cotangent])
        .unwrap();
    assert_eq!(
        cotangents[0].as_slice::<f64>().unwrap(),
        &[30.0, 70.0, 30.0]
    );
    assert_eq!(cotangents[1].as_slice::<f64>().unwrap(), &[5.0, 1.0, 5.0]);
}
