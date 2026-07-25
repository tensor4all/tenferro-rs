use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use tenferro_ad::AdContext;
use tenferro_cpu::CpuBackend;
use tenferro_einsum::Subscripts;
use tenferro_ops::dim_expr::DimExpr;
use tenferro_runtime::program::{ProgramInputSpec, SemanticProgramBuilder};
use tenferro_runtime::{GraphCompiler, Runtime};
use tenferro_tensor::{
    core::DType as CoreDType, DType, Error, ErrorKind, ShapeMismatch, Tensor, ValidationError,
    ValidationKind,
};

use super::{
    execute_tropical_reference_payload, extension_modules, tropical_semantic_ad_rules,
    TropicalEinsumJvpOp, TropicalEinsumOp, TropicalEinsumVjpOp, TROPICAL_EINSUM_JVP_FAMILY_ID,
    TROPICAL_EINSUM_VJP_FAMILY_ID,
};
use crate::TropicalKind;

fn matrix(shape: Vec<usize>) -> Tensor {
    let len = shape.iter().product();
    Tensor::from_vec_col_major(shape, vec![1.0_f64; len]).unwrap()
}

fn semantic_tropical_program() -> tenferro_runtime::program::FrozenProgram {
    let mut builder = SemanticProgramBuilder::new();
    let lhs = builder
        .input(ProgramInputSpec::new(
            DType::F64,
            [DimExpr::Const(2), DimExpr::Const(2)],
        ))
        .unwrap();
    let rhs = builder
        .input(ProgramInputSpec::new(
            DType::F64,
            [DimExpr::Const(2), DimExpr::Const(2)],
        ))
        .unwrap();
    let output = builder
        .add_extension(
            Arc::new(TropicalEinsumOp::new(
                TropicalKind::MaxPlus,
                Subscripts::parse("ij,jk->ik").unwrap(),
            )),
            &[lhs, rhs],
        )
        .unwrap()[0];
    builder.finish(&[output]).unwrap()
}

fn cpu_runtime_with_tropical() -> Runtime {
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

fn assert_invalid_argument(error: &Error, op: &str) {
    assert_eq!(
        error.kind(),
        ErrorKind::Validation(ValidationKind::InvalidArgument)
    );
    assert!(matches!(
        error,
        Error::Validation {
            op: actual_op,
            source: ValidationError::InvalidArgument {
                argument: "configuration",
                ..
            },
        } if *actual_op == op
    ));
    assert!(std::error::Error::source(error).is_some());
}

fn assert_dtype_mismatch(error: &Error, op: &str, expected: CoreDType, actual: CoreDType) {
    assert_eq!(
        error.kind(),
        ErrorKind::Validation(ValidationKind::DTypeMismatch)
    );
    assert!(matches!(
        error,
        Error::Validation {
            op: actual_op,
            source: ValidationError::DTypeMismatch {
                expected: actual_expected,
                actual: actual_actual,
            },
        } if *actual_op == op && *actual_expected == expected && *actual_actual == actual
    ));
    assert!(std::error::Error::source(error).is_some());
}

fn assert_shape_mismatch(error: &Error, op: &str, expected: &[usize], actual: &[usize]) {
    assert_eq!(
        error.kind(),
        ErrorKind::Validation(ValidationKind::ShapeMismatch)
    );
    assert!(matches!(
        error,
        Error::Validation {
            op: actual_op,
            source: ValidationError::ShapeMismatch(payload),
        } if *actual_op == op && matches!(
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
fn tropical_jvp_host_boundary_rejects_count_dtype_and_exact_shape() {
    let op = TropicalEinsumJvpOp::new(
        TropicalKind::MaxPlus,
        Subscripts::parse("ij,jk->ik").unwrap(),
        vec![0, 1],
    );
    let lhs = matrix(vec![2, 3]);
    let rhs = matrix(vec![3, 2]);
    let valid_lhs_tangent = matrix(vec![2, 3]);
    let wrong_dtype = Tensor::from_vec_col_major(vec![3, 2], vec![1_i64; 6]).unwrap();
    let wrong_shape = matrix(vec![2, 3]);

    let error = host_error(catch_unwind(AssertUnwindSafe(|| {
        execute_tropical_reference_payload(TROPICAL_EINSUM_JVP_FAMILY_ID, &op, &[&lhs])
    })));
    assert_invalid_argument(&error, "tropical_einsum_jvp");

    let error = host_error(catch_unwind(AssertUnwindSafe(|| {
        execute_tropical_reference_payload(
            TROPICAL_EINSUM_JVP_FAMILY_ID,
            &op,
            &[&lhs, &rhs, &valid_lhs_tangent, &wrong_dtype],
        )
    })));
    assert_dtype_mismatch(
        &error,
        "tropical_einsum_jvp",
        CoreDType::F64,
        CoreDType::I64,
    );

    let error = host_error(catch_unwind(AssertUnwindSafe(|| {
        execute_tropical_reference_payload(
            TROPICAL_EINSUM_JVP_FAMILY_ID,
            &op,
            &[&lhs, &rhs, &valid_lhs_tangent, &wrong_shape],
        )
    })));
    assert_shape_mismatch(&error, "tropical_einsum_jvp", &[3, 2], &[2, 3]);
}

#[test]
fn tropical_vjp_host_boundary_rejects_count_dtype_and_exact_shape() {
    let op = TropicalEinsumVjpOp::new(
        TropicalKind::MaxPlus,
        Subscripts::parse("ij,jk->ik").unwrap(),
        0,
    );
    let lhs = matrix(vec![2, 3]);
    let rhs = matrix(vec![3, 2]);
    let wrong_dtype = Tensor::from_vec_col_major(vec![2, 2], vec![1_i64; 4]).unwrap();
    let wrong_shape = matrix(vec![4]);

    let error = host_error(catch_unwind(AssertUnwindSafe(|| {
        execute_tropical_reference_payload(TROPICAL_EINSUM_VJP_FAMILY_ID, &op, &[&lhs, &rhs])
    })));
    assert_invalid_argument(&error, "tropical_einsum_vjp");

    let error = host_error(catch_unwind(AssertUnwindSafe(|| {
        execute_tropical_reference_payload(
            TROPICAL_EINSUM_VJP_FAMILY_ID,
            &op,
            &[&lhs, &rhs, &wrong_dtype],
        )
    })));
    assert_dtype_mismatch(
        &error,
        "tropical_einsum_vjp",
        CoreDType::F64,
        CoreDType::I64,
    );

    let error = host_error(catch_unwind(AssertUnwindSafe(|| {
        execute_tropical_reference_payload(
            TROPICAL_EINSUM_VJP_FAMILY_ID,
            &op,
            &[&lhs, &rhs, &wrong_shape],
        )
    })));
    assert_shape_mismatch(&error, "tropical_einsum_vjp", &[2, 2], &[4]);
}

#[test]
fn tropical_semantic_rules_execute_jvp_and_vjp_numerically() {
    let source = semantic_tropical_program();
    let ad = AdContext::builder()
        .with_semantic_extension_rules(
            tropical_semantic_ad_rules().expect("tropical semantic AD rules"),
        )
        .unwrap()
        .build()
        .unwrap();
    let lhs = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 4.0, 3.0, 2.0]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![2, 2], vec![5.0_f64, 1.0, 2.0, 6.0]).unwrap();
    let lhs_tangent =
        Tensor::from_vec_col_major(vec![2, 2], vec![10.0_f64, 20.0, 30.0, 40.0]).unwrap();
    let rhs_tangent = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();

    let jvp = ad.jvp_program(&source, &[true, true]).unwrap();
    let compiled = GraphCompiler::new()
        .compile_frozen_program(jvp.frozen())
        .unwrap();
    let runtime = cpu_runtime_with_tropical();
    let tangent = run_one(
        &runtime,
        &compiled,
        &[&lhs, &rhs, &lhs_tangent, &rhs_tangent],
    )
    .unwrap();
    assert_eq!(
        tangent.as_slice::<f64>().unwrap(),
        &[11.0, 21.0, 34.0, 44.0]
    );

    let output_cotangent =
        Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 1.0, 1.0, 1.0]).unwrap();
    let vjp = ad.vjp_program(&source, &[true, true], &[true]).unwrap();
    let compiled = GraphCompiler::new()
        .compile_frozen_program(vjp.frozen())
        .unwrap();
    let cotangents = runtime
        .run_compiled(&compiled, &[&lhs, &rhs, &output_cotangent])
        .unwrap();
    assert_eq!(
        cotangents[0].as_slice::<f64>().unwrap(),
        &[1.0, 1.0, 1.0, 1.0]
    );
    assert_eq!(
        cotangents[1].as_slice::<f64>().unwrap(),
        &[2.0, 0.0, 0.0, 2.0]
    );
}
