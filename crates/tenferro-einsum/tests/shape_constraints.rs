use tenferro_einsum::GraphCompilerEinsumExt;
use tenferro_runtime::{DType, Error, GraphCompiler, TracedTensor};

#[test]
fn independent_symbolic_contract_enforces_repeated_einsum_labels() {
    let lhs = TracedTensor::input_symbolic_shape(DType::F64, 2).unwrap();
    let rhs = TracedTensor::input_symbolic_shape(DType::F64, 2).unwrap();
    assert_ne!(
        lhs.axis_sym_dim(1).unwrap(),
        rhs.axis_sym_dim(0).unwrap(),
        "contracted axes must come from independent symbolic origins"
    );

    let mut compiler = GraphCompiler::new();
    let output = compiler.einsum(&[&lhs, &rhs], "ij,jk->ik").unwrap();
    assert!(
        tenferro_runtime::ad_support::constraint_scopes(&output)
            .iter()
            .any(|scope| !scope.is_empty()),
        "direct einsum lowering must retain the extension shape contract"
    );

    compiler
        .compile_with_input_specs(
            &output,
            &[(&lhs, DType::F64, &[2, 3]), (&rhs, DType::F64, &[3, 4])],
        )
        .expect("equal contracted axes should compile");

    let error = compiler
        .compile_with_input_specs(
            &output,
            &[(&lhs, DType::F64, &[2, 3]), (&rhs, DType::F64, &[5, 4])],
        )
        .expect_err("unequal contracted axes must fail before execution");
    assert!(matches!(
        error,
        Error::ShapeConstraintViolation {
            family: "tenferro.einsum.v1",
            ..
        }
    ));
}
