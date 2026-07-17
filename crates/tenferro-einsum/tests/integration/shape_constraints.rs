use tenferro_einsum::{EinsumOptimize, GraphCompilerEinsumExt};
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
        !tenferro_runtime::ad_support::ConstraintScopeTransfer::from_tensor(&output).is_empty(),
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

fn assert_expanded_strategy_enforces_contract(optimize: EinsumOptimize) {
    let lhs = TracedTensor::input_symbolic_shape(DType::F64, 2).unwrap();
    let middle = TracedTensor::input_symbolic_shape(DType::F64, 2).unwrap();
    let rhs = TracedTensor::input_symbolic_shape(DType::F64, 2).unwrap();
    assert_ne!(
        lhs.axis_sym_dim(1).unwrap(),
        middle.axis_sym_dim(0).unwrap()
    );
    assert_ne!(
        middle.axis_sym_dim(1).unwrap(),
        rhs.axis_sym_dim(0).unwrap()
    );

    let mut compiler = GraphCompiler::new();
    let output = compiler
        .einsum_with(&[&lhs, &middle, &rhs], "ab,bc,cd->ad", optimize)
        .unwrap();
    compiler
        .compile_with_input_specs(
            &output,
            &[
                (&lhs, DType::F64, &[2, 3]),
                (&middle, DType::F64, &[3, 4]),
                (&rhs, DType::F64, &[4, 5]),
            ],
        )
        .expect("matching expanded contraction axes should compile");

    let error = compiler
        .compile_with_input_specs(
            &output,
            &[
                (&lhs, DType::F64, &[2, 3]),
                (&middle, DType::F64, &[7, 4]),
                (&rhs, DType::F64, &[4, 5]),
            ],
        )
        .expect_err("expanded graph must retain repeated-label equality constraints");
    assert!(matches!(
        error,
        Error::ShapeConstraintViolation {
            family: "tenferro.einsum.v1",
            ..
        }
    ));
}

#[test]
fn explicit_false_expanded_strategy_retains_shape_contract() {
    assert_expanded_strategy_enforces_contract(EinsumOptimize::False);
}

#[test]
fn explicit_path_expanded_strategy_retains_shape_contract() {
    assert_expanded_strategy_enforces_contract(EinsumOptimize::Path(vec![(0, 1), (0, 1)]));
}
