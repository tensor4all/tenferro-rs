use tenferro_einsum::Subscripts;
use std::panic::{catch_unwind, AssertUnwindSafe};
use tenferro_ext_tropical::{
    traced::{tropical_einsum_subscripts, try_tropical_dot_general_fused},
    MaxPlus, MinPlus, TropicalKind,
};
use tenferro_runtime::{DType, TracedTensor};

#[test]
fn tropical_crate_exports_core_types() {
    assert_eq!(TropicalKind::MaxPlus, TropicalKind::MaxPlus);
    assert_eq!(MaxPlus(2.0_f64).value(), 2.0);
    assert_eq!(MinPlus(3.0_f64).value(), 3.0);
}

#[test]
fn traced_einsum_validation_rejects_inputs_before_extension_shape_inference() {
    let f64_lhs = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64; 4]);
    let f64_rhs = TracedTensor::from_vec_col_major(vec![3, 2], vec![1.0_f64; 6]);
    let f32_rhs = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f32; 4]);
    let i32_rhs = TracedTensor::input_concrete_shape(DType::I32, &[2, 2]);
    let matmul = Subscripts::parse("ij,jk->ik").unwrap();
    let missing_output = Subscripts::parse("ij,jk->ix").unwrap();

    assert!(
        tropical_einsum_subscripts(TropicalKind::MaxPlus, &[&f64_lhs, &f32_rhs], &matmul).is_err()
    );
    assert!(
        tropical_einsum_subscripts(TropicalKind::MaxPlus, &[&i32_rhs, &i32_rhs], &matmul).is_err()
    );
    assert!(
        tropical_einsum_subscripts(TropicalKind::MaxPlus, &[&f64_lhs, &f64_rhs], &matmul).is_err()
    );
    assert!(tropical_einsum_subscripts(
        TropicalKind::MaxPlus,
        &[&f64_lhs, &f64_lhs],
        &missing_output
    )
    .is_err());
}

#[test]
fn traced_einsum_accepts_symbolic_shapes_without_panicking() {
    let lhs = TracedTensor::input_symbolic_shape(DType::F64, 2);
    let rhs = TracedTensor::input_symbolic_shape(DType::F64, 2);
    let matmul = Subscripts::parse("ij,jk->ik").unwrap();

    let result = catch_unwind(AssertUnwindSafe(|| {
        tropical_einsum_subscripts(TropicalKind::MaxPlus, &[&lhs, &rhs], &matmul)
    }));

    assert!(result.is_ok(), "symbolic tropical einsum should not panic");
    let output = result.unwrap().unwrap();
    assert_eq!(output.rank, 2);
    assert!(output.sym_shape().is_none());
}

#[test]
fn fused_dot_general_has_fallible_validation_entrypoint() {
    let scalar = TracedTensor::input_concrete_shape(DType::F64, &[]);
    let matrix = TracedTensor::input_concrete_shape(DType::F64, &[2, 2]);
    let lhs = TracedTensor::input_concrete_shape(DType::F64, &[2, 3]);
    let rhs = TracedTensor::input_concrete_shape(DType::F64, &[4, 2]);

    assert!(try_tropical_dot_general_fused(&scalar, &matrix).is_err());
    assert!(try_tropical_dot_general_fused(&lhs, &rhs).is_err());
}
