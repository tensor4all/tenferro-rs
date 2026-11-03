use tenferro_tensor::DType;

use super::ones_tensor;

#[test]
fn ones_tensor_fills_every_core_dtype_and_rejects_external_scalars() {
    use tenferro_tensor::DType;

    assert_eq!(
        ones_tensor(DType::F32, vec![2])
            .unwrap()
            .as_slice::<f32>()
            .unwrap(),
        &[1.0, 1.0]
    );
    assert_eq!(
        ones_tensor(DType::F64, vec![2])
            .unwrap()
            .as_slice::<f64>()
            .unwrap(),
        &[1.0, 1.0]
    );
    assert_eq!(
        ones_tensor(DType::I32, vec![2])
            .unwrap()
            .as_slice::<i32>()
            .unwrap(),
        &[1, 1]
    );
    assert_eq!(
        ones_tensor(DType::I64, vec![2])
            .unwrap()
            .as_slice::<i64>()
            .unwrap(),
        &[1, 1]
    );
    assert_eq!(
        ones_tensor(DType::Bool, vec![2])
            .unwrap()
            .as_slice::<bool>()
            .unwrap(),
        &[true, true]
    );
    assert_eq!(
        ones_tensor(DType::C32, vec![1])
            .unwrap()
            .as_slice::<tenferro_tensor::Complex32>()
            .unwrap(),
        &[tenferro_tensor::Complex32::new(1.0, 0.0)]
    );
    assert_eq!(
        ones_tensor(DType::C64, vec![1])
            .unwrap()
            .as_slice::<tenferro_tensor::Complex64>()
            .unwrap(),
        &[tenferro_tensor::Complex64::new(1.0, 0.0)]
    );

    // An externally defined scalar is caller-owned; the runtime refuses to
    // fabricate a core identity tensor for it instead of guessing one.
    let external = DType::External(std::any::TypeId::of::<f64>());
    let error = ones_tensor(external, vec![1]).expect_err("external dtype must be rejected");
    assert!(
        matches!(error, crate::Error::Validation { .. }),
        "expected a typed validation error, got {error:?}"
    );
}

#[test]
fn leaf_and_linear_input_keys_reject_computed_values() {
    use super::{leaf_input_key, linear_input_key};
    use crate::TracedTensor;

    let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let computed = x.neg().unwrap();
    let error = leaf_input_key(&computed).expect_err("a computed value has no leaf input key");
    assert!(
        matches!(error, crate::Error::Validation { .. }),
        "expected a typed validation error, got {error:?}"
    );
    assert!(leaf_input_key(&x).is_ok());

    let graph = computed.graph();
    let output = graph.outputs()[0];
    let error = linear_input_key(graph, output).expect_err("an op output is not a linear input");
    assert!(
        matches!(error, crate::Error::Validation { .. }),
        "expected a typed validation error, got {error:?}"
    );
}

#[test]
fn constraint_scope_transfer_reports_whether_it_carries_scopes() {
    assert!(ConstraintScopeTransfer::empty().is_empty());
    assert!(!ConstraintScopeTransfer::with_new(nonempty_constraint_scope(21), []).is_empty());
}
