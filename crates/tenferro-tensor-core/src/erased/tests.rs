use crate::{ErasedHostTensor, HostTensor};

fn f64_tensor(values: &[f64]) -> HostTensor<f64> {
    HostTensor::from_vec_col_major(vec![values.len()], values.to_vec()).unwrap()
}

#[test]
fn type_identity_uses_the_actual_rust_type() {
    let erased = ErasedHostTensor::new(f64_tensor(&[1.0]));

    assert_eq!(erased.type_id(), core::any::TypeId::of::<HostTensor<f64>>());
    assert!(erased.is::<f64>());
    assert!(!erased.is::<f32>());
    assert!(!erased.is::<i32>());
}

#[test]
fn recovery_by_the_wrong_type_returns_nothing() {
    let mut erased = ErasedHostTensor::new(f64_tensor(&[1.0]));

    assert!(erased.downcast_ref::<f32>().is_none());
    assert!(erased.downcast_mut::<f32>().is_none());
    assert!(erased.into_typed::<f32>().is_none());
}

#[test]
fn recovery_by_the_right_type_returns_the_tensor() {
    let mut erased = ErasedHostTensor::new(f64_tensor(&[1.0, 2.0]));

    assert_eq!(
        erased.downcast_ref::<f64>().unwrap().as_slice(),
        &[1.0, 2.0]
    );
    erased.downcast_mut::<f64>().unwrap().as_mut_slice()[0] = 9.0;
    assert_eq!(
        erased.downcast_ref::<f64>().unwrap().as_slice(),
        &[9.0, 2.0]
    );
    assert_eq!(erased.into_typed::<f64>().unwrap().as_slice(), &[9.0, 2.0]);
}

#[test]
fn debug_reports_a_stable_description() {
    let erased = ErasedHostTensor::new(f64_tensor(&[1.0]));
    let rendered = format!("{erased:?}");

    assert!(rendered.starts_with("ErasedHostTensor"));
    assert!(rendered.contains("type_id"));
}

fn matrix(values: &[f64]) -> ErasedHostTensor {
    ErasedHostTensor::new(HostTensor::from_vec_col_major(vec![2, 2], values.to_vec()).unwrap())
}

#[test]
fn a_new_payload_presents_a_dense_column_major_view() {
    let erased = matrix(&[1.0, 2.0, 3.0, 4.0]);

    assert_eq!(erased.shape(), &[2, 2]);
    assert_eq!(erased.strides(), &[1, 2]);
    assert_eq!(erased.offset(), 0);
    assert_eq!(erased.element_count(), 4);
    assert_eq!(erased.payload_element_count(), 4);
    assert!(erased.is_contiguous());
    assert_eq!(erased.as_dense::<f64>().unwrap().0, &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn a_permutation_shares_the_payload_and_exchanges_the_axes() {
    let erased = matrix(&[1.0, 2.0, 3.0, 4.0]);
    let permuted = erased.permuted(&[1, 0]).unwrap();

    assert!(permuted.shares_payload_with(&erased));
    assert_eq!(permuted.shape(), &[2, 2]);
    assert_eq!(permuted.strides(), &[2, 1]);
    assert!(!permuted.is_contiguous());
    assert_eq!(permuted.element_count(), 4);
    assert_eq!(permuted.payload_element_count(), 4);

    // The view names the exchanged elements.
    assert_eq!(permuted.element_at::<f64>(&[0, 0]), Some(&1.0));
    assert_eq!(permuted.element_at::<f64>(&[1, 0]), Some(&3.0));
    assert_eq!(permuted.element_at::<f64>(&[0, 1]), Some(&2.0));
    assert_eq!(permuted.element_at::<f64>(&[1, 1]), Some(&4.0));

    // The identity permutation stays dense.
    assert!(erased.permuted(&[0, 1]).unwrap().is_contiguous());
}

#[test]
fn index_queries_outside_the_view_return_nothing() {
    let erased = matrix(&[1.0, 2.0, 3.0, 4.0]);

    assert_eq!(erased.element_at::<f64>(&[2, 0]), None);
    assert_eq!(erased.element_at::<f64>(&[0]), None);
    assert_eq!(erased.element_at::<f64>(&[0, 0, 0]), None);
    assert_eq!(erased.element_at::<f32>(&[0, 0]), None);
}

#[test]
fn an_invalid_permutation_reports_the_specific_error() {
    let erased = matrix(&[1.0, 2.0, 3.0, 4.0]);

    assert!(matches!(
        erased.permuted(&[0]),
        Err(crate::ValidationError::InvalidPermutationLength {
            expected: 2,
            actual: 1
        })
    ));
    assert!(matches!(
        erased.permuted(&[0, 2]),
        Err(crate::ValidationError::AxisOutOfBounds { axis: 2, rank: 2 })
    ));
    assert!(matches!(
        erased.permuted(&[0, 0]),
        Err(crate::ValidationError::DuplicateAxis { axis: 0, .. })
    ));
}

#[test]
fn materialization_gathers_the_view_into_dense_order() {
    let erased = matrix(&[1.0, 2.0, 3.0, 4.0]);
    let contiguous = erased.permuted(&[1, 0]).unwrap().to_contiguous().unwrap();

    assert!(!contiguous.shares_payload_with(&erased));
    assert!(contiguous.is_contiguous());
    assert_eq!(contiguous.shape(), &[2, 2]);
    assert_eq!(contiguous.payload_element_count(), 4);
    assert_eq!(
        contiguous.as_dense::<f64>().unwrap().0,
        &[1.0, 3.0, 2.0, 4.0]
    );
}

#[test]
fn duplication_copies_the_payload_while_a_clone_shares_it() {
    let erased = matrix(&[1.0, 2.0, 3.0, 4.0]);

    let mut copy = erased.duplicate();
    assert!(!copy.shares_payload_with(&erased));
    copy.downcast_mut::<f64>().unwrap().as_mut_slice()[0] = 9.0;
    assert_eq!(erased.as_dense::<f64>().unwrap().0, &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(copy.as_dense::<f64>().unwrap().0, &[9.0, 2.0, 3.0, 4.0]);

    // A shared clone cannot borrow mutably and cannot be taken by value.
    let shared = erased.clone();
    assert!(shared.clone().downcast_mut::<f64>().is_none());
    assert!(erased.clone().into_typed::<f64>().is_none());
    drop(shared);
    assert_eq!(erased.into_typed::<f64>().unwrap().as_slice().len(), 4);
}

#[test]
fn a_strided_view_refuses_the_dense_accessors() {
    let erased = matrix(&[1.0, 2.0, 3.0, 4.0]);
    let mut permuted = erased.permuted(&[1, 0]).unwrap();

    assert!(permuted.downcast_ref::<f64>().is_none());
    assert!(permuted.clone().downcast_mut::<f64>().is_none());
    assert!(permuted.as_dense::<f64>().is_none());
    assert!(permuted.clone().into_typed::<f64>().is_none());

    // A mutable element borrow needs the only reference to the payload.
    assert!(permuted.element_at_mut::<f64>(&[1, 0]).is_none());

    // The independent copy accepts it, and the write follows the view's strides.
    let mut owned = erased.permuted(&[1, 0]).unwrap().duplicate();
    *owned.element_at_mut::<f64>(&[1, 0]).unwrap() = 30.0;
    assert_eq!(owned.element_at::<f64>(&[1, 0]), Some(&30.0));
    assert_eq!(owned.element_at::<f64>(&[0, 1]), Some(&2.0));
    assert_eq!(erased.as_dense::<f64>().unwrap().0, &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn a_debug_rendering_includes_the_view_layout() {
    let rendered = format!(
        "{:?}",
        matrix(&[1.0, 2.0, 3.0, 4.0]).permuted(&[1, 0]).unwrap()
    );

    assert!(rendered.starts_with("ErasedHostTensor"));
    assert!(rendered.contains("strides"));
    assert!(rendered.contains("offset"));
}
