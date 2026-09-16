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
