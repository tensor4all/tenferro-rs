use super::*;
use crate::{DType, ErrorKind};

fn f64_tensor() -> Tensor {
    Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap()
}

#[test]
fn typed_wrap_matches_dtype() {
    let typed = <Typed<f64> as Repr>::wrap_value(f64_tensor()).unwrap();
    assert_eq!(typed.as_slice().unwrap(), &[1.0, 2.0]);
}

#[test]
fn typed_wrap_mismatch_is_internal_error() {
    let err = <Typed<f32> as Repr>::wrap_value(f64_tensor()).unwrap_err();
    assert!(matches!(err, Error::Internal(_)));
    assert_eq!(err.kind(), ErrorKind::Internal);
}

#[test]
fn erased_wrap_is_identity() {
    let t = <Erased as Repr>::wrap_value(f64_tensor()).unwrap();
    assert_eq!(t.dtype(), DType::F64);
}

#[test]
fn double_reference_goes_through_ref_impl() {
    let t = f64_tensor();
    let r = &&t;
    assert_eq!(r.as_read().shape(), &[2]);
}

#[test]
fn typed_view_and_read_share_storage() {
    let typed = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap();
    let view = typed.as_view();
    let from_view = view.as_read();
    assert_eq!(
        from_view.as_slice::<f64>().unwrap().as_ptr(),
        typed.as_slice().unwrap().as_ptr()
    );
    let read = TensorRead::from_view(f64::tensor_view(view));
    assert_eq!(read.as_read().shape(), &[2]);
}

#[test]
fn typed_erase_write_preserves_metadata() {
    let mut typed = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![0.0; 2]).unwrap();
    let w = <Typed<f64> as Repr>::erase_write(TypedTensorWrite::from_tensor(&mut typed));
    assert_eq!(w.shape(), &[2]);
}
