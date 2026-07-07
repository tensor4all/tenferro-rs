use num_complex::Complex64;

use crate::{BackendId, DType, Error, Tensor, TensorRead, TensorScalar, TypedTensor};

fn dtype_of_typed<T: crate::TensorScalar>(_: &TypedTensor<T>) -> DType {
    T::dtype()
}

fn dtype_of_view<T: crate::TensorScalar>(_: &crate::TypedTensorView<'_, T>) -> DType {
    T::dtype()
}

#[test]
fn with_scalar_dispatches_allowed_float_complex_tensors() {
    let tensor = Tensor::from_vec_col_major(
        vec![2],
        vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)],
    )
    .unwrap();

    let dtype = crate::with_scalar!(
        &tensor,
        float_complex,
        backend = BackendId::Cpu,
        op = "dtype_probe",
        |typed| -> crate::Result<DType> { Ok(dtype_of_typed(typed)) }
    )
    .unwrap();

    assert_eq!(dtype, DType::C64);
}

#[test]
fn with_scalar_rejects_dtype_outside_guard_with_structured_error() {
    let tensor = Tensor::from_vec_col_major(vec![1], vec![7_i32]).unwrap();
    let err = crate::with_scalar!(
        &tensor,
        float_only,
        backend = BackendId::Cuda,
        op = "dtype_probe",
        |typed| -> crate::Result<DType> { Ok(dtype_of_typed(typed)) }
    )
    .unwrap_err();

    assert!(matches!(
        err,
        Error::UnsupportedOpDType {
            op: "dtype_probe",
            dtype: DType::I32,
            backend: BackendId::Cuda,
        }
    ));
}

#[test]
fn with_scalar_read_dispatches_tensor_reads_and_views() {
    let tensor = Tensor::from_vec_col_major(vec![2], vec![1.0_f32, 2.0]).unwrap();
    let read = TensorRead::from_tensor(&tensor);
    let dtype = crate::with_scalar_read!(
        read,
        float_only,
        backend = BackendId::Cpu,
        op = "read_probe",
        |view| -> crate::Result<DType> { Ok(dtype_of_view(&view)) }
    )
    .unwrap();

    assert_eq!(dtype, DType::F32);

    let typed = TypedTensor::<f32>::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap();
    let read = TensorRead::from_view(f32::tensor_view(typed.as_view()));
    let dtype = crate::with_scalar_read!(
        read,
        float_only,
        backend = BackendId::Cpu,
        op = "read_probe",
        |view| -> crate::Result<DType> { Ok(dtype_of_view(&view)) }
    )
    .unwrap();

    assert_eq!(dtype, DType::F32);
}
