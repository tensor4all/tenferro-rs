use num_complex::Complex64;

use crate::Complex32;

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
        Error::UnsupportedDType {
            op: "dtype_probe",
            dtype: DType::I32,
            ..
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

#[test]
fn scalar_operand_refuses_a_dtype_other_than_the_requested_scalar() {
    let tensor = Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap();
    let error = crate::dispatch::scalar_operand::<f32>(&tensor, "coverage_probe", BackendId::Cpu)
        .expect_err("an f64 tensor is not an f32 scalar");
    assert!(matches!(error, Error::UnsupportedDType { .. }));
}

/// Dispatch one preset dtype through every guard of both exported macros.
macro_rules! cover_every_guard {
    ($scalar:ty, $values:expr) => {{
        let mk = || Tensor::from_vec_col_major(vec![2], $values).unwrap();
        let expected = <$scalar as TensorScalar>::dtype();

        // `all` admits every preset scalar, through the value and the read macro.
        let all_dtype = crate::with_scalar!(
            &mk(),
            all,
            backend = BackendId::Cpu,
            op = "coverage",
            |typed| -> crate::Result<DType> { Ok(dtype_of_typed(typed)) }
        )
        .unwrap();
        assert_eq!(all_dtype, expected);

        let read = TensorRead::Tensor(&mk());
        let all_read = crate::with_scalar_read!(
            &read,
            all,
            backend = BackendId::Cpu,
            op = "coverage",
            |view| -> crate::Result<DType> { Ok(dtype_of_view(&view)) }
        )
        .unwrap();
        assert_eq!(all_read, expected);

        // The narrower guards admit a prefix of the presets and refuse the rest.
        let read = TensorRead::Tensor(&mk());
        match crate::with_scalar_read!(
            &read,
            numeric,
            backend = BackendId::Cpu,
            op = "coverage",
            |view| -> crate::Result<DType> { Ok(dtype_of_view(&view)) }
        ) {
            Ok(dtype) => assert_eq!(dtype, expected),
            Err(_) => assert_eq!(expected, DType::Bool),
        }
        let read = TensorRead::Tensor(&mk());
        match crate::with_scalar_read!(
            &read,
            float_complex,
            backend = BackendId::Cpu,
            op = "coverage",
            |view| -> crate::Result<DType> { Ok(dtype_of_view(&view)) }
        ) {
            Ok(dtype) => assert_eq!(dtype, expected),
            Err(_) => assert!(matches!(expected, DType::I32 | DType::I64 | DType::Bool)),
        }
        let read = TensorRead::Tensor(&mk());
        match crate::with_scalar_read!(
            &read,
            float_only,
            backend = BackendId::Cpu,
            op = "coverage",
            |view| -> crate::Result<DType> { Ok(dtype_of_view(&view)) }
        ) {
            Ok(dtype) => assert_eq!(dtype, expected),
            Err(_) => assert!(!matches!(expected, DType::F32 | DType::F64)),
        }

        // The value macro's narrower guards refuse the same way.
        match crate::with_scalar!(
            &mk(),
            float_only,
            backend = BackendId::Cpu,
            op = "coverage",
            |typed| -> crate::Result<DType> { Ok(dtype_of_typed(typed)) }
        ) {
            Ok(dtype) => assert_eq!(dtype, expected),
            Err(_) => assert!(!matches!(expected, DType::F32 | DType::F64)),
        }
        match crate::with_scalar!(
            &mk(),
            float_complex,
            backend = BackendId::Cpu,
            op = "coverage",
            |typed| -> crate::Result<DType> { Ok(dtype_of_typed(typed)) }
        ) {
            Ok(dtype) => assert_eq!(dtype, expected),
            Err(_) => assert!(matches!(expected, DType::I32 | DType::I64 | DType::Bool)),
        }
        match crate::with_scalar!(
            &mk(),
            numeric,
            backend = BackendId::Cpu,
            op = "coverage",
            |typed| -> crate::Result<DType> { Ok(dtype_of_typed(typed)) }
        ) {
            Ok(dtype) => assert_eq!(dtype, expected),
            Err(_) => assert_eq!(expected, DType::Bool),
        }
    }};
}

#[test]
fn every_preset_scalar_dispatches_through_both_macros() {
    cover_every_guard!(f32, vec![1.0_f32, 2.0]);
    cover_every_guard!(f64, vec![1.0_f64, 2.0]);
    cover_every_guard!(i32, vec![1_i32, 2]);
    cover_every_guard!(i64, vec![1_i64, 2]);
    cover_every_guard!(bool, vec![true, false]);
    cover_every_guard!(
        Complex32,
        vec![Complex32::new(1.0, 0.0), Complex32::new(2.0, 0.0)]
    );
    cover_every_guard!(
        Complex64,
        vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)]
    );
}
