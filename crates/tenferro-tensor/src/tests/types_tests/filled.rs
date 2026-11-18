use num_complex::{Complex32, Complex64};

use crate::types::{DType, Tensor};

#[test]
fn zeros_ones_and_full_fill_every_core_dtype() {
    let shape = [2, 3];
    let f32s = Tensor::full(DType::F32, shape, 0.1).unwrap();
    assert_eq!(f32s.shape(), &shape);
    assert_eq!(f32s.as_slice::<f32>().unwrap(), &[0.1_f32; 6]);
    assert_eq!(
        Tensor::ones(DType::F64, shape)
            .unwrap()
            .as_slice::<f64>()
            .unwrap(),
        &[1.0; 6]
    );
    assert_eq!(
        Tensor::full(DType::I32, shape, -7.0)
            .unwrap()
            .as_slice::<i32>()
            .unwrap(),
        &[-7; 6]
    );
    assert_eq!(
        Tensor::zeros(DType::I64, shape)
            .unwrap()
            .as_slice::<i64>()
            .unwrap(),
        &[0; 6]
    );
    assert_eq!(
        Tensor::zeros(DType::Bool, shape)
            .unwrap()
            .as_slice::<bool>()
            .unwrap(),
        &[false; 6]
    );
    assert_eq!(
        Tensor::ones(DType::Bool, shape)
            .unwrap()
            .as_slice::<bool>()
            .unwrap(),
        &[true; 6]
    );
    assert_eq!(
        Tensor::ones(DType::C32, shape)
            .unwrap()
            .as_slice::<Complex32>()
            .unwrap(),
        &[Complex32::new(1.0, 0.0); 6]
    );
    assert_eq!(
        Tensor::full(DType::C64, shape, -2.5)
            .unwrap()
            .as_slice::<Complex64>()
            .unwrap(),
        &[Complex64::new(-2.5, 0.0); 6]
    );
}

#[test]
fn full_handles_rank_zero_and_empty_shapes() {
    let scalar = Tensor::full(DType::F64, [0_usize; 0], 3.0).unwrap();
    assert_eq!(scalar.shape(), &[] as &[usize]);
    assert_eq!(scalar.as_slice::<f64>().unwrap(), &[3.0]);
    let empty = Tensor::ones(DType::I32, vec![2, 0]).unwrap();
    assert_eq!(empty.shape(), &[2, 0]);
    assert!(empty.as_slice::<i32>().unwrap().is_empty());
}

#[test]
fn full_rejects_values_the_dtype_cannot_hold() {
    for (dtype, value) in [
        (DType::I32, 0.5),
        (DType::I32, 2.0_f64.powi(31)),
        (DType::I64, f64::NAN),
        (DType::I64, f64::INFINITY),
        (DType::I64, 9.0e15),
        (DType::Bool, 2.0),
        (DType::Bool, -1.0),
    ] {
        let error = Tensor::full(dtype, [2], value).unwrap_err();
        assert!(
            error.to_string().contains("not representable"),
            "{dtype:?} {value}: {error}"
        );
    }
    assert!(Tensor::zeros(DType::F64, [usize::MAX, 2]).is_err());
    assert!(Tensor::zeros(DType::Bool, [usize::MAX, 2]).is_err());

    // A caller-owned scalar has no core value; refuse instead of guessing one.
    let external = DType::External(std::any::TypeId::of::<f64>());
    let error = Tensor::ones(external, [1]).unwrap_err();
    assert!(error.to_string().contains("externally defined"), "{error}");
}
