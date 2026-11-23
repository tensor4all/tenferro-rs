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
fn full_converts_every_exactly_representable_integer() {
    // 2^53 is where f64 spacing first exceeds 1, not the last exact integer.
    for value in [
        2.0_f64.powi(53),
        i64::MIN as f64,           // -(2^63), the inclusive lower bound
        2.0_f64.powi(63) - 1024.0, // largest binary64 integer below 2^63
    ] {
        let tensor = Tensor::full(DType::I64, [1], value).unwrap();
        assert_eq!(
            tensor.as_slice::<i64>().unwrap(),
            &[value as i64],
            "{value:e}"
        );
    }
    // Outside i64, where the saturating `as i64` cast would silently clamp.
    for value in [
        2.0_f64.powi(63),
        -2.0_f64.powi(63) - 2048.0, // one binary64 step below -(2^63)
    ] {
        assert!(Tensor::full(DType::I64, [1], value).is_err(), "{value:e}");
    }
    // The exact-integer widening belongs to i64; i32 keeps its own range.
    assert!(Tensor::full(DType::I32, [1], 2.0_f64.powi(53)).is_err());
    assert_eq!(
        Tensor::full(DType::I32, [1], 2.0_f64.powi(30))
            .unwrap()
            .as_slice::<i32>()
            .unwrap(),
        &[1 << 30]
    );
}

#[test]
fn full_rejects_values_the_dtype_cannot_hold() {
    for (dtype, value) in [
        (DType::I32, 0.5),
        (DType::I32, 2.0_f64.powi(31)),
        (DType::I64, 0.5),
        (DType::I64, f64::NAN),
        (DType::I64, f64::INFINITY),
        (DType::Bool, 2.0),
        (DType::Bool, -1.0),
    ] {
        let error = Tensor::full(dtype, [2], value).unwrap_err();
        assert!(
            error.to_string().contains("not representable"),
            "{dtype:?} {value}: {error}"
        );
    }

    // An overflowing element count is a shape fault, reported before allocation.
    for dtype in [DType::F64, DType::Bool] {
        let error = Tensor::zeros(dtype, [usize::MAX, 2]).unwrap_err();
        assert!(
            matches!(
                error,
                crate::Error::Validation {
                    source: crate::ValidationError::InvalidArgument {
                        argument: "shape",
                        ..
                    },
                    ..
                }
            ),
            "{dtype:?}: {error}"
        );
    }

    // A caller-owned scalar has no core value; refuse instead of guessing one.
    let external = DType::External(std::any::TypeId::of::<f64>());
    let error = Tensor::ones(external, [1]).unwrap_err();
    assert!(error.to_string().contains("externally defined"), "{error}");
}
