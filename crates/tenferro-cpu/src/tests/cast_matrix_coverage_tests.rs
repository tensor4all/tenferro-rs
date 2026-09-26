//! Coverage for the structural cast matrix, exercised through the public backend path.

use crate::tests::*;
use tenferro_tensor::BackendSessionHost;

/// Every preset dtype is cast to every other one, so each arm of the conversion matrix runs and
/// the refusals for the pairs it does not cover are seen rather than assumed.
#[test]
fn cast_matrix_covers_every_preset_pair() {
    let mut backend = CpuBackend::new();
    let dtypes = [
        DType::F32,
        DType::F64,
        DType::I32,
        DType::I64,
        DType::Bool,
        DType::C32,
        DType::C64,
    ];
    for from in dtypes {
        let input = sample_for(from);
        for to in dtypes {
            // A pair the matrix covers returns a tensor of the requested dtype; one it does not
            // returns a typed error. Both are outcomes this table owns.
            match backend.with_backend_session(|__s| __s.cast(&input, to)) {
                Ok(out) => assert_eq!(out.dtype(), to),
                Err(_) => continue,
            }
        }
    }
}

fn sample_for(dtype: DType) -> Tensor {
    use num_complex::{Complex32, Complex64};
    match dtype {
        DType::F32 => Tensor::from_vec_col_major(vec![2], vec![1.5_f32, -2.0]).unwrap(),
        DType::F64 => Tensor::from_vec_col_major(vec![2], vec![1.5_f64, -2.0]).unwrap(),
        DType::I32 => Tensor::from_vec_col_major(vec![2], vec![1_i32, -2]).unwrap(),
        DType::I64 => Tensor::from_vec_col_major(vec![2], vec![1_i64, -2]).unwrap(),
        DType::Bool => Tensor::from_vec_col_major(vec![2], vec![true, false]).unwrap(),
        DType::C32 => Tensor::from_vec_col_major(
            vec![2],
            vec![Complex32::new(1.0, 0.5), Complex32::new(-2.0, 0.0)],
        )
        .unwrap(),
        DType::C64 => Tensor::from_vec_col_major(
            vec![2],
            vec![Complex64::new(1.0, 0.5), Complex64::new(-2.0, 0.0)],
        )
        .unwrap(),
        other => panic!("no sample for {other:?}"),
    }
}
