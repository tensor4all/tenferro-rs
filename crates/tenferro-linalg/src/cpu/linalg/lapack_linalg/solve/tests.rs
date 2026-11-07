use super::*;
use num_complex::{Complex32, Complex64};

/// Borrow a test operand the way the CPU backend boundary hands it to the provider.
fn host<T: 'static>(tensor: &TypedTensor<T>) -> crate::cpu::linalg::HostView<'_, T> {
    tensor.as_view().try_host("test").unwrap()
}

fn check_batches<T>(make: impl Fn(f64) -> T, error: impl Fn(T, T) -> f64)
where
    T: LapackLinalg + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
{
    let mut buffers = BufferPool::new();
    for batch_shape in [vec![], vec![1], vec![2, 3]] {
        let batches = batch_shape.iter().product::<usize>();
        for transpose in [false, true] {
            let mut matrices = Vec::new();
            let mut rhs = Vec::new();
            let mut expected = Vec::new();
            for batch in 0..batches {
                // Nonsymmetric and (for complex types) non-real: transpose must
                // remain transpose, not adjoint. Different matrices catch stale LU.
                let a = [make(2.0 + batch as f64), make(1.0), make(-1.0), make(3.0)];
                matrices.extend_from_slice(&a);
                for col in 0..2 {
                    let x = [make(1.0 + col as f64), make(2.0 + batch as f64)];
                    expected.extend_from_slice(&x);
                    if transpose {
                        rhs.extend_from_slice(&[
                            a[0] * x[0] + a[1] * x[1],
                            a[2] * x[0] + a[3] * x[1],
                        ]);
                    } else {
                        rhs.extend_from_slice(&[
                            a[0] * x[0] + a[2] * x[1],
                            a[1] * x[0] + a[3] * x[1],
                        ]);
                    }
                }
            }
            let mut shape = vec![2, 2];
            shape.extend_from_slice(&batch_shape);
            let a = TypedTensor::from_vec_col_major(shape.clone(), matrices.clone()).unwrap();
            let b = TypedTensor::from_vec_col_major(shape.clone(), rhs.clone()).unwrap();
            let out = solve(&mut buffers, &host(&a), &host(&b), transpose).unwrap();
            assert_eq!(out.shape(), shape);
            for (&actual, &expected) in out.host_data().unwrap().iter().zip(&expected) {
                assert!(
                    error(actual, expected) < 2e-5,
                    "solution residual too large"
                );
            }
            for (&actual, &expected) in a.host_data().unwrap().iter().zip(&matrices) {
                assert_eq!(error(actual, expected), 0.0, "A was modified");
            }
            for (&actual, &expected) in b.host_data().unwrap().iter().zip(&rhs) {
                assert_eq!(error(actual, expected), 0.0, "B was modified");
            }
        }
    }
}

#[test]
fn batched_solve_reuses_scratch_without_changing_values_or_inputs() {
    check_batches(|x| x, |a, b| (a - b).abs());
    check_batches(|x| x as f32, |a, b| (a - b).abs() as f64);
    check_batches(|x| Complex64::new(x, 0.25 * x), |a, b| (a - b).norm());
    check_batches(
        |x| Complex32::new(x as f32, 0.25 * x as f32),
        |a, b| (a - b).norm() as f64,
    );
}

#[test]
fn batched_solve_validates_empty_batches_and_late_singularity() {
    let mut buffers = BufferPool::new();
    let a = TypedTensor::<f64>::from_vec_col_major([2, 2, 0], vec![]).unwrap();
    let b = TypedTensor::<f64>::from_vec_col_major([2, 1, 0], vec![]).unwrap();
    assert_eq!(
        solve(&mut buffers, &host(&a), &host(&b), false)
            .unwrap()
            .shape(),
        &[2, 1, 0]
    );
    let bad = TypedTensor::<f64>::from_vec_col_major([3, 1, 0], vec![]).unwrap();
    assert!(solve(&mut buffers, &host(&a), &host(&bad), false).is_err());
    let bad = TypedTensor::<f64>::from_vec_col_major([2, 1, 1], vec![1.0; 2]).unwrap();
    assert!(solve(&mut buffers, &host(&a), &host(&bad), false).is_err());
    let a =
        TypedTensor::from_vec_col_major([2, 2, 2], vec![2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0])
            .unwrap();
    let b = TypedTensor::from_vec_col_major([2, 1, 2], vec![1.0; 4]).unwrap();
    assert!(solve(&mut buffers, &host(&a), &host(&b), false).is_err());
    assert_eq!(b.host_data().unwrap(), &[1.0; 4]);
}
