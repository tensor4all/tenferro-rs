use super::*;
use crate::provider::{
    tests::execution_context_fixture, BlasGemmProvider, CpuBatchedMatrixLayout as Layout,
    CpuGemmUninitRequest, CpuUninitGemmProvider, ParallelMode,
};

macro_rules! check_type {
    ($test:ident, $ty:ty, $variant:ident, $value:expr) => {
        #[test]
        fn $test() {
            let value: fn(usize) -> $ty = $value;
            let fixture = execution_context_fixture(1);
            fixture.with_context(ParallelMode::Sequential, |ctx| {
                for (m, n, k, batches) in [
                    (3, 5, 7, 2),
                    (1, 11, 11, 3),
                    (0, 2, 3, 1),
                    (2, 0, 3, 1),
                    (2, 3, 0, 2),
                    (2, 3, 4, 0),
                ] {
                    for transposed in [false, true] {
                        let a: Vec<$ty> = (0..m * k * batches).map(|i| value(i % 13)).collect();
                        let b: Vec<$ty> = (0..k * n * batches).map(|i| value(i % 7)).collect();
                        let lhs = crate::Tensor::from_vec_col_major(
                            if transposed {
                                vec![k, m, batches]
                            } else {
                                vec![m, k, batches]
                            },
                            a.clone(),
                        )
                        .unwrap();
                        let rhs = crate::Tensor::from_vec_col_major(vec![k, n, batches], b.clone())
                            .unwrap();
                        for mode in 0..8 {
                            let l = if mode & 1 == 0 {
                                TensorRead::from_tensor(&lhs)
                            } else {
                                TensorRead::from_view(TensorRead::from_tensor(&lhs).tensor_view())
                            };
                            let r = if mode & 2 == 0 {
                                TensorRead::from_tensor(&rhs)
                            } else {
                                TensorRead::from_view(TensorRead::from_tensor(&rhs).tensor_view())
                            };
                            let mut out = vec![MaybeUninit::<$ty>::uninit(); m * n * batches];
                            // Low bits select owned/view inputs; bit 2 tests
                            // BLAS's alpha=0 full-overwrite early return.
                            let alpha: $ty = if mode & 4 == 0 {
                                num_traits::One::one()
                            } else {
                                Zero::zero()
                            };
                            let mut accumulation =
                                DotGeneralAccumulation::overwrite(lhs.dtype()).unwrap();
                            accumulation.alpha = ContractionScalar::$variant(alpha);
                            let request = CpuGemmUninitRequest::new(
                                &l,
                                &r,
                                m,
                                n,
                                k,
                                batches,
                                if transposed {
                                    Layout::new(0, k as isize, 1, (m * k) as isize)
                                } else {
                                    Layout::new(0, 1, m as isize, (m * k) as isize)
                                },
                                Layout::new(0, 1, k as isize, (k * n) as isize),
                                Layout::new(0, 1, m as isize, (m * n) as isize),
                                accumulation,
                            );
                            // SAFETY: disjoint, aligned, exact-size output; dense
                            // descriptor ranges fit the two input allocations.
                            let outcome = unsafe {
                                let bytes = std::slice::from_raw_parts_mut(
                                    out.as_mut_ptr().cast::<MaybeUninit<u8>>(),
                                    out.len() * size_of::<$ty>(),
                                );
                                BlasGemmProvider.gemm_into_uninit(ctx, request, bytes)
                            }
                            .unwrap();
                            assert_eq!(outcome, CpuProviderOutcome::Executed);
                            for batch in 0..batches {
                                for col in 0..n {
                                    for row in 0..m {
                                        let mut expected = <$ty>::zero();
                                        for t in 0..k {
                                            let ai =
                                                if transposed { t + row * k } else { row + t * m };
                                            expected += a[batch * m * k + ai]
                                                * b[batch * k * n + t + col * k];
                                        }
                                        // SAFETY: Executed proves every output initialized.
                                        let actual = unsafe {
                                            out[batch * m * n + row + col * m].assume_init()
                                        };
                                        assert!(
                                            (num_complex::ComplexFloat::abs(
                                                actual - alpha * expected
                                            ) as f64)
                                                < 1e-4
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            });
        }
    };
}
check_type!(blas_uninit_f32, f32, F32, |i| i as f32 / 8.0 - 0.5);
check_type!(blas_uninit_f64, f64, F64, |i| i as f64 / 8.0 - 0.5);
check_type!(blas_uninit_c32, num_complex::Complex32, C32, |i| {
    num_complex::Complex32::new(i as f32 / 8.0 - 0.5, 0.25)
});
check_type!(blas_uninit_c64, num_complex::Complex64, C64, |i| {
    num_complex::Complex64::new(i as f64 / 8.0 - 0.5, 0.25)
});

#[test]
fn blas_uninit_unsupported_layout_leaves_storage_untouched() {
    let fixture = execution_context_fixture(1);
    fixture.with_context(ParallelMode::Sequential, |ctx| {
        let lhs = crate::Tensor::from_vec_col_major(vec![2, 2], vec![1.0f64; 4]).unwrap();
        let read = TensorRead::from_tensor(&lhs);
        for (conj, layout, reason) in [
            (
                false,
                Layout::new(0, 2, 1, 4),
                CpuProviderUnsupported::Layout(CpuOperand::Output),
            ),
            (
                true,
                Layout::new(0, 1, 2, 4),
                CpuProviderUnsupported::Conjugation,
            ),
        ] {
            let mut accumulation = DotGeneralAccumulation::overwrite(lhs.dtype()).unwrap();
            accumulation.lhs_conj = conj;
            let request = CpuGemmUninitRequest::new(
                &read,
                &read,
                2,
                2,
                2,
                1,
                Layout::new(0, 1, 2, 4),
                Layout::new(0, 1, 2, 4),
                layout,
                accumulation,
            );
            let mut storage = [42.0f64; 4];
            // SAFETY: an aligned, exclusive byte view of this initialized array.
            let bytes = unsafe {
                std::slice::from_raw_parts_mut(
                    storage.as_mut_ptr().cast::<MaybeUninit<u8>>(),
                    size_of::<[f64; 4]>(),
                )
            };
            assert_eq!(
                execute_blas_gemm_request_into_uninit(ctx, request, bytes).unwrap(),
                CpuProviderOutcome::Unsupported(reason)
            );
            assert_eq!(storage, [42.0; 4]);
        }
    });
}

#[test]
fn blas_uninit_rejects_accumulation_and_invalid_storage() {
    let fixture = execution_context_fixture(1);
    fixture.with_context(ParallelMode::Sequential, |ctx| {
        let lhs = crate::Tensor::from_vec_col_major(vec![2, 2], vec![1.0f64; 4]).unwrap();
        let read = TensorRead::from_tensor(&lhs);
        for beta in [0.0, 1.0] {
            let mut accumulation = DotGeneralAccumulation::overwrite(lhs.dtype()).unwrap();
            accumulation.beta = ContractionScalar::F64(beta);
            let request = CpuGemmUninitRequest::new(
                &read,
                &read,
                2,
                2,
                2,
                1,
                Layout::new(0, 1, 2, 4),
                Layout::new(0, 1, 2, 4),
                Layout::new(0, 1, 2, 4),
                accumulation,
            );
            // Exercise the checked leaf, without violating the unsafe witness's
            // caller contract with deliberately short storage.
            let result = execute_blas_gemm_request_into_uninit(ctx, request, &mut []);
            if beta == 0.0 {
                assert!(result.is_err());
            } else {
                assert_eq!(
                    result.unwrap(),
                    CpuProviderOutcome::Unsupported(CpuProviderUnsupported::Accumulation)
                );
            }
        }
    });
}
