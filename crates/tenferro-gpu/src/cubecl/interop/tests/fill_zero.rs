//! Device coverage for `fill_zero_write` (issue #1834).
//!
//! The contract is bit-exact: a `beta = 0` consumer needs the destination fully
//! overwritten with `+0.0`, so these tests compare bit patterns rather than
//! values. `NaN`, `Inf`, and `-0.0` all survive a `0 * y` multiply, which is
//! what makes the previous workarounds unusable.

use num_complex::{Complex32, Complex64};
use tenferro_tensor::{ErrorKind, Tensor, TensorViewMut, TensorWrite};

use crate::cubecl::interop::fill_zero_write;
use crate::cubecl::{download_tensor, gpu_available, upload_tensor, CudaDeviceId, CudaRuntime};

macro_rules! cuda_test {
    ($name:ident, $body:block) => {
        #[test]
        #[ignore = "requires CUDA 12.8+ GPU"]
        fn $name() {
            assert!(gpu_available(), "CUDA test requires an available device");
            $body
        }
    };
}

fn runtime() -> CudaRuntime {
    CudaRuntime::new(CudaDeviceId::from_ordinal(0)).unwrap()
}

fn upload(runtime: &CudaRuntime, tensor: Tensor) -> Tensor {
    upload_tensor(runtime, &tensor).unwrap()
}

cuda_test!(
    fill_zero_replaces_nan_inf_and_negative_zero_with_positive_zero,
    {
        let runtime = runtime();

        let poisoned_f64 = vec![f64::NAN, f64::INFINITY, -f64::INFINITY, -0.0, 1.5, -2.5];
        let mut output = upload(
            &runtime,
            Tensor::from_vec_col_major(vec![6], poisoned_f64).unwrap(),
        );
        fill_zero_write(&runtime, TensorWrite::from_tensor(&mut output)).unwrap();
        let actual = download_tensor(&runtime, &output).unwrap();
        for &value in actual.as_slice::<f64>().unwrap() {
            assert_eq!(value.to_bits(), 0.0_f64.to_bits(), "expected +0.0 bits");
        }

        let poisoned_f32 = vec![f32::NAN, -0.0, f32::INFINITY, 3.0];
        let mut output = upload(
            &runtime,
            Tensor::from_vec_col_major(vec![4], poisoned_f32).unwrap(),
        );
        fill_zero_write(&runtime, TensorWrite::from_tensor(&mut output)).unwrap();
        let actual = download_tensor(&runtime, &output).unwrap();
        for &value in actual.as_slice::<f32>().unwrap() {
            assert_eq!(value.to_bits(), 0.0_f32.to_bits(), "expected +0.0 bits");
        }

        let poisoned_c64 = vec![
            Complex64::new(f64::NAN, -0.0),
            Complex64::new(-f64::INFINITY, f64::NAN),
            Complex64::new(-0.0, 7.0),
        ];
        let mut output = upload(
            &runtime,
            Tensor::from_vec_col_major(vec![3], poisoned_c64).unwrap(),
        );
        fill_zero_write(&runtime, TensorWrite::from_tensor(&mut output)).unwrap();
        let actual = download_tensor(&runtime, &output).unwrap();
        for value in actual.as_slice::<Complex64>().unwrap() {
            assert_eq!(value.re.to_bits(), 0.0_f64.to_bits());
            assert_eq!(value.im.to_bits(), 0.0_f64.to_bits());
        }

        let poisoned_c32 = vec![Complex32::new(f32::NAN, -0.0), Complex32::new(-0.0, 1.0)];
        let mut output = upload(
            &runtime,
            Tensor::from_vec_col_major(vec![2], poisoned_c32).unwrap(),
        );
        fill_zero_write(&runtime, TensorWrite::from_tensor(&mut output)).unwrap();
        let actual = download_tensor(&runtime, &output).unwrap();
        for value in actual.as_slice::<Complex32>().unwrap() {
            assert_eq!(value.re.to_bits(), 0.0_f32.to_bits());
            assert_eq!(value.im.to_bits(), 0.0_f32.to_bits());
        }
    }
);

cuda_test!(fill_zero_covers_integer_and_bool_destinations, {
    let runtime = runtime();

    let mut output = upload(
        &runtime,
        Tensor::from_vec_col_major(vec![4], vec![1_i32, -2, 3, -4]).unwrap(),
    );
    fill_zero_write(&runtime, TensorWrite::from_tensor(&mut output)).unwrap();
    let actual = download_tensor(&runtime, &output).unwrap();
    assert_eq!(actual.as_slice::<i32>().unwrap(), &[0, 0, 0, 0]);

    let mut output = upload(
        &runtime,
        Tensor::from_vec_col_major(vec![3], vec![7_i64, -8, 9]).unwrap(),
    );
    fill_zero_write(&runtime, TensorWrite::from_tensor(&mut output)).unwrap();
    let actual = download_tensor(&runtime, &output).unwrap();
    assert_eq!(actual.as_slice::<i64>().unwrap(), &[0, 0, 0]);

    let mut output = upload(
        &runtime,
        Tensor::from_vec_col_major(vec![3], vec![true, true, false]).unwrap(),
    );
    fill_zero_write(&runtime, TensorWrite::from_tensor(&mut output)).unwrap();
    let actual = download_tensor(&runtime, &output).unwrap();
    assert_eq!(actual.as_slice::<bool>().unwrap(), &[false, false, false]);
});

cuda_test!(
    fill_zero_leaves_elements_outside_a_strided_region_untouched,
    {
        let runtime = runtime();

        // 20-element allocation; the destination is a 2x3 region at offset 1 with
        // axis strides (3, 1), so it is neither compact nor zero-offset.
        let original: Vec<f64> = (0..20).map(|value| f64::from(value) + 0.5).collect();
        let mut output = upload(
            &runtime,
            Tensor::from_vec_col_major(vec![20], original.clone()).unwrap(),
        );
        {
            let typed = output.as_typed_mut::<f64>().unwrap();
            let region = typed
                .backend_region_view_mut(vec![2, 3], vec![3, 1], 1)
                .unwrap();
            fill_zero_write(&runtime, TensorWrite::from_view(TensorViewMut::F64(region))).unwrap();
        }

        let mut expected = original;
        for row in 0..2usize {
            for column in 0..3usize {
                expected[1 + 3 * row + column] = 0.0;
            }
        }
        let actual = download_tensor(&runtime, &output).unwrap();
        assert_eq!(actual.as_slice::<f64>().unwrap(), expected.as_slice());
    }
);

cuda_test!(fill_zero_clears_a_compact_region_at_a_nonzero_offset, {
    let runtime = runtime();

    let original: Vec<Complex64> = (0..12)
        .map(|value| Complex64::new(f64::from(value), -f64::from(value)))
        .collect();
    let mut output = upload(
        &runtime,
        Tensor::from_vec_col_major(vec![12], original.clone()).unwrap(),
    );
    {
        let typed = output.as_typed_mut::<Complex64>().unwrap();
        // Compact 2x2 block starting at element 4: the memset path with a
        // nonzero offset.
        let region = typed
            .backend_region_view_mut(vec![2, 2], vec![1, 2], 4)
            .unwrap();
        fill_zero_write(&runtime, TensorWrite::from_view(TensorViewMut::C64(region))).unwrap();
    }

    let mut expected = original;
    for value in expected.iter_mut().skip(4).take(4) {
        *value = Complex64::new(0.0, 0.0);
    }
    let actual = download_tensor(&runtime, &output).unwrap();
    assert_eq!(actual.as_slice::<Complex64>().unwrap(), expected.as_slice());
});

cuda_test!(
    fill_zero_reports_a_strided_bool_destination_as_unsupported,
    {
        let runtime = runtime();

        let mut output = upload(
            &runtime,
            Tensor::from_vec_col_major(vec![8], vec![true; 8]).unwrap(),
        );
        let typed = output.as_typed_mut::<bool>().unwrap();
        let region = typed
            .backend_region_view_mut(vec![2, 2], vec![2, 1], 0)
            .unwrap();
        let error = fill_zero_write(
            &runtime,
            TensorWrite::from_view(TensorViewMut::Bool(region)),
        )
        .unwrap_err();

        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert!(
            error.to_string().contains("strided Bool destination"),
            "unexpected diagnosis: {error}"
        );
    }
);

cuda_test!(fill_zero_is_a_no_op_for_an_empty_destination, {
    let runtime = runtime();

    let mut output = upload(
        &runtime,
        Tensor::from_vec_col_major(vec![0], Vec::<f64>::new()).unwrap(),
    );
    fill_zero_write(&runtime, TensorWrite::from_tensor(&mut output)).unwrap();
    let actual = download_tensor(&runtime, &output).unwrap();
    assert!(actual.as_slice::<f64>().unwrap().is_empty());
});

cuda_test!(fill_zero_does_not_allocate, {
    let runtime = runtime();

    let mut output = upload(
        &runtime,
        Tensor::from_vec_col_major(vec![256], vec![1.5_f64; 256]).unwrap(),
    );
    // Warm the path once, then pin the active-allocation count across repeats:
    // a materialized zero source or a staging buffer would show up here.
    fill_zero_write(&runtime, TensorWrite::from_tensor(&mut output)).unwrap();
    let before = runtime.client().memory_usage().unwrap();
    for _ in 0..4 {
        fill_zero_write(&runtime, TensorWrite::from_tensor(&mut output)).unwrap();
    }
    let after = runtime.client().memory_usage().unwrap();
    assert_eq!(after.number_allocs, before.number_allocs);
    assert_eq!(after.bytes_in_use, before.bytes_in_use);

    let actual = download_tensor(&runtime, &output).unwrap();
    assert!(actual.as_slice::<f64>().unwrap().iter().all(|&v| v == 0.0));
});
