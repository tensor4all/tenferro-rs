use num_complex::{Complex32, Complex64};
use tenferro_tensor::{
    Error, ErrorKind, Tensor, TensorRead, TensorViewMut, TensorWrite, ValidationKind,
};

use crate::cubecl::interop::scale_tensor_write;
use crate::cubecl::{download_tensor, gpu_available, upload_tensor, CudaDeviceId, CudaRuntime};

macro_rules! cuda_test {
    ($name:ident, $body:block) => {
        #[test]
        #[ignore = "requires CUDA 12.8+ GPU"]
        fn $name() {
            if !gpu_available() {
                eprintln!("skipping {} — no CUDA device found", stringify!($name));
                return;
            }
            $body
        }
    };
}

fn runtime() -> CudaRuntime {
    CudaRuntime::new(CudaDeviceId::from_ordinal(0)).unwrap()
}

fn assert_state_unchanged(
    output: &Tensor,
    placement: &tenferro_tensor::Placement,
    allocation_domain: Option<tenferro_tensor::AllocationDomainId>,
    runtime: &CudaRuntime,
    runtime_identity: crate::cubecl::CudaRuntimeIdentity,
) {
    assert_eq!(output.placement(), placement);
    assert_eq!(
        TensorRead::from_tensor(output).allocation_domain(),
        allocation_domain
    );
    assert_eq!(runtime.runtime_identity(), runtime_identity);
}

cuda_test!(scale_supported_dtypes_and_factors, {
    let runtime = runtime();
    let factors = [1.0, 0.25, 1.0 / 7.0_f64.sqrt()];

    for &factor in &factors {
        let input = vec![1.0_f32, -2.0, 3.5, -4.5];
        let mut output = upload(
            &runtime,
            Tensor::from_vec_col_major(vec![2, 2], input.clone()).unwrap(),
        );
        let placement = output.placement().clone();
        let allocation_domain = TensorRead::from_tensor(&output).allocation_domain();
        let runtime_identity = runtime.runtime_identity();
        scale_tensor_write(&runtime, TensorWrite::from_tensor(&mut output), factor).unwrap();
        let actual = download_tensor(&runtime, &output).unwrap();
        let actual = actual.as_slice::<f32>().unwrap();
        for (&value, &expected) in actual.iter().zip(input.iter()) {
            assert!((value - expected * factor as f32).abs() < 1e-6);
        }
        assert_state_unchanged(
            &output,
            &placement,
            allocation_domain,
            &runtime,
            runtime_identity,
        );
    }

    for &factor in &factors {
        let input = vec![1.0_f64, -2.0, 3.5, -4.5];
        let mut output = upload(
            &runtime,
            Tensor::from_vec_col_major(vec![2, 2], input.clone()).unwrap(),
        );
        let placement = output.placement().clone();
        let allocation_domain = TensorRead::from_tensor(&output).allocation_domain();
        let runtime_identity = runtime.runtime_identity();
        scale_tensor_write(&runtime, TensorWrite::from_tensor(&mut output), factor).unwrap();
        let actual = download_tensor(&runtime, &output).unwrap();
        let actual = actual.as_slice::<f64>().unwrap();
        for (&value, &expected) in actual.iter().zip(input.iter()) {
            assert!((value - expected * factor).abs() < 1e-12);
        }
        assert_state_unchanged(
            &output,
            &placement,
            allocation_domain,
            &runtime,
            runtime_identity,
        );
    }

    for &factor in &factors {
        let input = vec![
            Complex32::new(1.0, 2.0),
            Complex32::new(-2.0, 0.5),
            Complex32::new(3.5, -1.5),
            Complex32::new(-4.5, 4.0),
        ];
        let mut output = upload(
            &runtime,
            Tensor::from_vec_col_major(vec![2, 2], input.clone()).unwrap(),
        );
        let placement = output.placement().clone();
        let allocation_domain = TensorRead::from_tensor(&output).allocation_domain();
        let runtime_identity = runtime.runtime_identity();
        scale_tensor_write(&runtime, TensorWrite::from_tensor(&mut output), factor).unwrap();
        let actual = download_tensor(&runtime, &output).unwrap();
        let actual = actual.as_slice::<Complex32>().unwrap();
        for (&value, &expected) in actual.iter().zip(input.iter()) {
            let expected = expected * factor as f32;
            assert!((value.re - expected.re).abs() < 1e-5);
            assert!((value.im - expected.im).abs() < 1e-5);
        }
        assert_state_unchanged(
            &output,
            &placement,
            allocation_domain,
            &runtime,
            runtime_identity,
        );
    }

    for &factor in &factors {
        let input = vec![
            Complex64::new(1.0, 2.0),
            Complex64::new(-2.0, 0.5),
            Complex64::new(3.5, -1.5),
            Complex64::new(-4.5, 4.0),
        ];
        let mut output = upload(
            &runtime,
            Tensor::from_vec_col_major(vec![2, 2], input.clone()).unwrap(),
        );
        let placement = output.placement().clone();
        let allocation_domain = TensorRead::from_tensor(&output).allocation_domain();
        let runtime_identity = runtime.runtime_identity();
        scale_tensor_write(&runtime, TensorWrite::from_tensor(&mut output), factor).unwrap();
        let actual = download_tensor(&runtime, &output).unwrap();
        let actual = actual.as_slice::<Complex64>().unwrap();
        for (&value, &expected) in actual.iter().zip(input.iter()) {
            let expected = expected * factor;
            assert!((value.re - expected.re).abs() < 1e-12);
            assert!((value.im - expected.im).abs() < 1e-12);
        }
        assert_state_unchanged(
            &output,
            &placement,
            allocation_domain,
            &runtime,
            runtime_identity,
        );
    }
});

cuda_test!(scale_empty_output_is_a_valid_noop, {
    let runtime = runtime();
    let mut output = upload(
        &runtime,
        Tensor::from_vec_col_major(vec![0, 2], Vec::<f64>::new()).unwrap(),
    );
    let placement = output.placement().clone();
    let allocation_domain = TensorRead::from_tensor(&output).allocation_domain();
    let runtime_identity = runtime.runtime_identity();

    scale_tensor_write(&runtime, TensorWrite::from_tensor(&mut output), 0.25).unwrap();

    let actual = download_tensor(&runtime, &output).unwrap();
    assert_eq!(actual.shape(), &[0, 2]);
    assert!(actual.as_slice::<f64>().unwrap().is_empty());
    assert_state_unchanged(
        &output,
        &placement,
        allocation_domain,
        &runtime,
        runtime_identity,
    );
});

cuda_test!(scale_rejects_host_output, {
    let runtime = runtime();
    let mut output = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();

    let error =
        scale_tensor_write(&runtime, TensorWrite::from_tensor(&mut output), 0.25).unwrap_err();
    assert!(matches!(
        error,
        Error::RuntimeState {
            op: "scale_tensor_write",
            ..
        }
    ));
});

cuda_test!(scale_rejects_foreign_runtime_output, {
    let runtime = runtime();
    let foreign_runtime = CudaRuntime::new(CudaDeviceId::from_ordinal(0)).unwrap();
    let mut output = upload(
        &foreign_runtime,
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(),
    );

    let error =
        scale_tensor_write(&runtime, TensorWrite::from_tensor(&mut output), 0.25).unwrap_err();
    assert!(matches!(
        error,
        Error::RuntimeState {
            op: "scale_tensor_write",
            ..
        }
    ));
});

cuda_test!(scale_rejects_unsupported_dtype, {
    let runtime = runtime();
    let mut output = upload(
        &runtime,
        Tensor::from_vec_col_major(vec![2], vec![1_i32, 2]).unwrap(),
    );

    let error =
        scale_tensor_write(&runtime, TensorWrite::from_tensor(&mut output), 0.25).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(error.to_string().contains("scale_tensor_write"));
});

cuda_test!(scale_validates_placement_before_unsupported_dtype, {
    let runtime = runtime();
    let mut host_output = Tensor::from_vec_col_major(vec![2], vec![1_i32, 2]).unwrap();
    let error =
        scale_tensor_write(&runtime, TensorWrite::from_tensor(&mut host_output), 0.25).unwrap_err();
    assert!(matches!(
        error,
        Error::RuntimeState {
            op: "scale_tensor_write",
            ..
        }
    ));

    let foreign_runtime = CudaRuntime::new(CudaDeviceId::from_ordinal(0)).unwrap();
    let mut foreign_output = upload(
        &foreign_runtime,
        Tensor::from_vec_col_major(vec![2], vec![1_i32, 2]).unwrap(),
    );
    let error = scale_tensor_write(
        &runtime,
        TensorWrite::from_tensor(&mut foreign_output),
        0.25,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        Error::RuntimeState {
            op: "scale_tensor_write",
            ..
        }
    ));
});

cuda_test!(scale_tensor_write_view_full_buffer_is_safe, {
    let runtime = runtime();
    let input = vec![1.0_f64, -2.0, 3.5, -4.5];
    let mut output = upload(
        &runtime,
        Tensor::from_vec_col_major(vec![input.len()], input.clone()).unwrap(),
    );

    {
        let Tensor::F64(output) = &mut output else {
            unreachable!()
        };
        let view = output
            .backend_region_view_mut(vec![input.len()], vec![1], 0)
            .unwrap();
        scale_tensor_write(
            &runtime,
            TensorWrite::from_view(TensorViewMut::F64(view)),
            0.25,
        )
        .unwrap();
    }

    let actual = download_tensor(&runtime, &output).unwrap();
    assert_eq!(
        actual.as_slice::<f64>().unwrap(),
        &[0.25, -0.5, 0.875, -1.125]
    );
});

cuda_test!(scale_tensor_write_view_compact_prefix_preserves_outside, {
    let runtime = runtime();
    let input = vec![1.0_f64, -2.0, 3.5, -4.5, 8.0, 9.0];
    let mut output = upload(
        &runtime,
        Tensor::from_vec_col_major(vec![input.len()], input.clone()).unwrap(),
    );

    {
        let Tensor::F64(output) = &mut output else {
            unreachable!()
        };
        let view = output.backend_region_view_mut(vec![3], vec![1], 0).unwrap();
        scale_tensor_write(
            &runtime,
            TensorWrite::from_view(TensorViewMut::F64(view)),
            0.25,
        )
        .unwrap();
    }

    let actual = download_tensor(&runtime, &output).unwrap();
    assert_eq!(
        actual.as_slice::<f64>().unwrap(),
        &[0.25, -0.5, 0.875, -4.5, 8.0, 9.0]
    );
});

cuda_test!(
    scale_tensor_write_view_rejects_nonzero_offset_without_writes,
    {
        let runtime = runtime();
        let input = vec![1.0_f64, -2.0, 3.5, -4.5, 8.0, 9.0];
        let mut output = upload(
            &runtime,
            Tensor::from_vec_col_major(vec![input.len()], input.clone()).unwrap(),
        );

        let error = {
            let Tensor::F64(output) = &mut output else {
                unreachable!()
            };
            let view = output.backend_region_view_mut(vec![3], vec![1], 1).unwrap();
            scale_tensor_write(
                &runtime,
                TensorWrite::from_view(TensorViewMut::F64(view)),
                0.25,
            )
            .unwrap_err()
        };
        assert_eq!(
            error.kind(),
            ErrorKind::Validation(ValidationKind::InvalidArgument)
        );

        let actual = download_tensor(&runtime, &output).unwrap();
        assert_eq!(actual.as_slice::<f64>().unwrap(), input.as_slice());
    }
);

cuda_test!(scale_tensor_write_view_rejects_strided_without_writes, {
    let runtime = runtime();
    let input = vec![1.0_f64, -2.0, 3.5, -4.5, 8.0, 9.0];
    let mut output = upload(
        &runtime,
        Tensor::from_vec_col_major(vec![input.len()], input.clone()).unwrap(),
    );

    let error = {
        let Tensor::F64(output) = &mut output else {
            unreachable!()
        };
        let view = output.backend_region_view_mut(vec![3], vec![2], 0).unwrap();
        scale_tensor_write(
            &runtime,
            TensorWrite::from_view(TensorViewMut::F64(view)),
            0.25,
        )
        .unwrap_err()
    };
    assert_eq!(
        error.kind(),
        ErrorKind::Validation(ValidationKind::InvalidArgument)
    );

    let actual = download_tensor(&runtime, &output).unwrap();
    assert_eq!(actual.as_slice::<f64>().unwrap(), input.as_slice());
});

cuda_test!(scale_tensor_write_view_empty_is_a_noop, {
    let runtime = runtime();
    let input = vec![1.0_f64, -2.0, 3.5, -4.5];
    let mut output = upload(
        &runtime,
        Tensor::from_vec_col_major(vec![input.len()], input.clone()).unwrap(),
    );

    {
        let Tensor::F64(output) = &mut output else {
            unreachable!()
        };
        let view = output.backend_region_view_mut(vec![0], vec![1], 0).unwrap();
        scale_tensor_write(
            &runtime,
            TensorWrite::from_view(TensorViewMut::F64(view)),
            0.25,
        )
        .unwrap();
    }

    let actual = download_tensor(&runtime, &output).unwrap();
    assert_eq!(actual.as_slice::<f64>().unwrap(), input.as_slice());
});

fn upload(runtime: &CudaRuntime, tensor: Tensor) -> Tensor {
    upload_tensor(runtime, &tensor).unwrap()
}
