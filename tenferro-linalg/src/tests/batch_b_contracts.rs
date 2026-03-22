use super::*;
#[cfg(feature = "cuda")]
use num_complex::Complex;
#[cfg(feature = "cuda")]
use num_traits::Float;
use tenferro_tensor::{MemoryOrder, Tensor};

#[cfg(feature = "cuda")]
fn with_cuda_ctx<T>(f: impl FnOnce(&mut tenferro_prims::CudaContext) -> T) -> Option<T> {
    let path = [
        "/usr/lib/x86_64-linux-gnu/libcutensor/12/libcutensor.so",
        "/usr/lib/x86_64-linux-gnu/libcutensor.so",
        "/usr/lib/libcutensor.so",
    ]
    .into_iter()
    .find(|path| std::path::Path::new(path).exists())?;
    let (_backend, mut ctx) = tenferro_prims::CudaBackend::load(path).ok()?;
    Some(f(&mut ctx))
}

#[cfg(not(feature = "cuda"))]
fn with_cuda_ctx<T>(f: impl FnOnce(&mut tenferro_prims::CudaContext) -> T) -> Option<T> {
    let mut ctx = tenferro_prims::CudaContext::new();
    Some(f(&mut ctx))
}

#[cfg(feature = "cuda")]
fn assert_close_slice(label: &str, got: &[f64], expected: &[f64], atol: f64) {
    assert_eq!(
        got.len(),
        expected.len(),
        "{label}: length mismatch {} vs {}",
        got.len(),
        expected.len()
    );
    for (idx, (&got_v, &exp_v)) in got.iter().zip(expected.iter()).enumerate() {
        let diff = (got_v - exp_v).abs();
        assert!(
            diff <= atol,
            "{label}[{idx}] diff {diff} exceeded tolerance {atol}; got {got_v}, expected {exp_v}"
        );
    }
}

#[cfg(feature = "cuda")]
fn assert_close_real_slice<T: Float + std::fmt::Debug>(
    label: &str,
    got: &[T],
    expected: &[T],
    atol: T,
) {
    assert_eq!(
        got.len(),
        expected.len(),
        "{label}: length mismatch {} vs {}",
        got.len(),
        expected.len()
    );
    for (idx, (&got_v, &exp_v)) in got.iter().zip(expected.iter()).enumerate() {
        let diff = (got_v - exp_v).abs();
        assert!(
            diff <= atol,
            "{label}[{idx}] diff {diff:?} exceeded tolerance {atol:?}; got {got_v:?}, expected {exp_v:?}"
        );
    }
}

#[cfg(feature = "cuda")]
fn tensor_data_on_cpu<T: tenferro_algebra::Scalar + Copy>(tensor: &Tensor<T>) -> Vec<T> {
    let cpu = tensor
        .to_memory_space_async(tenferro_device::LogicalMemorySpace::MainMemory)
        .unwrap();
    let contiguous = cpu.contiguous(MemoryOrder::ColumnMajor);
    let offset = contiguous.offset() as usize;
    let len = contiguous.dims().iter().product::<usize>().max(1);
    contiguous.buffer().as_slice().unwrap()[offset..offset + len].to_vec()
}

#[cfg(feature = "cuda")]
fn matmul_col_major(lhs: &[f64], m: usize, k: usize, rhs: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0; m * n];
    for col in 0..n {
        for row in 0..m {
            let mut acc = 0.0;
            for inner in 0..k {
                acc += lhs[row + inner * m] * rhs[inner + col * k];
            }
            out[row + col * m] = acc;
        }
    }
    out
}

#[cfg(feature = "cuda")]
fn reconstruct_thin_svd_col_major(
    u: &[f64],
    m: usize,
    k: usize,
    s: &[f64],
    vt: &[f64],
    n: usize,
) -> Vec<f64> {
    let mut scaled_u = u.to_vec();
    for col in 0..k {
        for row in 0..m {
            scaled_u[row + col * m] *= s[col];
        }
    }
    matmul_col_major(&scaled_u, m, k, vt, n)
}

#[cfg(feature = "cuda")]
fn reconstruct_thin_svd_col_major_complex<T: Float>(
    u: &[Complex<T>],
    m: usize,
    k: usize,
    s: &[T],
    vt: &[Complex<T>],
    n: usize,
) -> Vec<Complex<T>> {
    let mut scaled_u = u.to_vec();
    for col in 0..k {
        for row in 0..m {
            scaled_u[row + col * m] = scaled_u[row + col * m] * Complex::new(s[col], T::zero());
        }
    }
    matmul_col_major_complex(&scaled_u, m, k, vt, n)
}

#[cfg(feature = "cuda")]
fn matmul_col_major_complex<T: Float>(
    lhs: &[Complex<T>],
    m: usize,
    k: usize,
    rhs: &[Complex<T>],
    n: usize,
) -> Vec<Complex<T>> {
    let mut out = vec![Complex::new(T::zero(), T::zero()); m * n];
    for col in 0..n {
        for row in 0..m {
            let mut acc = Complex::new(T::zero(), T::zero());
            for inner in 0..k {
                acc = acc + lhs[row + inner * m] * rhs[inner + col * k];
            }
            out[row + col * m] = acc;
        }
    }
    out
}

#[cfg(feature = "cuda")]
fn gram_col_major_complex<T: Float>(q: &[Complex<T>], m: usize, k: usize) -> Vec<Complex<T>> {
    let mut out = vec![Complex::new(T::zero(), T::zero()); k * k];
    for col in 0..k {
        for row in 0..k {
            let mut acc = Complex::new(T::zero(), T::zero());
            for inner in 0..m {
                acc = acc + q[inner + row * m].conj() * q[inner + col * m];
            }
            out[row + col * k] = acc;
        }
    }
    out
}

#[cfg(feature = "cuda")]
fn assert_close_complex_slice<T: Float + std::fmt::Debug>(
    label: &str,
    got: &[Complex<T>],
    expected: &[Complex<T>],
    atol: T,
) {
    assert_eq!(
        got.len(),
        expected.len(),
        "{label}: length mismatch {} vs {}",
        got.len(),
        expected.len()
    );
    for (idx, (&got_v, &exp_v)) in got.iter().zip(expected.iter()).enumerate() {
        let re_diff = (got_v.re - exp_v.re).abs();
        let im_diff = (got_v.im - exp_v.im).abs();
        assert!(
            re_diff <= atol,
            "{label}[{idx}].re diff {re_diff:?} exceeded tolerance {atol:?}; got {got_v:?}, expected {exp_v:?}"
        );
        assert!(
            im_diff <= atol,
            "{label}[{idx}].im diff {im_diff:?} exceeded tolerance {atol:?}; got {got_v:?}, expected {exp_v:?}"
        );
    }
}

#[cfg(feature = "cuda")]
fn public_cuda_svd_matrix(wide: bool) -> Tensor<f64> {
    if wide {
        Tensor::from_slice(
            &[3.0_f64, 1.0, 1.0, 2.0, 0.0, 1.0],
            &[2, 3],
            MemoryOrder::ColumnMajor,
        )
        .unwrap()
    } else {
        Tensor::from_slice(
            &[3.0_f64, 1.0, 0.0, 1.0, 2.0, 1.0],
            &[3, 2],
            MemoryOrder::ColumnMajor,
        )
        .unwrap()
    }
}

#[cfg(feature = "cuda")]
fn cuda_public_svdvals_matches_cpu_generic(wide: bool) {
    let Some(()) = with_cuda_ctx(|ctx| {
        let mut cpu_ctx = CpuContext::new(1);
        let a_cpu = public_cuda_svd_matrix(wide);
        let expected = svdvals(&mut cpu_ctx, &a_cpu).unwrap();
        let a_gpu = a_cpu
            .to_memory_space_async(tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 })
            .unwrap();

        let got = svdvals(ctx, &a_gpu).unwrap();
        assert_eq!(
            got.logical_memory_space(),
            tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 }
        );
        assert_eq!(got.dims(), expected.dims());
        assert_close_slice(
            if wide {
                "cuda public svdvals wide"
            } else {
                "cuda public svdvals tall"
            },
            &tensor_data_on_cpu(&got),
            &tensor_data(&expected),
            2048.0 * f64::EPSILON,
        );
    }) else {
        return;
    };
}

#[cfg(feature = "cuda")]
fn cuda_public_svd_reconstructs_generic(wide: bool) {
    let Some(()) = with_cuda_ctx(|ctx| {
        let mut cpu_ctx = CpuContext::new(1);
        let a_cpu = public_cuda_svd_matrix(wide);
        let expected = svd(&mut cpu_ctx, &a_cpu, None).unwrap();
        let a_gpu = a_cpu
            .to_memory_space_async(tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 })
            .unwrap();

        let got = svd(ctx, &a_gpu, None).unwrap();
        assert_eq!(
            got.u.logical_memory_space(),
            tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 }
        );
        assert_eq!(
            got.s.logical_memory_space(),
            tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 }
        );
        assert_eq!(
            got.vt.logical_memory_space(),
            tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 }
        );
        assert_eq!(got.u.dims(), expected.u.dims());
        assert_eq!(got.s.dims(), expected.s.dims());
        assert_eq!(got.vt.dims(), expected.vt.dims());
        assert_close_slice(
            if wide {
                "cuda public svd singular values wide"
            } else {
                "cuda public svd singular values tall"
            },
            &tensor_data_on_cpu(&got.s),
            &tensor_data(&expected.s),
            2048.0 * f64::EPSILON,
        );

        let reconstructed = reconstruct_thin_svd_col_major(
            &tensor_data_on_cpu(&got.u),
            a_cpu.dims()[0],
            got.s.dims()[0],
            &tensor_data_on_cpu(&got.s),
            &tensor_data_on_cpu(&got.vt),
            a_cpu.dims()[1],
        );
        assert_close_slice(
            if wide {
                "cuda public svd reconstruction wide"
            } else {
                "cuda public svd reconstruction tall"
            },
            &reconstructed,
            &tensor_data(&a_cpu),
            4096.0 * f64::EPSILON,
        );
    }) else {
        return;
    };
}

#[cfg(feature = "cuda")]
fn cuda_public_svd_cutoff_preserves_zero_fill_semantics() {
    let Some(()) = with_cuda_ctx(|ctx| {
        let a_cpu = Tensor::from_slice(
            &[3.0_f64, 0.0, 0.0, 0.25],
            &[2, 2],
            MemoryOrder::ColumnMajor,
        )
        .unwrap();
        let a_gpu = a_cpu
            .to_memory_space_async(tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 })
            .unwrap();
        let opts = SvdOptions {
            max_rank: Some(2),
            cutoff: Some(0.5),
        };

        let result = svd(ctx, &a_gpu, Some(&opts)).unwrap();
        assert_eq!(
            result.u.logical_memory_space(),
            tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 }
        );
        assert_eq!(
            result.s.logical_memory_space(),
            tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 }
        );
        assert_eq!(
            result.vt.logical_memory_space(),
            tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 }
        );
        assert_eq!(result.u.dims(), &[2, 2]);
        assert_eq!(result.s.dims(), &[2]);
        assert_eq!(result.vt.dims(), &[2, 2]);
        assert_eq!(tensor_data_on_cpu(&result.s), vec![3.0, 0.0]);

        let u = tensor_data_on_cpu(&result.u);
        let vt = tensor_data_on_cpu(&result.vt);
        assert_eq!(u[2], 0.0);
        assert_eq!(u[3], 0.0);
        assert_eq!(vt[1], 0.0);
        assert_eq!(vt[3], 0.0);
    }) else {
        return;
    };
}

#[cfg(feature = "cuda")]
fn cuda_public_pinv_matches_cpu_small_real_matrix() {
    let Some(()) = with_cuda_ctx(|ctx| {
        let mut cpu_ctx = CpuContext::new(1);
        let a_cpu = Tensor::from_slice(
            &[1.0_f64, 2.0, 3.0, 4.0, 5.0, 7.0],
            &[3, 2],
            MemoryOrder::ColumnMajor,
        )
        .unwrap();
        let expected = pinv(&mut cpu_ctx, &a_cpu, None).unwrap();
        let a_gpu = a_cpu
            .to_memory_space_async(tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 })
            .unwrap();

        let got = pinv(ctx, &a_gpu, None).unwrap();
        assert_eq!(
            got.logical_memory_space(),
            tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 }
        );
        assert_eq!(got.dims(), expected.dims());
        assert_close_slice(
            "cuda public pinv small real matrix",
            &tensor_data_on_cpu(&got),
            &tensor_data_on_cpu(&expected),
            4096.0 * f64::EPSILON,
        );
    }) else {
        return;
    };
}

#[cfg(feature = "cuda")]
fn cuda_public_pinv_matches_cpu_rank_deficient_real_matrix() {
    let Some(()) = with_cuda_ctx(|ctx| {
        let mut cpu_ctx = CpuContext::new(1);
        let a_cpu = Tensor::from_slice(
            &[1.0_f64, 0.0, 0.0, 0.0, 0.0, 0.0],
            &[3, 2],
            MemoryOrder::ColumnMajor,
        )
        .unwrap();
        let expected = pinv(&mut cpu_ctx, &a_cpu, None).unwrap();
        let a_gpu = a_cpu
            .to_memory_space_async(tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 })
            .unwrap();

        let got = pinv(ctx, &a_gpu, None).unwrap();
        assert_eq!(
            got.logical_memory_space(),
            tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 }
        );
        assert_eq!(got.dims(), expected.dims());
        assert_close_slice(
            "cuda public pinv rank-deficient real matrix",
            &tensor_data_on_cpu(&got),
            &tensor_data_on_cpu(&expected),
            4096.0 * f64::EPSILON,
        );
    }) else {
        return;
    };
}

#[cfg(feature = "cuda")]
fn cuda_public_inv_matches_cpu_small_real_matrix() {
    let Some(()) = with_cuda_ctx(|ctx| {
        let mut cpu_ctx = CpuContext::new(1);
        let a_cpu =
            Tensor::from_slice(&[3.0_f64, 1.0, 1.0, 2.0], &[2, 2], MemoryOrder::ColumnMajor)
                .unwrap();
        let expected = inv(&mut cpu_ctx, &a_cpu).unwrap();
        let a_gpu = a_cpu
            .to_memory_space_async(tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 })
            .unwrap();

        let got = inv(ctx, &a_gpu).unwrap();
        assert_eq!(
            got.logical_memory_space(),
            tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 }
        );
        assert_eq!(got.dims(), expected.dims());
        assert_close_slice(
            "cuda public inv small real matrix",
            &tensor_data_on_cpu(&got),
            &tensor_data_on_cpu(&expected),
            4096.0 * f64::EPSILON,
        );
    }) else {
        return;
    };
}

#[cfg(feature = "cuda")]
fn cuda_public_inv_ex_matches_cpu_mixed_batch_real_matrix() {
    let Some(()) = with_cuda_ctx(|ctx| {
        let mut cpu_ctx = CpuContext::new(1);
        let a_cpu = Tensor::from_slice(
            &[
                3.0_f64, 1.0, 1.0, 2.0, //
                1.0, 0.0, 0.0, 1.0,
            ],
            &[2, 2, 2],
            MemoryOrder::ColumnMajor,
        )
        .unwrap();
        let expected = inv_ex(&mut cpu_ctx, &a_cpu).unwrap();
        let a_gpu = a_cpu
            .to_memory_space_async(tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 })
            .unwrap();

        let got = inv_ex(ctx, &a_gpu).unwrap();
        assert_eq!(
            got.inverse.logical_memory_space(),
            tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 }
        );
        assert_eq!(got.inverse.dims(), expected.inverse.dims());
        assert_eq!(got.info, expected.info);
        assert_close_slice(
            "cuda public inv_ex mixed batch real matrix",
            &tensor_data_on_cpu(&got.inverse),
            &tensor_data_on_cpu(&expected.inverse),
            4096.0 * f64::EPSILON,
        );
    }) else {
        return;
    };
}

#[cfg(feature = "cuda")]
fn cuda_public_fro_norm_matches_cpu_small_real_matrix() {
    let Some(()) = with_cuda_ctx(|ctx| {
        let mut cpu_ctx = CpuContext::new(1);
        let a_cpu = Tensor::from_slice(
            &[1.0_f64, 2.0, 3.0, 4.0, 5.0, 7.0],
            &[3, 2],
            MemoryOrder::ColumnMajor,
        )
        .unwrap();
        let expected = norm(&mut cpu_ctx, &a_cpu, NormKind::Fro).unwrap();
        let a_gpu = a_cpu
            .to_memory_space_async(tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 })
            .unwrap();

        let got = norm(ctx, &a_gpu, NormKind::Fro).unwrap();
        assert_eq!(
            got.logical_memory_space(),
            tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 }
        );
        assert_eq!(got.dims(), expected.dims());
        assert_close_slice(
            "cuda public fro norm small real matrix",
            &tensor_data_on_cpu(&got),
            &tensor_data_on_cpu(&expected),
            4096.0 * f64::EPSILON,
        );
    }) else {
        return;
    };
}

#[cfg(feature = "cuda")]
fn cuda_public_lp_norm_matches_cpu_vector() {
    let Some(()) = with_cuda_ctx(|ctx| {
        let mut cpu_ctx = CpuContext::new(1);
        let a_cpu =
            Tensor::from_slice(&[2.0_f64, -3.0, 4.0], &[3], MemoryOrder::ColumnMajor).unwrap();
        let expected = norm(&mut cpu_ctx, &a_cpu, NormKind::Lp(3.0)).unwrap();
        let a_gpu = a_cpu
            .to_memory_space_async(tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 })
            .unwrap();

        let got = norm(ctx, &a_gpu, NormKind::Lp(3.0)).unwrap();
        assert_eq!(
            got.logical_memory_space(),
            tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 }
        );
        assert_eq!(got.dims(), expected.dims());
        assert_close_slice(
            "cuda public lp norm vector",
            &tensor_data_on_cpu(&got),
            &tensor_data_on_cpu(&expected),
            4096.0 * f64::EPSILON,
        );
    }) else {
        return;
    };
}

#[cfg(feature = "cuda")]
fn cuda_public_lu_no_pivot_rejects_gpu_tensor_before_host_slice_fallback() {
    let Some(()) = with_cuda_ctx(|ctx| {
        let a_gpu =
            Tensor::from_slice(&[2.0_f64, 1.0, 1.0, 3.0], &[2, 2], MemoryOrder::ColumnMajor)
                .unwrap()
                .to_memory_space_async(tenferro_device::LogicalMemorySpace::GpuMemory {
                    device_id: 0,
                })
                .unwrap();

        let err = lu(ctx, &a_gpu, LuPivot::NoPivot).unwrap_err();
        assert!(matches!(err, tenferro_device::Error::DeviceError(_)));
        assert!(
            err.to_string()
                .contains("NoPivot LU is only implemented for main-memory tensors"),
            "lu(NoPivot) should fail before host-slice extraction, got: {err}"
        );
    }) else {
        return;
    };
}

#[cfg(feature = "cuda")]
fn cuda_public_matrix_power_matches_cpu_small_real_matrix_f64() {
    let Some(()) = with_cuda_ctx(|ctx| {
        let mut cpu_ctx = CpuContext::new(1);
        let a_cpu =
            Tensor::from_slice(&[2.0_f64, 0.0, 0.0, 3.0], &[2, 2], MemoryOrder::ColumnMajor)
                .unwrap();
        let a_gpu = a_cpu
            .to_memory_space_async(tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 })
            .unwrap();

        let expected_pos = matrix_power(&mut cpu_ctx, &a_cpu, 3).unwrap();
        let got_pos = matrix_power(ctx, &a_gpu, 3).unwrap();
        assert_eq!(
            got_pos.logical_memory_space(),
            tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 }
        );
        assert_eq!(got_pos.dims(), expected_pos.dims());
        assert_close_real_slice(
            "cuda public matrix power positive exponent",
            &tensor_data_on_cpu(&got_pos),
            &tensor_data_on_cpu(&expected_pos),
            2048.0_f64 * f64::epsilon(),
        );

        let expected_neg = matrix_power(&mut cpu_ctx, &a_cpu, -1).unwrap();
        let got_neg = matrix_power(ctx, &a_gpu, -1).unwrap();
        assert_eq!(
            got_neg.logical_memory_space(),
            tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 }
        );
        assert_eq!(got_neg.dims(), expected_neg.dims());
        assert_close_real_slice(
            "cuda public matrix power negative exponent",
            &tensor_data_on_cpu(&got_neg),
            &tensor_data_on_cpu(&expected_neg),
            2048.0_f64 * f64::epsilon(),
        );
    }) else {
        return;
    };
}

#[cfg(feature = "cuda")]
fn cuda_public_matrix_power_matches_cpu_small_complex_matrix_c64() {
    let Some(()) = with_cuda_ctx(|ctx| {
        let mut cpu_ctx = CpuContext::new(1);
        let a_cpu = Tensor::from_slice(
            &[
                Complex::new(1.0_f64, 1.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(2.0, -1.0),
            ],
            &[2, 2],
            MemoryOrder::ColumnMajor,
        )
        .unwrap();
        let a_gpu = a_cpu
            .to_memory_space_async(tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 })
            .unwrap();

        let expected_pos = matrix_power(&mut cpu_ctx, &a_cpu, 2).unwrap();
        let got_pos = matrix_power(ctx, &a_gpu, 2).unwrap();
        assert_eq!(
            got_pos.logical_memory_space(),
            tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 }
        );
        assert_eq!(got_pos.dims(), expected_pos.dims());
        assert_close_complex_slice(
            "cuda public matrix power complex positive exponent",
            &tensor_data_on_cpu(&got_pos),
            &tensor_data_on_cpu(&expected_pos),
            4096.0_f64 * f64::EPSILON,
        );

        let expected_neg = matrix_power(&mut cpu_ctx, &a_cpu, -1).unwrap();
        let got_neg = matrix_power(ctx, &a_gpu, -1).unwrap();
        assert_eq!(
            got_neg.logical_memory_space(),
            tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 }
        );
        assert_eq!(got_neg.dims(), expected_neg.dims());
        assert_close_complex_slice(
            "cuda public matrix power complex negative exponent",
            &tensor_data_on_cpu(&got_neg),
            &tensor_data_on_cpu(&expected_neg),
            4096.0_f64 * f64::EPSILON,
        );
    }) else {
        return;
    };
}

#[cfg(feature = "cuda")]
fn cpu_context_matrix_exp_reject_gpu_tensors_before_host_slice_fallback_impl() {
    let Some(()) = with_cuda_ctx(|_| {
        let mut cpu_ctx = CpuContext::new(1);
        let a_gpu =
            Tensor::from_slice(&[2.0_f64, 1.0, 1.0, 3.0], &[2, 2], MemoryOrder::ColumnMajor)
                .unwrap()
                .to_memory_space_async(tenferro_device::LogicalMemorySpace::GpuMemory {
                    device_id: 0,
                })
                .unwrap();

        let exp_err = matrix_exp(&mut cpu_ctx, &a_gpu).unwrap_err();
        assert!(matches!(exp_err, tenferro_device::Error::DeviceError(_)));
        assert!(
            exp_err
                .to_string()
                .contains("matrix_exp is only implemented for main-memory tensors"),
            "matrix_exp should fail before host-slice extraction, got: {exp_err}"
        );
    }) else {
        return;
    };
}

#[cfg(feature = "cuda")]
fn cuda_public_det_matches_cpu_f32_impl() {
    let Some(()) = with_cuda_ctx(|ctx| {
        let mut cpu_ctx = CpuContext::new(1);
        let a_cpu =
            Tensor::from_slice(&[2.0_f32, 1.0, 1.0, 3.0], &[2, 2], MemoryOrder::ColumnMajor)
                .unwrap();
        let expected = det(&mut cpu_ctx, &a_cpu).unwrap();
        let a_gpu = a_cpu
            .to_memory_space_async(tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 })
            .unwrap();

        let got = det(ctx, &a_gpu).unwrap();
        assert_eq!(
            got.logical_memory_space(),
            tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 }
        );
        assert_eq!(got.dims(), expected.dims());
        assert_close_real_slice(
            "cuda public det f32",
            &tensor_data_on_cpu(&got),
            &tensor_data_on_cpu(&expected),
            2048.0_f32 * f32::EPSILON,
        );
    }) else {
        return;
    };
}

#[cfg(feature = "cuda")]
fn cuda_public_det_matches_cpu_f64_impl() {
    let Some(()) = with_cuda_ctx(|ctx| {
        let mut cpu_ctx = CpuContext::new(1);
        let a_cpu =
            Tensor::from_slice(&[2.0_f64, 1.0, 1.0, 3.0], &[2, 2], MemoryOrder::ColumnMajor)
                .unwrap();
        let expected = det(&mut cpu_ctx, &a_cpu).unwrap();
        let a_gpu = a_cpu
            .to_memory_space_async(tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 })
            .unwrap();

        let got = det(ctx, &a_gpu).unwrap();
        assert_eq!(
            got.logical_memory_space(),
            tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 }
        );
        assert_eq!(got.dims(), expected.dims());
        assert_close_real_slice(
            "cuda public det f64",
            &tensor_data_on_cpu(&got),
            &tensor_data_on_cpu(&expected),
            2048.0_f64 * f64::EPSILON,
        );
    }) else {
        return;
    };
}

#[cfg(feature = "cuda")]
fn cuda_public_slogdet_matches_cpu_f32_impl() {
    let Some(()) = with_cuda_ctx(|ctx| {
        let mut cpu_ctx = CpuContext::new(1);
        let a_cpu =
            Tensor::from_slice(&[2.0_f32, 1.0, 1.0, 3.0], &[2, 2], MemoryOrder::ColumnMajor)
                .unwrap();
        let expected = slogdet(&mut cpu_ctx, &a_cpu).unwrap();
        let a_gpu = a_cpu
            .to_memory_space_async(tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 })
            .unwrap();

        let got = slogdet(ctx, &a_gpu).unwrap();
        assert_eq!(
            got.sign.logical_memory_space(),
            tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 }
        );
        assert_eq!(
            got.logabsdet.logical_memory_space(),
            tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 }
        );
        assert_eq!(got.sign.dims(), expected.sign.dims());
        assert_eq!(got.logabsdet.dims(), expected.logabsdet.dims());
        assert_close_real_slice(
            "cuda public slogdet sign f32",
            &tensor_data_on_cpu(&got.sign),
            &tensor_data_on_cpu(&expected.sign),
            2048.0_f32 * f32::EPSILON,
        );
        assert_close_real_slice(
            "cuda public slogdet logabsdet f32",
            &tensor_data_on_cpu(&got.logabsdet),
            &tensor_data_on_cpu(&expected.logabsdet),
            2048.0_f32 * f32::EPSILON,
        );
    }) else {
        return;
    };
}

#[cfg(feature = "cuda")]
fn cuda_public_slogdet_matches_cpu_f64_impl() {
    let Some(()) = with_cuda_ctx(|ctx| {
        let mut cpu_ctx = CpuContext::new(1);
        let a_cpu =
            Tensor::from_slice(&[2.0_f64, 1.0, 1.0, 3.0], &[2, 2], MemoryOrder::ColumnMajor)
                .unwrap();
        let expected = slogdet(&mut cpu_ctx, &a_cpu).unwrap();
        let a_gpu = a_cpu
            .to_memory_space_async(tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 })
            .unwrap();

        let got = slogdet(ctx, &a_gpu).unwrap();
        assert_eq!(
            got.sign.logical_memory_space(),
            tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 }
        );
        assert_eq!(
            got.logabsdet.logical_memory_space(),
            tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 }
        );
        assert_eq!(got.sign.dims(), expected.sign.dims());
        assert_eq!(got.logabsdet.dims(), expected.logabsdet.dims());
        assert_close_real_slice(
            "cuda public slogdet sign f64",
            &tensor_data_on_cpu(&got.sign),
            &tensor_data_on_cpu(&expected.sign),
            2048.0_f64 * f64::EPSILON,
        );
        assert_close_real_slice(
            "cuda public slogdet logabsdet f64",
            &tensor_data_on_cpu(&got.logabsdet),
            &tensor_data_on_cpu(&expected.logabsdet),
            2048.0_f64 * f64::EPSILON,
        );
    }) else {
        return;
    };
}

#[cfg(feature = "cuda")]
fn cuda_public_lu_factor_matches_cpu_complex32_impl() {
    let Some(()) = with_cuda_ctx(|ctx| {
        let mut cpu_ctx = CpuContext::new(1);
        let a_cpu = Tensor::from_slice(
            &[
                Complex::new(2.0_f32, 1.0),
                Complex::new(1.0, -1.0),
                Complex::new(3.0, 0.5),
                Complex::new(4.0, -2.0),
            ],
            &[2, 2],
            MemoryOrder::ColumnMajor,
        )
        .unwrap();
        let expected = lu_factor(&mut cpu_ctx, &a_cpu).unwrap();
        let a_gpu = a_cpu
            .to_memory_space_async(tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 })
            .unwrap();

        let got = lu_factor(ctx, &a_gpu).unwrap();
        assert_eq!(
            got.factors.logical_memory_space(),
            tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 }
        );
        assert_eq!(got.factors.dims(), expected.factors.dims());
        assert_eq!(got.pivots, expected.pivots);
        assert_close_complex_slice(
            "cuda public lu_factor complex32 factors",
            &tensor_data_on_cpu(&got.factors),
            &tensor_data_on_cpu(&expected.factors),
            4096.0_f32 * f32::EPSILON,
        );
    }) else {
        return;
    };
}

#[cfg(feature = "cuda")]
fn cuda_public_lu_factor_ex_matches_cpu_mixed_batch_f64_impl() {
    let Some(()) = with_cuda_ctx(|ctx| {
        let mut cpu_ctx = CpuContext::new(1);
        let a_cpu = Tensor::from_slice(
            &[
                1.0_f64, 0.0, 0.0, 1.0, //
                1.0, 2.0, 2.0, 4.0,
            ],
            &[2, 2, 2],
            MemoryOrder::ColumnMajor,
        )
        .unwrap();
        let expected = lu_factor_ex(&mut cpu_ctx, &a_cpu).unwrap();
        let a_gpu = a_cpu
            .to_memory_space_async(tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 })
            .unwrap();

        let got = lu_factor_ex(ctx, &a_gpu).unwrap();
        assert_eq!(
            got.factors.logical_memory_space(),
            tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 }
        );
        assert_eq!(got.factors.dims(), expected.factors.dims());
        assert_eq!(got.pivots, expected.pivots);
        assert_eq!(got.info, expected.info);
        assert_eq!(
            tensor_data_on_cpu(&got.factors),
            tensor_data_on_cpu(&expected.factors)
        );
    }) else {
        return;
    };
}

#[cfg(feature = "cuda")]
fn cuda_public_solve_ex_matches_cpu_generic() {
    let Some(()) = with_cuda_ctx(|ctx| {
        let mut cpu_ctx = CpuContext::new(1);
        let a_cpu = Tensor::from_slice(
            &[
                1.0_f64, 0.0, 0.0, 1.0, //
                1.0, 2.0, 2.0, 4.0,
            ],
            &[2, 2, 2],
            MemoryOrder::ColumnMajor,
        )
        .unwrap();
        let b_cpu = Tensor::from_slice(
            &[3.0_f64, -1.0, 1.0, 1.0],
            &[2, 2],
            MemoryOrder::ColumnMajor,
        )
        .unwrap();
        let expected = solve_ex(&mut cpu_ctx, &a_cpu, &b_cpu).unwrap();
        let a_gpu = a_cpu
            .to_memory_space_async(tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 })
            .unwrap();
        let b_gpu = b_cpu
            .to_memory_space_async(tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 })
            .unwrap();

        let got = solve_ex(ctx, &a_gpu, &b_gpu).unwrap();
        assert_eq!(
            got.solution.logical_memory_space(),
            tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 }
        );
        assert_eq!(got.info, expected.info);
        assert_close_slice(
            "cuda public solve_ex",
            &tensor_data_on_cpu(&got.solution),
            &tensor_data(&expected.solution),
            256.0 * f64::EPSILON,
        );
    }) else {
        return;
    };
}

#[cfg(feature = "cuda")]
fn cuda_public_solve_ex_matches_cpu_complex32_impl() {
    let Some(()) = with_cuda_ctx(|ctx| {
        let mut cpu_ctx = CpuContext::new(1);
        let zero = 0.0_f32;
        let one = 1.0_f32;
        let two = one + one;
        let three = two + one;
        let half = 0.5_f32;
        let quarter = 0.25_f32;
        let minus_one = -one;

        let a_cpu = Tensor::from_slice(
            &[
                Complex::new(one, zero),
                Complex::new(zero, zero),
                Complex::new(zero, zero),
                Complex::new(one, zero),
                Complex::new(one, zero),
                Complex::new(two, zero),
                Complex::new(two, zero),
                Complex::new(4.0_f32, zero),
            ],
            &[2, 2, 2],
            MemoryOrder::ColumnMajor,
        )
        .unwrap();
        let b_cpu = Tensor::from_slice(
            &[
                Complex::new(three, minus_one),
                Complex::new(one, half),
                Complex::new(-one, -quarter),
                Complex::new(-two, half),
            ],
            &[2, 2],
            MemoryOrder::ColumnMajor,
        )
        .unwrap();
        let expected = solve_ex(&mut cpu_ctx, &a_cpu, &b_cpu).unwrap();
        let a_gpu = a_cpu
            .to_memory_space_async(tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 })
            .unwrap();
        let b_gpu = b_cpu
            .to_memory_space_async(tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 })
            .unwrap();

        let got = solve_ex(ctx, &a_gpu, &b_gpu).unwrap();
        assert_eq!(
            got.solution.logical_memory_space(),
            tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 }
        );
        assert_eq!(got.solution.dims(), expected.solution.dims());
        assert_eq!(got.info, expected.info);
        assert_close_complex_slice(
            "cuda public complex solve_ex",
            &tensor_data_on_cpu(&got.solution),
            &tensor_data_on_cpu(&expected.solution),
            4096.0_f32 * f32::EPSILON,
        );
    }) else {
        return;
    };
}

#[cfg(feature = "cuda")]
fn cuda_public_solve_ex_matches_cpu_complex64_impl() {
    let Some(()) = with_cuda_ctx(|ctx| {
        let mut cpu_ctx = CpuContext::new(1);
        let zero = 0.0_f64;
        let one = 1.0_f64;
        let two = one + one;
        let three = two + one;
        let half = 0.5_f64;
        let quarter = 0.25_f64;
        let minus_one = -one;

        let a_cpu = Tensor::from_slice(
            &[
                Complex::new(one, zero),
                Complex::new(zero, zero),
                Complex::new(zero, zero),
                Complex::new(one, zero),
                Complex::new(one, zero),
                Complex::new(two, zero),
                Complex::new(two, zero),
                Complex::new(4.0_f64, zero),
            ],
            &[2, 2, 2],
            MemoryOrder::ColumnMajor,
        )
        .unwrap();
        let b_cpu = Tensor::from_slice(
            &[
                Complex::new(three, minus_one),
                Complex::new(one, half),
                Complex::new(-one, -quarter),
                Complex::new(-two, half),
            ],
            &[2, 2],
            MemoryOrder::ColumnMajor,
        )
        .unwrap();
        let expected = solve_ex(&mut cpu_ctx, &a_cpu, &b_cpu).unwrap();
        let a_gpu = a_cpu
            .to_memory_space_async(tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 })
            .unwrap();
        let b_gpu = b_cpu
            .to_memory_space_async(tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 })
            .unwrap();

        let got = solve_ex(ctx, &a_gpu, &b_gpu).unwrap();
        assert_eq!(
            got.solution.logical_memory_space(),
            tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 }
        );
        assert_eq!(got.solution.dims(), expected.solution.dims());
        assert_eq!(got.info, expected.info);
        assert_close_complex_slice(
            "cuda public complex solve_ex",
            &tensor_data_on_cpu(&got.solution),
            &tensor_data_on_cpu(&expected.solution),
            4096.0_f64 * f64::EPSILON,
        );
    }) else {
        return;
    };
}

#[cfg(feature = "cuda")]
fn cuda_public_solve_triangular_matches_cpu_complex32_impl() {
    let Some(()) = with_cuda_ctx(|ctx| {
        let mut cpu_ctx = CpuContext::new(1);
        let a_cpu = Tensor::from_slice(
            &[
                Complex::new(2.0_f32, 1.0),
                Complex::new(0.0, 0.0),
                Complex::new(1.0, -2.0),
                Complex::new(3.0, 4.0),
            ],
            &[2, 2],
            MemoryOrder::ColumnMajor,
        )
        .unwrap();
        let b_cpu = Tensor::from_slice(
            &[Complex::new(5.0_f32, -1.0), Complex::new(2.0, 3.0)],
            &[2],
            MemoryOrder::ColumnMajor,
        )
        .unwrap();
        let expected = solve_triangular(&mut cpu_ctx, &a_cpu, &b_cpu, true).unwrap();
        let a_gpu = a_cpu
            .to_memory_space_async(tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 })
            .unwrap();
        let b_gpu = b_cpu
            .to_memory_space_async(tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 })
            .unwrap();

        let got = solve_triangular(ctx, &a_gpu, &b_gpu, true).unwrap();
        assert_eq!(
            got.logical_memory_space(),
            tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 }
        );
        assert_eq!(got.dims(), expected.dims());
        assert_close_complex_slice(
            "cuda public solve_triangular complex32",
            &tensor_data_on_cpu(&got),
            &tensor_data_on_cpu(&expected),
            4096.0_f32 * f32::EPSILON,
        );
    }) else {
        return;
    };
}

#[cfg(feature = "cuda")]
fn cuda_public_solve_triangular_matches_cpu_complex64_impl() {
    let Some(()) = with_cuda_ctx(|ctx| {
        let mut cpu_ctx = CpuContext::new(1);
        let a_cpu = Tensor::from_slice(
            &[
                Complex::new(2.0_f64, 1.0),
                Complex::new(0.0, 0.0),
                Complex::new(1.0, -2.0),
                Complex::new(3.0, 4.0),
            ],
            &[2, 2],
            MemoryOrder::ColumnMajor,
        )
        .unwrap();
        let b_cpu = Tensor::from_slice(
            &[Complex::new(5.0_f64, -1.0), Complex::new(2.0, 3.0)],
            &[2],
            MemoryOrder::ColumnMajor,
        )
        .unwrap();
        let expected = solve_triangular(&mut cpu_ctx, &a_cpu, &b_cpu, true).unwrap();
        let a_gpu = a_cpu
            .to_memory_space_async(tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 })
            .unwrap();
        let b_gpu = b_cpu
            .to_memory_space_async(tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 })
            .unwrap();

        let got = solve_triangular(ctx, &a_gpu, &b_gpu, true).unwrap();
        assert_eq!(
            got.logical_memory_space(),
            tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 }
        );
        assert_eq!(got.dims(), expected.dims());
        assert_close_complex_slice(
            "cuda public solve_triangular complex64",
            &tensor_data_on_cpu(&got),
            &tensor_data_on_cpu(&expected),
            4096.0_f64 * f64::EPSILON,
        );
    }) else {
        return;
    };
}

#[cfg(feature = "cuda")]
fn cuda_public_cholesky_ex_matches_cpu_complex32_impl() {
    let Some(()) = with_cuda_ctx(|ctx| {
        let mut cpu_ctx = CpuContext::new(1);
        let zero = 0.0_f32;
        let one = 1.0_f32;
        let two = one + one;
        let three = two + one;
        let four = two + two;
        let half = 0.5_f32;
        let minus_half = -half;

        let a_cpu = Tensor::from_slice(
            &[
                Complex::new(four, zero),
                Complex::new(one, half),
                Complex::new(one, minus_half),
                Complex::new(three, zero),
                Complex::new(one, zero),
                Complex::new(two, minus_half),
                Complex::new(two, half),
                Complex::new(one, zero),
            ],
            &[2, 2, 2],
            MemoryOrder::ColumnMajor,
        )
        .unwrap();
        let expected = cholesky_ex(&mut cpu_ctx, &a_cpu).unwrap();
        let a_gpu = a_cpu
            .to_memory_space_async(tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 })
            .unwrap();

        let got = cholesky_ex(ctx, &a_gpu).unwrap();
        assert_eq!(
            got.l.logical_memory_space(),
            tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 }
        );
        assert_eq!(got.l.dims(), expected.l.dims());
        assert_eq!(got.info, expected.info);
        assert_close_complex_slice(
            "cuda public complex cholesky_ex",
            &tensor_data_on_cpu(&got.l),
            &tensor_data_on_cpu(&expected.l),
            4096.0_f32 * f32::EPSILON,
        );
    }) else {
        return;
    };
}

#[cfg(feature = "cuda")]
fn cuda_public_cholesky_ex_matches_cpu_complex64_impl() {
    let Some(()) = with_cuda_ctx(|ctx| {
        let mut cpu_ctx = CpuContext::new(1);
        let zero = 0.0_f64;
        let one = 1.0_f64;
        let two = one + one;
        let three = two + one;
        let four = two + two;
        let half = 0.5_f64;
        let minus_half = -half;

        let a_cpu = Tensor::from_slice(
            &[
                Complex::new(four, zero),
                Complex::new(one, half),
                Complex::new(one, minus_half),
                Complex::new(three, zero),
                Complex::new(one, zero),
                Complex::new(two, minus_half),
                Complex::new(two, half),
                Complex::new(one, zero),
            ],
            &[2, 2, 2],
            MemoryOrder::ColumnMajor,
        )
        .unwrap();
        let expected = cholesky_ex(&mut cpu_ctx, &a_cpu).unwrap();
        let a_gpu = a_cpu
            .to_memory_space_async(tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 })
            .unwrap();

        let got = cholesky_ex(ctx, &a_gpu).unwrap();
        assert_eq!(
            got.l.logical_memory_space(),
            tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 }
        );
        assert_eq!(got.l.dims(), expected.l.dims());
        assert_eq!(got.info, expected.info);
        assert_close_complex_slice(
            "cuda public complex cholesky_ex",
            &tensor_data_on_cpu(&got.l),
            &tensor_data_on_cpu(&expected.l),
            4096.0_f64 * f64::EPSILON,
        );
    }) else {
        return;
    };
}

#[cfg(feature = "cuda")]
fn cuda_public_qr_reconstructs_complex32_impl() {
    let Some(()) = with_cuda_ctx(|ctx| {
        let a_cpu = Tensor::from_slice(
            &[
                Complex::new(1.0_f32, 0.5),
                Complex::new(2.0, -1.0),
                Complex::new(3.0, 0.25),
                Complex::new(4.0, 1.5),
                Complex::new(5.0, -0.75),
                Complex::new(6.0, 0.0),
            ],
            &[3, 2],
            MemoryOrder::ColumnMajor,
        )
        .unwrap();
        let a_gpu = a_cpu
            .to_memory_space_async(tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 })
            .unwrap();

        let got = qr(ctx, &a_gpu).unwrap();
        assert_eq!(
            got.q.logical_memory_space(),
            tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 }
        );
        assert_eq!(
            got.r.logical_memory_space(),
            tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 }
        );
        assert_eq!(got.q.dims(), &[3, 2]);
        assert_eq!(got.r.dims(), &[2, 2]);

        let q = tensor_data_on_cpu(&got.q);
        let r = tensor_data_on_cpu(&got.r);
        let reconstructed = matmul_col_major_complex(&q, 3, 2, &r, 2);
        assert_close_complex_slice(
            "cuda public complex32 qr reconstruction",
            &reconstructed,
            &tensor_data_on_cpu(&a_cpu),
            4096.0_f32 * f32::EPSILON,
        );

        let gram = gram_col_major_complex(&q, 3, 2);
        let identity = vec![
            Complex::new(1.0_f32, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
        ];
        assert_close_complex_slice(
            "cuda public complex32 qr unitary",
            &gram,
            &identity,
            4096.0_f32 * f32::EPSILON,
        );
    }) else {
        return;
    };
}

#[cfg(feature = "cuda")]
fn cuda_public_qr_reconstructs_complex64_impl() {
    let Some(()) = with_cuda_ctx(|ctx| {
        let a_cpu = Tensor::from_slice(
            &[
                Complex::new(1.0_f64, 0.5),
                Complex::new(2.0, -1.0),
                Complex::new(3.0, 0.25),
                Complex::new(4.0, 1.5),
                Complex::new(5.0, -0.75),
                Complex::new(6.0, 0.0),
            ],
            &[3, 2],
            MemoryOrder::ColumnMajor,
        )
        .unwrap();
        let a_gpu = a_cpu
            .to_memory_space_async(tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 })
            .unwrap();

        let got = qr(ctx, &a_gpu).unwrap();
        assert_eq!(
            got.q.logical_memory_space(),
            tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 }
        );
        assert_eq!(
            got.r.logical_memory_space(),
            tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 }
        );
        assert_eq!(got.q.dims(), &[3, 2]);
        assert_eq!(got.r.dims(), &[2, 2]);

        let q = tensor_data_on_cpu(&got.q);
        let r = tensor_data_on_cpu(&got.r);
        let reconstructed = matmul_col_major_complex(&q, 3, 2, &r, 2);
        assert_close_complex_slice(
            "cuda public complex64 qr reconstruction",
            &reconstructed,
            &tensor_data_on_cpu(&a_cpu),
            4096.0_f64 * f64::EPSILON,
        );

        let gram = gram_col_major_complex(&q, 3, 2);
        let identity = vec![
            Complex::new(1.0_f64, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
        ];
        assert_close_complex_slice(
            "cuda public complex64 qr unitary",
            &gram,
            &identity,
            4096.0_f64 * f64::EPSILON,
        );
    }) else {
        return;
    };
}

#[cfg(feature = "cuda")]
fn cuda_public_svdvals_matches_cpu_complex32_impl(wide: bool) {
    let Some(()) = with_cuda_ctx(|ctx| {
        let mut cpu_ctx = CpuContext::new(1);
        let a_cpu = if wide {
            Tensor::from_slice(
                &[
                    Complex::new(3.0_f32, 0.5),
                    Complex::new(1.0, -0.25),
                    Complex::new(1.0, 0.25),
                    Complex::new(2.0, -1.0),
                    Complex::new(0.0, 0.5),
                    Complex::new(1.0, 0.0),
                ],
                &[2, 3],
                MemoryOrder::ColumnMajor,
            )
            .unwrap()
        } else {
            Tensor::from_slice(
                &[
                    Complex::new(3.0_f32, 0.5),
                    Complex::new(1.0, -0.25),
                    Complex::new(0.0, 0.5),
                    Complex::new(1.0, 0.25),
                    Complex::new(2.0, -1.0),
                    Complex::new(1.0, 0.0),
                ],
                &[3, 2],
                MemoryOrder::ColumnMajor,
            )
            .unwrap()
        };
        let expected = svdvals(&mut cpu_ctx, &a_cpu).unwrap();
        let a_gpu = a_cpu
            .to_memory_space_async(tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 })
            .unwrap();

        let got = svdvals(ctx, &a_gpu).unwrap();
        assert_eq!(
            got.logical_memory_space(),
            tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 }
        );
        assert_eq!(got.dims(), expected.dims());
        assert_close_real_slice(
            if wide {
                "cuda public complex32 svdvals wide"
            } else {
                "cuda public complex32 svdvals tall"
            },
            &tensor_data_on_cpu(&got),
            &tensor_data_on_cpu(&expected),
            4096.0_f32 * f32::EPSILON,
        );
    }) else {
        return;
    };
}

#[cfg(feature = "cuda")]
fn cuda_public_svdvals_matches_cpu_complex64_impl(wide: bool) {
    let Some(()) = with_cuda_ctx(|ctx| {
        let mut cpu_ctx = CpuContext::new(1);
        let a_cpu = if wide {
            Tensor::from_slice(
                &[
                    Complex::new(3.0_f64, 0.5),
                    Complex::new(1.0, -0.25),
                    Complex::new(1.0, 0.25),
                    Complex::new(2.0, -1.0),
                    Complex::new(0.0, 0.5),
                    Complex::new(1.0, 0.0),
                ],
                &[2, 3],
                MemoryOrder::ColumnMajor,
            )
            .unwrap()
        } else {
            Tensor::from_slice(
                &[
                    Complex::new(3.0_f64, 0.5),
                    Complex::new(1.0, -0.25),
                    Complex::new(0.0, 0.5),
                    Complex::new(1.0, 0.25),
                    Complex::new(2.0, -1.0),
                    Complex::new(1.0, 0.0),
                ],
                &[3, 2],
                MemoryOrder::ColumnMajor,
            )
            .unwrap()
        };
        let expected = svdvals(&mut cpu_ctx, &a_cpu).unwrap();
        let a_gpu = a_cpu
            .to_memory_space_async(tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 })
            .unwrap();

        let got = svdvals(ctx, &a_gpu).unwrap();
        assert_eq!(
            got.logical_memory_space(),
            tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 }
        );
        assert_eq!(got.dims(), expected.dims());
        assert_close_real_slice(
            if wide {
                "cuda public complex64 svdvals wide"
            } else {
                "cuda public complex64 svdvals tall"
            },
            &tensor_data_on_cpu(&got),
            &tensor_data_on_cpu(&expected),
            4096.0_f64 * f64::EPSILON,
        );
    }) else {
        return;
    };
}

#[cfg(feature = "cuda")]
fn cuda_public_svd_reconstructs_complex32_impl(wide: bool) {
    let Some(()) = with_cuda_ctx(|ctx| {
        let mut cpu_ctx = CpuContext::new(1);
        let a_cpu = if wide {
            Tensor::from_slice(
                &[
                    Complex::new(3.0_f32, 0.5),
                    Complex::new(1.0, -0.25),
                    Complex::new(1.0, 0.25),
                    Complex::new(2.0, -1.0),
                    Complex::new(0.0, 0.5),
                    Complex::new(1.0, 0.0),
                ],
                &[2, 3],
                MemoryOrder::ColumnMajor,
            )
            .unwrap()
        } else {
            Tensor::from_slice(
                &[
                    Complex::new(3.0_f32, 0.5),
                    Complex::new(1.0, -0.25),
                    Complex::new(0.0, 0.5),
                    Complex::new(1.0, 0.25),
                    Complex::new(2.0, -1.0),
                    Complex::new(1.0, 0.0),
                ],
                &[3, 2],
                MemoryOrder::ColumnMajor,
            )
            .unwrap()
        };
        let expected = svd(&mut cpu_ctx, &a_cpu, None).unwrap();
        let a_gpu = a_cpu
            .to_memory_space_async(tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 })
            .unwrap();

        let got = svd(ctx, &a_gpu, None).unwrap();
        assert_eq!(
            got.u.logical_memory_space(),
            tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 }
        );
        assert_eq!(
            got.s.logical_memory_space(),
            tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 }
        );
        assert_eq!(
            got.vt.logical_memory_space(),
            tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 }
        );
        assert_eq!(got.u.dims(), expected.u.dims());
        assert_eq!(got.s.dims(), expected.s.dims());
        assert_eq!(got.vt.dims(), expected.vt.dims());
        assert_close_real_slice(
            if wide {
                "cuda public complex32 svd singular values wide"
            } else {
                "cuda public complex32 svd singular values tall"
            },
            &tensor_data_on_cpu(&got.s),
            &tensor_data_on_cpu(&expected.s),
            4096.0_f32 * f32::EPSILON,
        );

        let reconstructed = reconstruct_thin_svd_col_major_complex(
            &tensor_data_on_cpu(&got.u),
            a_cpu.dims()[0],
            got.s.dims()[0],
            &tensor_data_on_cpu(&got.s),
            &tensor_data_on_cpu(&got.vt),
            a_cpu.dims()[1],
        );
        assert_close_complex_slice(
            if wide {
                "cuda public complex32 svd reconstruction wide"
            } else {
                "cuda public complex32 svd reconstruction tall"
            },
            &reconstructed,
            &tensor_data_on_cpu(&a_cpu),
            8192.0_f32 * f32::EPSILON,
        );
    }) else {
        return;
    };
}

#[cfg(feature = "cuda")]
fn cuda_public_svd_reconstructs_complex64_impl(wide: bool) {
    let Some(()) = with_cuda_ctx(|ctx| {
        let mut cpu_ctx = CpuContext::new(1);
        let a_cpu = if wide {
            Tensor::from_slice(
                &[
                    Complex::new(3.0_f64, 0.5),
                    Complex::new(1.0, -0.25),
                    Complex::new(1.0, 0.25),
                    Complex::new(2.0, -1.0),
                    Complex::new(0.0, 0.5),
                    Complex::new(1.0, 0.0),
                ],
                &[2, 3],
                MemoryOrder::ColumnMajor,
            )
            .unwrap()
        } else {
            Tensor::from_slice(
                &[
                    Complex::new(3.0_f64, 0.5),
                    Complex::new(1.0, -0.25),
                    Complex::new(0.0, 0.5),
                    Complex::new(1.0, 0.25),
                    Complex::new(2.0, -1.0),
                    Complex::new(1.0, 0.0),
                ],
                &[3, 2],
                MemoryOrder::ColumnMajor,
            )
            .unwrap()
        };
        let expected = svd(&mut cpu_ctx, &a_cpu, None).unwrap();
        let a_gpu = a_cpu
            .to_memory_space_async(tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 })
            .unwrap();

        let got = svd(ctx, &a_gpu, None).unwrap();
        assert_eq!(
            got.u.logical_memory_space(),
            tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 }
        );
        assert_eq!(
            got.s.logical_memory_space(),
            tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 }
        );
        assert_eq!(
            got.vt.logical_memory_space(),
            tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 }
        );
        assert_eq!(got.u.dims(), expected.u.dims());
        assert_eq!(got.s.dims(), expected.s.dims());
        assert_eq!(got.vt.dims(), expected.vt.dims());
        assert_close_real_slice(
            if wide {
                "cuda public complex64 svd singular values wide"
            } else {
                "cuda public complex64 svd singular values tall"
            },
            &tensor_data_on_cpu(&got.s),
            &tensor_data_on_cpu(&expected.s),
            4096.0_f64 * f64::EPSILON,
        );

        let reconstructed = reconstruct_thin_svd_col_major_complex(
            &tensor_data_on_cpu(&got.u),
            a_cpu.dims()[0],
            got.s.dims()[0],
            &tensor_data_on_cpu(&got.s),
            &tensor_data_on_cpu(&got.vt),
            a_cpu.dims()[1],
        );
        assert_close_complex_slice(
            if wide {
                "cuda public complex64 svd reconstruction wide"
            } else {
                "cuda public complex64 svd reconstruction tall"
            },
            &reconstructed,
            &tensor_data_on_cpu(&a_cpu),
            8192.0_f64 * f64::EPSILON,
        );
    }) else {
        return;
    };
}

#[test]
fn cross_matches_right_hand_rule_with_trailing_batches() {
    let mut ctx = CpuContext::new(1);
    let a = Tensor::from_slice(
        &[
            1.0_f64, 0.0, 0.0, //
            0.0, 1.0, 0.0,
        ],
        &[3, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let b = Tensor::from_slice(
        &[
            0.0_f64, 1.0, 0.0, //
            0.0, 0.0, 1.0,
        ],
        &[3, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();

    let out = cross(&mut ctx, &a, &b).unwrap();
    assert_eq!(out.dims(), &[3, 2]);
    assert_eq!(
        tensor_data(&out),
        vec![
            0.0, 0.0, 1.0, //
            1.0, 0.0, 0.0,
        ]
    );
}

#[test]
fn cross_rejects_non_three_vector_axis() {
    let mut ctx = CpuContext::new(1);
    let a =
        Tensor::from_slice(&[1.0_f64, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap();
    let err = cross(&mut ctx, &a, &a).unwrap_err();
    assert!(matches!(err, tenferro_device::Error::InvalidArgument(_)));
}

#[test]
fn cross_supports_singleton_broadcasting_and_rejects_rank_mismatch() {
    let mut ctx = CpuContext::new(1);
    let a = Tensor::from_slice(
        &[
            1.0_f64, 0.0, 0.0, //
            0.0, 1.0, 0.0,
        ],
        &[3, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let b = Tensor::from_slice(&[0.0_f64, 1.0, 0.0], &[3, 1], MemoryOrder::ColumnMajor).unwrap();
    let out = cross(&mut ctx, &a, &b).unwrap();
    assert_eq!(out.dims(), &[3, 2]);
    assert_eq!(tensor_data(&out), vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);

    let rank_mismatch =
        Tensor::from_slice(&[0.0_f64, 1.0, 0.0], &[3], MemoryOrder::ColumnMajor).unwrap();
    let err = cross(&mut ctx, &a, &rank_mismatch).unwrap_err();
    assert!(matches!(err, tenferro_device::Error::InvalidArgument(_)));
}

#[test]
fn cross_rejects_rhs_vector_axis_and_broadcast_mismatch() {
    let mut ctx = CpuContext::new(1);
    let a = Tensor::from_slice(
        &[1.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0],
        &[3, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let bad_rhs =
        Tensor::from_slice(&[1.0_f64, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap();
    let rhs_err = cross(&mut ctx, &a, &bad_rhs).unwrap_err();
    assert!(matches!(
        rhs_err,
        tenferro_device::Error::InvalidArgument(_)
    ));

    let mismatch = Tensor::from_slice(
        &[0.0_f64, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        &[3, 3],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let mismatch_err = cross(&mut ctx, &a, &mismatch).unwrap_err();
    assert!(matches!(
        mismatch_err,
        tenferro_device::Error::InvalidArgument(_)
    ));
}

#[test]
fn householder_product_zero_tau_returns_identity_columns() {
    let mut ctx = CpuContext::new(1);
    let reflectors = Tensor::from_slice(
        &[
            1.0_f64, 0.0, 0.0, 0.0, //
            2.0, 1.0, 0.0, 0.0, //
            3.0, 4.0, 1.0, 0.0,
        ],
        &[4, 3],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let tau = Tensor::from_slice(&[0.0_f64, 0.0, 0.0], &[3], MemoryOrder::ColumnMajor).unwrap();

    let q = householder_product(&mut ctx, &reflectors, &tau).unwrap();
    assert_eq!(q.dims(), &[4, 3]);
    assert_eq!(
        tensor_data(&q),
        vec![
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0,
        ]
    );
}

#[test]
fn householder_product_applies_nonzero_reflector_and_rejects_oversized_k() {
    let mut ctx = CpuContext::new(1);
    let reflectors =
        Tensor::from_slice(&[1.0_f64, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap();
    let tau = Tensor::from_slice(&[2.0_f64], &[1], MemoryOrder::ColumnMajor).unwrap();
    let out = householder_product(&mut ctx, &reflectors, &tau).unwrap();
    assert_eq!(out.dims(), &[2, 2]);
    assert_eq!(tensor_data(&out), vec![-1.0, 0.0, 0.0, 1.0]);

    let skinny_reflectors =
        Tensor::from_slice(&[1.0_f64, 0.0], &[2, 1], MemoryOrder::ColumnMajor).unwrap();
    let oversized_tau =
        Tensor::from_slice(&[1.0_f64, 1.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    let err = householder_product(&mut ctx, &skinny_reflectors, &oversized_tau).unwrap_err();
    assert!(matches!(err, tenferro_device::Error::InvalidArgument(_)));
}

#[test]
fn householder_product_rejects_invalid_tau_shape() {
    let mut ctx = CpuContext::new(1);
    let reflectors =
        Tensor::from_slice(&[1.0_f64, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap();
    let tau = Tensor::from_slice(&[1.0_f64, 2.0, 3.0], &[3], MemoryOrder::ColumnMajor).unwrap();
    let err = householder_product(&mut ctx, &reflectors, &tau).unwrap_err();
    assert!(matches!(err, tenferro_device::Error::InvalidArgument(_)));
}

#[test]
fn householder_product_supports_batches_and_rejects_batch_mismatch() {
    let mut ctx = CpuContext::new(1);
    let reflectors = Tensor::from_slice(
        &[
            1.0_f64, 0.0, 0.0, 1.0, //
            1.0, 0.0, 0.0, 1.0,
        ],
        &[2, 2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let tau = Tensor::from_slice(&[0.0_f64, 0.0], &[1, 2], MemoryOrder::ColumnMajor).unwrap();
    let out = householder_product(&mut ctx, &reflectors, &tau).unwrap();
    assert_eq!(out.dims(), &[2, 2, 2]);

    let bad_tau = Tensor::from_slice(&[0.0_f64, 0.0], &[2, 1], MemoryOrder::ColumnMajor).unwrap();
    let err = householder_product(&mut ctx, &reflectors, &bad_tau).unwrap_err();
    assert!(matches!(err, tenferro_device::Error::InvalidArgument(_)));
}

#[test]
fn vander_supports_default_and_custom_column_counts() {
    let mut ctx = CpuContext::new(1);
    let x = Tensor::from_slice(&[2.0_f64, 3.0], &[2], MemoryOrder::ColumnMajor).unwrap();

    let default = vander(&mut ctx, &x, None, false).unwrap();
    assert_eq!(default.dims(), &[2, 2]);
    assert_eq!(tensor_data(&default), vec![2.0, 3.0, 1.0, 1.0]);

    let increasing = vander(&mut ctx, &x, Some(4), true).unwrap();
    assert_eq!(increasing.dims(), &[2, 4]);
    assert_eq!(
        tensor_data(&increasing),
        vec![
            1.0, 1.0, //
            2.0, 3.0, //
            4.0, 9.0, //
            8.0, 27.0,
        ]
    );
}

#[test]
fn vander_handles_scalar_inputs() {
    let mut ctx = CpuContext::new(1);
    let x = Tensor::from_slice(&[2.0_f64], &[], MemoryOrder::ColumnMajor).unwrap();
    let out = vander(&mut ctx, &x, Some(4), false).unwrap();
    assert_eq!(out.dims(), &[1, 4]);
    assert_eq!(tensor_data(&out), vec![8.0, 4.0, 2.0, 1.0]);
}

#[test]
fn vander_supports_batched_vectors() {
    let mut ctx = CpuContext::new(1);
    let x =
        Tensor::from_slice(&[2.0_f64, 3.0, 4.0, 5.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap();
    let out = vander(&mut ctx, &x, Some(3), false).unwrap();
    assert_eq!(out.dims(), &[2, 3, 2]);
    assert_eq!(
        tensor_data(&out),
        vec![
            4.0, 9.0, 2.0, 3.0, 1.0, 1.0, //
            16.0, 25.0, 4.0, 5.0, 1.0, 1.0,
        ]
    );
}

#[test]
fn tensorinv_inverts_tensorized_identity() {
    let mut ctx = CpuContext::new(1);
    let eye = Tensor::from_slice(
        &[
            1.0_f64, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ],
        &[4, 4],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let tensor = eye.reshape(&[2, 2, 2, 2]).unwrap();

    let inverse = tensorinv(&mut ctx, &tensor, 2).unwrap();
    assert_eq!(inverse.dims(), &[2, 2, 2, 2]);
    assert_eq!(tensor_data(&inverse), tensor_data(&tensor));
}

#[test]
#[cfg(feature = "cuda")]
fn public_cuda_svdvals_matches_cpu_for_tall_matrix() {
    cuda_public_svdvals_matches_cpu_generic(false);
}

#[test]
#[cfg(feature = "cuda")]
fn public_cuda_svdvals_matches_cpu_for_wide_matrix() {
    cuda_public_svdvals_matches_cpu_generic(true);
}

#[test]
#[cfg(feature = "cuda")]
fn public_cuda_svd_reconstructs_tall_matrix() {
    cuda_public_svd_reconstructs_generic(false);
}

#[test]
#[cfg(feature = "cuda")]
fn public_cuda_svd_reconstructs_wide_matrix() {
    cuda_public_svd_reconstructs_generic(true);
}

#[test]
#[cfg(feature = "cuda")]
fn public_cuda_svdvals_matches_cpu_for_tall_complex32_matrix() {
    cuda_public_svdvals_matches_cpu_complex32_impl(false);
}

#[test]
#[cfg(feature = "cuda")]
fn public_cuda_svdvals_matches_cpu_for_wide_complex32_matrix() {
    cuda_public_svdvals_matches_cpu_complex32_impl(true);
}

#[test]
#[cfg(feature = "cuda")]
fn public_cuda_svdvals_matches_cpu_for_tall_complex64_matrix() {
    cuda_public_svdvals_matches_cpu_complex64_impl(false);
}

#[test]
#[cfg(feature = "cuda")]
fn public_cuda_svdvals_matches_cpu_for_wide_complex64_matrix() {
    cuda_public_svdvals_matches_cpu_complex64_impl(true);
}

#[test]
#[cfg(feature = "cuda")]
fn public_cuda_svd_reconstructs_tall_complex32_matrix() {
    cuda_public_svd_reconstructs_complex32_impl(false);
}

#[test]
#[cfg(feature = "cuda")]
fn public_cuda_svd_reconstructs_wide_complex32_matrix() {
    cuda_public_svd_reconstructs_complex32_impl(true);
}

#[test]
#[cfg(feature = "cuda")]
fn public_cuda_svd_reconstructs_tall_complex64_matrix() {
    cuda_public_svd_reconstructs_complex64_impl(false);
}

#[test]
#[cfg(feature = "cuda")]
fn public_cuda_svd_reconstructs_wide_complex64_matrix() {
    cuda_public_svd_reconstructs_complex64_impl(true);
}

#[test]
#[cfg(feature = "cuda")]
fn public_cuda_svd_cutoff_preserves_zero_fill_semantics() {
    cuda_public_svd_cutoff_preserves_zero_fill_semantics();
}

#[test]
#[cfg(feature = "cuda")]
fn public_cuda_pinv_matches_cpu_for_small_real_matrix() {
    cuda_public_pinv_matches_cpu_small_real_matrix();
}

#[test]
#[cfg(feature = "cuda")]
fn public_cuda_pinv_matches_cpu_for_rank_deficient_real_matrix() {
    cuda_public_pinv_matches_cpu_rank_deficient_real_matrix();
}

#[test]
#[cfg(feature = "cuda")]
fn public_cuda_inv_matches_cpu_for_small_real_matrix() {
    cuda_public_inv_matches_cpu_small_real_matrix();
}

#[test]
#[cfg(feature = "cuda")]
fn public_cuda_inv_ex_matches_cpu_for_mixed_batch_real_matrix() {
    cuda_public_inv_ex_matches_cpu_mixed_batch_real_matrix();
}

#[test]
#[cfg(feature = "cuda")]
fn public_cuda_fro_norm_matches_cpu_for_small_real_matrix() {
    cuda_public_fro_norm_matches_cpu_small_real_matrix();
}

#[test]
#[cfg(feature = "cuda")]
fn public_cuda_lp_norm_matches_cpu_for_small_real_vector() {
    cuda_public_lp_norm_matches_cpu_vector();
}

#[test]
#[cfg(feature = "cuda")]
fn public_cuda_lu_no_pivot_rejects_gpu_tensor_before_host_slice_fallback() {
    cuda_public_lu_no_pivot_rejects_gpu_tensor_before_host_slice_fallback();
}

#[test]
#[cfg(feature = "cuda")]
fn public_cuda_matrix_power_matches_cpu_for_small_real_matrix_f64() {
    cuda_public_matrix_power_matches_cpu_small_real_matrix_f64();
}

#[test]
#[cfg(feature = "cuda")]
fn public_cuda_matrix_power_matches_cpu_for_small_complex_matrix_c64() {
    cuda_public_matrix_power_matches_cpu_small_complex_matrix_c64();
}

#[test]
#[cfg(feature = "cuda")]
fn cpu_context_matrix_exp_reject_gpu_tensors_before_host_slice_fallback() {
    cpu_context_matrix_exp_reject_gpu_tensors_before_host_slice_fallback_impl();
}

#[test]
#[cfg(feature = "cuda")]
fn public_cuda_det_matches_cpu_for_small_matrix_f32() {
    cuda_public_det_matches_cpu_f32_impl();
}

#[test]
#[cfg(feature = "cuda")]
fn public_cuda_det_matches_cpu_for_small_matrix_f64() {
    cuda_public_det_matches_cpu_f64_impl();
}

#[test]
#[cfg(feature = "cuda")]
fn public_cuda_slogdet_matches_cpu_for_small_matrix_f32() {
    cuda_public_slogdet_matches_cpu_f32_impl();
}

#[test]
#[cfg(feature = "cuda")]
fn public_cuda_slogdet_matches_cpu_for_small_matrix_f64() {
    cuda_public_slogdet_matches_cpu_f64_impl();
}

#[test]
#[cfg(feature = "cuda")]
fn public_cuda_lu_factor_matches_cpu_for_small_complex32_matrix() {
    cuda_public_lu_factor_matches_cpu_complex32_impl();
}

#[test]
#[cfg(feature = "cuda")]
fn public_cuda_lu_factor_ex_matches_cpu_for_mixed_batch_f64() {
    cuda_public_lu_factor_ex_matches_cpu_mixed_batch_f64_impl();
}

#[test]
#[cfg(feature = "cuda")]
fn public_cuda_solve_ex_matches_cpu_for_mixed_batch() {
    cuda_public_solve_ex_matches_cpu_generic();
}

#[test]
#[cfg(feature = "cuda")]
fn public_cuda_solve_ex_matches_cpu_for_mixed_batch_complex32() {
    cuda_public_solve_ex_matches_cpu_complex32_impl();
}

#[test]
#[cfg(feature = "cuda")]
fn public_cuda_solve_ex_matches_cpu_for_mixed_batch_complex64() {
    cuda_public_solve_ex_matches_cpu_complex64_impl();
}

#[test]
#[cfg(feature = "cuda")]
fn public_cuda_solve_triangular_matches_cpu_for_small_complex32_matrix() {
    cuda_public_solve_triangular_matches_cpu_complex32_impl();
}

#[test]
#[cfg(feature = "cuda")]
fn public_cuda_solve_triangular_matches_cpu_for_small_complex64_matrix() {
    cuda_public_solve_triangular_matches_cpu_complex64_impl();
}

#[test]
#[cfg(feature = "cuda")]
fn public_cuda_cholesky_ex_matches_cpu_for_mixed_batch_complex32() {
    cuda_public_cholesky_ex_matches_cpu_complex32_impl();
}

#[test]
#[cfg(feature = "cuda")]
fn public_cuda_cholesky_ex_matches_cpu_for_mixed_batch_complex64() {
    cuda_public_cholesky_ex_matches_cpu_complex64_impl();
}

#[test]
#[cfg(feature = "cuda")]
fn public_cuda_qr_reconstructs_complex32_matrix() {
    cuda_public_qr_reconstructs_complex32_impl();
}

#[test]
#[cfg(feature = "cuda")]
fn public_cuda_qr_reconstructs_complex64_matrix() {
    cuda_public_qr_reconstructs_complex64_impl();
}

#[test]
fn tensorinv_rejects_non_square_partition() {
    let mut ctx = CpuContext::new(1);
    let tensor = Tensor::from_slice(
        &[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[1, 2, 3],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let err = tensorinv(&mut ctx, &tensor, 1).unwrap_err();
    assert!(matches!(err, tenferro_device::Error::InvalidArgument(_)));
}

#[test]
fn tensorinv_rejects_zero_and_terminal_split_points() {
    let mut ctx = CpuContext::new(1);
    let tensor =
        Tensor::from_slice(&[1.0_f64, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap();
    let zero_err = tensorinv(&mut ctx, &tensor, 0).unwrap_err();
    assert!(matches!(
        zero_err,
        tenferro_device::Error::InvalidArgument(_)
    ));
    let terminal_err = tensorinv(&mut ctx, &tensor, 2).unwrap_err();
    assert!(matches!(
        terminal_err,
        tenferro_device::Error::InvalidArgument(_)
    ));
}

#[test]
fn tensorsolve_matches_identity_tensor_operator() {
    let mut ctx = CpuContext::new(1);
    let eye = Tensor::from_slice(
        &[
            1.0_f64, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ],
        &[4, 4],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let a = eye.reshape(&[2, 2, 2, 2]).unwrap();
    let b =
        Tensor::from_slice(&[1.0_f64, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap();

    let x = tensorsolve(&mut ctx, &a, &b, None).unwrap();
    assert_eq!(x.dims(), &[2, 2]);
    assert_eq!(tensor_data(&x), tensor_data(&b));
}

#[test]
fn tensorsolve_respects_solution_axis_order() {
    let mut ctx = CpuContext::new(1);
    let eye = Tensor::from_slice(
        &[
            1.0_f64, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ],
        &[4, 4],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let a = eye.reshape(&[2, 2, 2, 2]).unwrap();
    let b =
        Tensor::from_slice(&[1.0_f64, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap();

    let x = tensorsolve(&mut ctx, &a, &b, Some(&[3, 2])).unwrap();
    assert_eq!(x.dims(), &[2, 2]);
    assert_eq!(tensor_data(&x), vec![1.0, 3.0, 2.0, 4.0]);
}

#[test]
fn tensorsolve_rejects_rank_and_shape_contract_violations() {
    let mut ctx = CpuContext::new(1);
    let a =
        Tensor::from_slice(&[1.0_f64, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap();
    let b_rank =
        Tensor::from_slice(&[1.0_f64, 2.0, 3.0], &[1, 1, 3], MemoryOrder::ColumnMajor).unwrap();
    let rank_err = tensorsolve(&mut ctx, &a, &b_rank, None).unwrap_err();
    assert!(matches!(
        rank_err,
        tenferro_device::Error::InvalidArgument(_)
    ));

    let reshaped = Tensor::from_slice(
        &[
            1.0_f64, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
        &[2, 3, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let b_lead = Tensor::from_slice(&[1.0_f64, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    let lead_err = tensorsolve(&mut ctx, &reshaped, &b_lead, Some(&[0, 2])).unwrap_err();
    assert!(matches!(
        lead_err,
        tenferro_device::Error::InvalidArgument(_)
    ));

    let a_size = Tensor::from_slice(
        &[
            1.0_f64, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        ],
        &[2, 2, 3],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let b_size =
        Tensor::from_slice(&[1.0_f64, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap();
    let size_err = tensorsolve(&mut ctx, &a_size, &b_size, None).unwrap_err();
    assert!(matches!(
        size_err,
        tenferro_device::Error::InvalidArgument(_)
    ));
}

#[test]
fn batch_b_ops_reject_cuda_context() {
    with_cuda_ctx(|ctx| {
        let a_vec =
            Tensor::from_slice(&[1.0_f64, 0.0, 0.0], &[3], MemoryOrder::ColumnMajor).unwrap();
        let b_vec =
            Tensor::from_slice(&[0.0_f64, 1.0, 0.0], &[3], MemoryOrder::ColumnMajor).unwrap();
        assert!(matches!(
            cross(ctx, &a_vec, &b_vec),
            Err(tenferro_device::Error::DeviceError(_))
        ));

        let reflectors =
            Tensor::from_slice(&[1.0_f64, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor)
                .unwrap();
        let tau = Tensor::from_slice(&[0.0_f64, 0.0], &[2], MemoryOrder::ColumnMajor).unwrap();
        assert!(matches!(
            householder_product(ctx, &reflectors, &tau),
            Err(tenferro_device::Error::DeviceError(_))
        ));

        let x = Tensor::from_slice(&[2.0_f64, 3.0], &[2], MemoryOrder::ColumnMajor).unwrap();
        assert!(matches!(
            vander(ctx, &x, Some(3), true),
            Err(tenferro_device::Error::DeviceError(_))
        ));

        let eye = Tensor::from_slice(&[1.0_f64, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap();
        assert!(matches!(
            tensorinv(ctx, &eye, 1),
            Err(tenferro_device::Error::DeviceError(_))
        ));

        #[cfg(feature = "cuda")]
        {
            let hermitian = Tensor::from_slice(
                &[
                    2.0_f64, 1.0, 1.0, 3.0, //
                    4.0, 0.5, 0.5, 5.0,
                ],
                &[2, 2, 2],
                MemoryOrder::ColumnMajor,
            )
            .unwrap()
            .to_memory_space_async(tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 })
            .unwrap();
            let eigen_err = eigen(ctx, &hermitian).unwrap_err();
            assert!(matches!(eigen_err, tenferro_device::Error::DeviceError(_)));
            assert!(
                eigen_err
                    .to_string()
                    .contains("not supported on the current linalg backend"),
                "eigen should fail through capability gating before host-slice validation"
            );
        }

        let rhs = Tensor::from_slice(&[1.0_f64, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
        assert!(matches!(
            tensorsolve(ctx, &eye, &rhs, None),
            Err(tenferro_device::Error::DeviceError(_))
        ));
    });
}
