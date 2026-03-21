use crate::LinalgCapabilityOp;
use num_traits::{Float, NumCast};
use tenferro_device::LogicalMemorySpace;
use tenferro_tensor::{MemoryOrder, Tensor};

fn cuda_runtime_available() -> bool {
    std::env::var_os("TENFERRO_TEST_CUDA").is_some()
}

#[cfg(feature = "cuda")]
fn cutensor_path() -> Option<&'static str> {
    [
        "/usr/lib/x86_64-linux-gnu/libcutensor/12/libcutensor.so",
        "/usr/lib/x86_64-linux-gnu/libcutensor.so",
        "/usr/lib/libcutensor.so",
    ]
    .into_iter()
    .find(|path| std::path::Path::new(path).exists())
}

fn cast<T: NumCast>(value: f64) -> T {
    num_traits::cast(value).unwrap()
}

fn assert_close_slice<T: Float + std::fmt::Debug>(label: &str, got: &[T], expected: &[T], atol: T) {
    assert_eq!(
        got.len(),
        expected.len(),
        "{label}: length mismatch {} vs {}",
        got.len(),
        expected.len()
    );
    for (index, (&got_value, &expected_value)) in got.iter().zip(expected.iter()).enumerate() {
        let diff = (got_value - expected_value).abs();
        assert!(
            diff <= atol,
            "{label}[{index}] diff {diff:?} exceeded tolerance {atol:?}; got {got_value:?}, expected {expected_value:?}"
        );
    }
}

fn tensor_data_on_cpu<T: crate::KernelLinalgScalar>(tensor: &Tensor<T>) -> Vec<T> {
    let cpu = tensor
        .to_memory_space_async(LogicalMemorySpace::MainMemory)
        .unwrap();
    let contiguous = cpu.contiguous(MemoryOrder::ColumnMajor);
    let offset = contiguous.offset() as usize;
    let len = contiguous.len();
    contiguous.buffer().as_slice().unwrap()[offset..offset + len].to_vec()
}

fn matmul_col_major<T: Float>(lhs: &[T], m: usize, k: usize, rhs: &[T], n: usize) -> Vec<T> {
    let mut out = vec![T::zero(); m * n];
    for col in 0..n {
        for row in 0..m {
            let mut acc = T::zero();
            for inner in 0..k {
                acc = acc + lhs[row + inner * m] * rhs[inner + col * k];
            }
            out[row + col * m] = acc;
        }
    }
    out
}

fn gram_col_major<T: Float>(q: &[T], m: usize, k: usize) -> Vec<T> {
    let mut out = vec![T::zero(); k * k];
    for col in 0..k {
        for row in 0..k {
            let mut acc = T::zero();
            for inner in 0..m {
                acc = acc + q[inner + row * m] * q[inner + col * m];
            }
            out[row + col * k] = acc;
        }
    }
    out
}

fn reconstruct_thin_svd_col_major<T: Float>(
    u: &[T],
    m: usize,
    k: usize,
    s: &[T],
    vt: &[T],
    n: usize,
) -> Vec<T> {
    let mut scaled_u = u.to_vec();
    for col in 0..k {
        for row in 0..m {
            scaled_u[row + col * m] = scaled_u[row + col * m] * s[col];
        }
    }
    matmul_col_major(&scaled_u, m, k, vt, n)
}

#[cfg(feature = "cuda")]
fn cuda_solve_matches_cpu_for_small_real_matrix_generic<T>()
where
    T: crate::KernelLinalgScalar<Real = T>
        + super::scalar_type::CudaLinalgScalar
        + Float
        + std::fmt::Debug,
{
    if !cuda_runtime_available() {
        return;
    }

    let path = cutensor_path().expect("TENFERRO_TEST_CUDA is set but libcutensor.so was not found");
    let (_backend, mut cuda_ctx) = tenferro_prims::CudaBackend::load(path).unwrap();
    let mut cpu_ctx = tenferro_prims::CpuContext::new(1);

    let a_cpu = Tensor::from_slice(
        &[
            cast::<T>(3.0),
            cast::<T>(1.0),
            cast::<T>(1.0),
            cast::<T>(2.0),
        ],
        &[2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let b_cpu = Tensor::from_slice(
        &[cast::<T>(9.0), cast::<T>(8.0)],
        &[2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let expected = <crate::backend::CpuTensorLinalgBackend as crate::TensorLinalgPrims<T>>::solve(
        &mut cpu_ctx,
        &a_cpu,
        &b_cpu,
    )
    .unwrap();

    let a_gpu = a_cpu
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let b_gpu = b_cpu
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();

    let got = <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<T>>::solve(
        &mut cuda_ctx,
        &a_gpu,
        &b_gpu,
    )
    .unwrap();

    assert_eq!(got.dims(), &[2]);
    assert_eq!(
        got.logical_memory_space(),
        LogicalMemorySpace::GpuMemory { device_id: 0 }
    );
    assert_close_slice(
        "cuda solve",
        &tensor_data_on_cpu(&got),
        &tensor_data_on_cpu(&expected),
        cast::<T>(256.0) * T::epsilon(),
    );
}

#[cfg(feature = "cuda")]
fn cuda_solve_triangular_matches_cpu_for_small_real_matrix_generic<T>(upper: bool)
where
    T: crate::KernelLinalgScalar<Real = T>
        + super::scalar_type::CudaLinalgScalar
        + Float
        + std::fmt::Debug,
{
    if !cuda_runtime_available() {
        return;
    }

    let path = cutensor_path().expect("TENFERRO_TEST_CUDA is set but libcutensor.so was not found");
    let (_backend, mut cuda_ctx) = tenferro_prims::CudaBackend::load(path).unwrap();
    let mut cpu_ctx = tenferro_prims::CpuContext::new(1);

    let (a_cpu, b_cpu) = if upper {
        (
            Tensor::from_slice(
                &[
                    cast::<T>(2.0),
                    cast::<T>(0.0),
                    cast::<T>(1.0),
                    cast::<T>(3.0), //
                    cast::<T>(4.0),
                    cast::<T>(0.0),
                    cast::<T>(2.0),
                    cast::<T>(5.0),
                ],
                &[2, 2, 2],
                MemoryOrder::ColumnMajor,
            )
            .unwrap(),
            Tensor::from_slice(
                &[
                    cast::<T>(5.0),
                    cast::<T>(9.0), //
                    cast::<T>(10.0),
                    cast::<T>(15.0),
                ],
                &[2, 2],
                MemoryOrder::ColumnMajor,
            )
            .unwrap(),
        )
    } else {
        (
            Tensor::from_slice(
                &[
                    cast::<T>(2.0),
                    cast::<T>(4.0),
                    cast::<T>(0.0),
                    cast::<T>(3.0), //
                    cast::<T>(5.0),
                    cast::<T>(1.0),
                    cast::<T>(0.0),
                    cast::<T>(6.0),
                ],
                &[2, 2, 2],
                MemoryOrder::ColumnMajor,
            )
            .unwrap(),
            Tensor::from_slice(
                &[
                    cast::<T>(2.0),
                    cast::<T>(14.0), //
                    cast::<T>(5.0),
                    cast::<T>(13.0),
                ],
                &[2, 2],
                MemoryOrder::ColumnMajor,
            )
            .unwrap(),
        )
    };

    let expected =
        <crate::backend::CpuTensorLinalgBackend as crate::TensorLinalgPrims<T>>::solve_triangular(
            &mut cpu_ctx,
            &a_cpu,
            &b_cpu,
            upper,
        )
        .unwrap();

    let a_gpu = a_cpu
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let b_gpu = b_cpu
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();

    let got = <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<T>>::solve_triangular(
        &mut cuda_ctx,
        &a_gpu,
        &b_gpu,
        upper,
    )
    .unwrap();

    assert_eq!(got.dims(), &[2, 2]);
    assert_eq!(
        got.logical_memory_space(),
        LogicalMemorySpace::GpuMemory { device_id: 0 }
    );
    assert_close_slice(
        "cuda solve_triangular",
        &tensor_data_on_cpu(&got),
        &tensor_data_on_cpu(&expected),
        cast::<T>(256.0) * T::epsilon(),
    );
}

#[cfg(feature = "cuda")]
fn cuda_solve_triangular_zero_diagonal_returns_error_generic<T>()
where
    T: crate::KernelLinalgScalar<Real = T>
        + super::scalar_type::CudaLinalgScalar
        + Float
        + std::fmt::Debug,
{
    if !cuda_runtime_available() {
        return;
    }

    let path = cutensor_path().expect("TENFERRO_TEST_CUDA is set but libcutensor.so was not found");
    let (_backend, mut cuda_ctx) = tenferro_prims::CudaBackend::load(path).unwrap();

    let a_gpu = Tensor::from_slice(
        &[
            cast::<T>(1.0),
            cast::<T>(0.0),
            cast::<T>(0.0),
            cast::<T>(0.0),
        ],
        &[2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap()
    .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
    .unwrap();
    let b_gpu = Tensor::from_slice(
        &[cast::<T>(1.0), cast::<T>(2.0)],
        &[2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap()
    .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
    .unwrap();

    let err = <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<T>>::solve_triangular(
        &mut cuda_ctx,
        &a_gpu,
        &b_gpu,
        true,
    )
    .unwrap_err();
    assert!(err.to_string().contains("zero diagonal"));
}

#[cfg(feature = "cuda")]
fn cuda_lu_factor_matches_cpu_for_small_real_matrix_generic<T>()
where
    T: crate::KernelLinalgScalar<Real = T>
        + super::scalar_type::CudaLinalgScalar
        + Float
        + std::fmt::Debug,
{
    if !cuda_runtime_available() {
        return;
    }

    let path = cutensor_path().expect("TENFERRO_TEST_CUDA is set but libcutensor.so was not found");
    let (_backend, mut cuda_ctx) = tenferro_prims::CudaBackend::load(path).unwrap();
    let mut cpu_ctx = tenferro_prims::CpuContext::new(1);

    let a_cpu = Tensor::from_slice(
        &[
            cast::<T>(1.0),
            cast::<T>(3.0),
            cast::<T>(2.0),
            cast::<T>(4.0),
        ],
        &[2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();

    let expected =
        <crate::backend::CpuTensorLinalgBackend as crate::TensorLinalgPrims<T>>::lu_factor(
            &mut cpu_ctx,
            &a_cpu,
        )
        .unwrap();

    let a_gpu = a_cpu
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let got = <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<T>>::lu_factor(
        &mut cuda_ctx,
        &a_gpu,
    )
    .unwrap();

    assert_eq!(
        tensor_data_on_cpu(&got.l),
        tensor_data_on_cpu(&expected.l),
        "cuda lu_factor lower factor mismatch"
    );
    assert_eq!(
        tensor_data_on_cpu(&got.u),
        tensor_data_on_cpu(&expected.u),
        "cuda lu_factor upper factor mismatch"
    );
    assert_eq!(got.pivots, expected.pivots);
}

#[cfg(feature = "cuda")]
fn cuda_lu_factor_ex_matches_cpu_for_mixed_batch_generic<T>()
where
    T: crate::KernelLinalgScalar<Real = T>
        + super::scalar_type::CudaLinalgScalar
        + Float
        + std::fmt::Debug,
{
    if !cuda_runtime_available() {
        return;
    }

    let path = cutensor_path().expect("TENFERRO_TEST_CUDA is set but libcutensor.so was not found");
    let (_backend, mut cuda_ctx) = tenferro_prims::CudaBackend::load(path).unwrap();
    let mut cpu_ctx = tenferro_prims::CpuContext::new(1);

    let a_cpu = Tensor::from_slice(
        &[
            cast::<T>(1.0),
            cast::<T>(0.0),
            cast::<T>(0.0),
            cast::<T>(1.0), //
            cast::<T>(1.0),
            cast::<T>(2.0),
            cast::<T>(2.0),
            cast::<T>(4.0),
        ],
        &[2, 2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();

    let expected =
        <crate::backend::CpuTensorLinalgBackend as crate::TensorLinalgPrims<T>>::lu_factor_ex(
            &mut cpu_ctx,
            &a_cpu,
        )
        .unwrap();

    let a_gpu = a_cpu
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let got = <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<T>>::lu_factor_ex(
        &mut cuda_ctx,
        &a_gpu,
    )
    .unwrap();

    assert_eq!(got.info, expected.info);
    assert_eq!(
        tensor_data_on_cpu(&got.l),
        tensor_data_on_cpu(&expected.l),
        "cuda lu_factor_ex lower factor mismatch"
    );
    assert_eq!(
        tensor_data_on_cpu(&got.u),
        tensor_data_on_cpu(&expected.u),
        "cuda lu_factor_ex upper factor mismatch"
    );
    assert_eq!(got.pivots, expected.pivots);
}

#[cfg(feature = "cuda")]
fn cuda_lu_factor_ex_does_not_treat_small_nonzero_pivot_as_zero_generic<T>()
where
    T: crate::KernelLinalgScalar<Real = T>
        + super::scalar_type::CudaLinalgScalar
        + Float
        + std::fmt::Debug,
{
    if !cuda_runtime_available() {
        return;
    }

    let path = cutensor_path().expect("TENFERRO_TEST_CUDA is set but libcutensor.so was not found");
    let (_backend, mut cuda_ctx) = tenferro_prims::CudaBackend::load(path).unwrap();
    let mut cpu_ctx = tenferro_prims::CpuContext::new(1);

    let a_cpu = Tensor::from_slice(
        &[
            cast::<T>(1.0e-20),
            cast::<T>(0.0),
            cast::<T>(0.0),
            cast::<T>(1.0),
        ],
        &[2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();

    let expected =
        <crate::backend::CpuTensorLinalgBackend as crate::TensorLinalgPrims<T>>::lu_factor_ex(
            &mut cpu_ctx,
            &a_cpu,
        )
        .unwrap();

    let a_gpu = a_cpu
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let got = <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<T>>::lu_factor_ex(
        &mut cuda_ctx,
        &a_gpu,
    )
    .unwrap();

    assert_eq!(got.info, vec![0]);
    assert_eq!(got.info, expected.info);
    assert_eq!(
        tensor_data_on_cpu(&got.l),
        tensor_data_on_cpu(&expected.l),
        "cuda lu_factor_ex lower factor mismatch for small nonzero pivot"
    );
    assert_eq!(
        tensor_data_on_cpu(&got.u),
        tensor_data_on_cpu(&expected.u),
        "cuda lu_factor_ex upper factor mismatch for small nonzero pivot"
    );
    assert_eq!(got.pivots, expected.pivots);
}

#[cfg(feature = "cuda")]
fn cuda_qr_reconstructs_small_real_matrix_generic<T>()
where
    T: crate::KernelLinalgScalar<Real = T>
        + super::scalar_type::CudaLinalgScalar
        + Float
        + std::fmt::Debug,
{
    if !cuda_runtime_available() {
        return;
    }

    let path = cutensor_path().expect("TENFERRO_TEST_CUDA is set but libcutensor.so was not found");
    let (_backend, mut cuda_ctx) = tenferro_prims::CudaBackend::load(path).unwrap();

    let a_cpu = Tensor::from_slice(
        &[
            cast::<T>(1.0),
            cast::<T>(2.0),
            cast::<T>(3.0),
            cast::<T>(4.0),
            cast::<T>(5.0),
            cast::<T>(6.0),
        ],
        &[3, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let a_gpu = a_cpu
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();

    let got =
        <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<T>>::qr(&mut cuda_ctx, &a_gpu)
            .unwrap();

    assert_eq!(got.q.dims(), &[3, 2]);
    assert_eq!(got.r.dims(), &[2, 2]);
    assert_eq!(
        got.q.logical_memory_space(),
        LogicalMemorySpace::GpuMemory { device_id: 0 }
    );
    assert_eq!(
        got.r.logical_memory_space(),
        LogicalMemorySpace::GpuMemory { device_id: 0 }
    );

    let q = tensor_data_on_cpu(&got.q);
    let r = tensor_data_on_cpu(&got.r);
    let reconstructed = matmul_col_major(&q, 3, 2, &r, 2);
    assert_close_slice(
        "cuda qr reconstruction",
        &reconstructed,
        a_cpu.buffer().as_slice().unwrap(),
        cast::<T>(1024.0) * T::epsilon(),
    );

    let gram = gram_col_major(&q, 3, 2);
    let identity = vec![T::one(), T::zero(), T::zero(), T::one()];
    assert_close_slice(
        "cuda qr orthogonality",
        &gram,
        &identity,
        cast::<T>(1024.0) * T::epsilon(),
    );
}

#[cfg(feature = "cuda")]
fn cuda_cholesky_matches_cpu_for_small_real_matrix_generic<T>()
where
    T: crate::KernelLinalgScalar<Real = T>
        + super::scalar_type::CudaLinalgScalar
        + Float
        + std::fmt::Debug,
{
    if !cuda_runtime_available() {
        return;
    }

    let path = cutensor_path().expect("TENFERRO_TEST_CUDA is set but libcutensor.so was not found");
    let (_backend, mut cuda_ctx) = tenferro_prims::CudaBackend::load(path).unwrap();
    let mut cpu_ctx = tenferro_prims::CpuContext::new(1);

    let a_cpu = Tensor::from_slice(
        &[
            cast::<T>(4.0),
            cast::<T>(2.0),
            cast::<T>(2.0),
            cast::<T>(3.0),
        ],
        &[2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let expected =
        <crate::backend::CpuTensorLinalgBackend as crate::TensorLinalgPrims<T>>::cholesky(
            &mut cpu_ctx,
            &a_cpu,
        )
        .unwrap();

    let a_gpu = a_cpu
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let got = <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<T>>::cholesky(
        &mut cuda_ctx,
        &a_gpu,
    )
    .unwrap();

    assert_close_slice(
        "cuda cholesky",
        &tensor_data_on_cpu(&got),
        &tensor_data_on_cpu(&expected),
        cast::<T>(256.0) * T::epsilon(),
    );
}

#[cfg(feature = "cuda")]
fn cuda_cholesky_ex_matches_cpu_for_mixed_batch_generic<T>()
where
    T: crate::KernelLinalgScalar<Real = T>
        + super::scalar_type::CudaLinalgScalar
        + Float
        + std::fmt::Debug,
{
    if !cuda_runtime_available() {
        return;
    }

    let path = cutensor_path().expect("TENFERRO_TEST_CUDA is set but libcutensor.so was not found");
    let (_backend, mut cuda_ctx) = tenferro_prims::CudaBackend::load(path).unwrap();
    let mut cpu_ctx = tenferro_prims::CpuContext::new(1);

    let a_cpu = Tensor::from_slice(
        &[
            cast::<T>(4.0),
            cast::<T>(2.0),
            cast::<T>(2.0),
            cast::<T>(3.0), //
            cast::<T>(1.0),
            cast::<T>(2.0),
            cast::<T>(2.0),
            cast::<T>(1.0),
        ],
        &[2, 2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let expected =
        <crate::backend::CpuTensorLinalgBackend as crate::TensorLinalgPrims<T>>::cholesky_ex(
            &mut cpu_ctx,
            &a_cpu,
        )
        .unwrap();

    let a_gpu = a_cpu
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let got = <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<T>>::cholesky_ex(
        &mut cuda_ctx,
        &a_gpu,
    )
    .unwrap();

    assert_eq!(got.info, expected.info);
    assert_close_slice(
        "cuda cholesky_ex",
        &tensor_data_on_cpu(&got.l),
        &tensor_data_on_cpu(&expected.l),
        cast::<T>(256.0) * T::epsilon(),
    );
}

#[cfg(feature = "cuda")]
fn cuda_cholesky_reports_minor_for_non_spd_matrix_generic<T>()
where
    T: crate::KernelLinalgScalar<Real = T>
        + super::scalar_type::CudaLinalgScalar
        + Float
        + std::fmt::Debug,
{
    if !cuda_runtime_available() {
        return;
    }

    let path = cutensor_path().expect("TENFERRO_TEST_CUDA is set but libcutensor.so was not found");
    let (_backend, mut cuda_ctx) = tenferro_prims::CudaBackend::load(path).unwrap();
    let a_gpu = Tensor::from_slice(
        &[
            cast::<T>(1.0),
            cast::<T>(2.0),
            cast::<T>(2.0),
            cast::<T>(1.0),
        ],
        &[2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap()
    .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
    .unwrap();

    let err = <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<T>>::cholesky(
        &mut cuda_ctx,
        &a_gpu,
    )
    .unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("positive definite"));
    assert!(msg.contains("minor 2"), "unexpected cholesky error: {msg}");
}

#[cfg(feature = "cuda")]
fn cuda_svdvals_matches_cpu_for_small_real_matrix_generic<T>(wide: bool)
where
    T: crate::KernelLinalgScalar<Real = T>
        + super::scalar_type::CudaLinalgScalar
        + Float
        + std::fmt::Debug,
{
    if !cuda_runtime_available() {
        return;
    }

    let path = cutensor_path().expect("TENFERRO_TEST_CUDA is set but libcutensor.so was not found");
    let (_backend, mut cuda_ctx) = tenferro_prims::CudaBackend::load(path).unwrap();
    let mut cpu_ctx = tenferro_prims::CpuContext::new(1);

    let a_cpu = if wide {
        Tensor::from_slice(
            &[
                cast::<T>(3.0),
                cast::<T>(1.0),
                cast::<T>(1.0),
                cast::<T>(2.0),
                cast::<T>(0.0),
                cast::<T>(1.0),
            ],
            &[2, 3],
            MemoryOrder::ColumnMajor,
        )
        .unwrap()
    } else {
        Tensor::from_slice(
            &[
                cast::<T>(3.0),
                cast::<T>(1.0),
                cast::<T>(0.0),
                cast::<T>(1.0),
                cast::<T>(2.0),
                cast::<T>(1.0),
            ],
            &[3, 2],
            MemoryOrder::ColumnMajor,
        )
        .unwrap()
    };

    let expected =
        <crate::backend::CpuTensorLinalgBackend as crate::TensorLinalgPrims<T>>::thin_svd(
            &mut cpu_ctx,
            &a_cpu,
        )
        .unwrap();

    let a_gpu = a_cpu
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let got = <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<T>>::svdvals(
        &mut cuda_ctx,
        &a_gpu,
    )
    .unwrap();

    assert_eq!(got.dims(), expected.s.dims());
    assert_eq!(
        got.logical_memory_space(),
        LogicalMemorySpace::GpuMemory { device_id: 0 }
    );
    assert_close_slice(
        if wide {
            "cuda svdvals wide"
        } else {
            "cuda svdvals tall"
        },
        &tensor_data_on_cpu(&got),
        &tensor_data_on_cpu(&expected.s),
        cast::<T>(2048.0) * T::epsilon(),
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_linalg_scalar_maps_supported_standard_dtypes() {
    assert_eq!(
        <f32 as super::scalar_type::CudaLinalgScalar>::cuda_data_type(),
        super::scalar_type::CudaDataType::F32
    );
    assert_eq!(
        <f64 as super::scalar_type::CudaLinalgScalar>::cuda_data_type(),
        super::scalar_type::CudaDataType::F64
    );
    assert_eq!(
        <num_complex::Complex32 as super::scalar_type::CudaLinalgScalar>::cuda_data_type(),
        super::scalar_type::CudaDataType::Complex32
    );
    assert_eq!(
        <num_complex::Complex64 as super::scalar_type::CudaLinalgScalar>::cuda_data_type(),
        super::scalar_type::CudaDataType::Complex64
    );
}

#[test]
fn cuda_backend_reports_only_wired_capabilities() {
    let has_native_cuda = cfg!(feature = "cuda");
    assert!(
        <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<f64>>::has_linalg_support(
            LinalgCapabilityOp::Cholesky
        ) == has_native_cuda
    );
    assert!(
        <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<f64>>::has_linalg_support(
            LinalgCapabilityOp::Solve
        ) == has_native_cuda
    );
    assert!(
        <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<f64>>::has_linalg_support(
            LinalgCapabilityOp::SolveTriangular
        ) == has_native_cuda
    );
    assert!(
        <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<f64>>::has_linalg_support(
            LinalgCapabilityOp::Qr
        ) == has_native_cuda
    );
    assert!(
        <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<f64>>::has_linalg_support(
            LinalgCapabilityOp::LuFactor
        ) == has_native_cuda
    );
    assert!(
        <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<f64>>::has_linalg_support(
            LinalgCapabilityOp::LuFactorEx
        ) == has_native_cuda
    );
    assert!(
        !<super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<num_complex::Complex64>>::has_linalg_support(
            LinalgCapabilityOp::LuFactor
        )
    );
    assert!(
        !<super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<num_complex::Complex64>>::has_linalg_support(
            LinalgCapabilityOp::LuFactorEx
        )
    );
    assert!(
        <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<f64>>::has_linalg_support(
            LinalgCapabilityOp::ThinSvd
        ) == has_native_cuda
    );
    assert!(
        <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<f32>>::has_linalg_support(
            LinalgCapabilityOp::ThinSvd
        ) == has_native_cuda
    );
    assert!(
        !<super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<num_complex::Complex64>>::has_linalg_support(
            LinalgCapabilityOp::Solve
        )
    );
    assert!(
        !<super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<num_complex::Complex64>>::has_linalg_support(
            LinalgCapabilityOp::SolveTriangular
        )
    );
    assert!(
        !<super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<num_complex::Complex64>>::has_linalg_support(
            LinalgCapabilityOp::Qr
        )
    );
    assert!(
        !<super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<num_complex::Complex64>>::has_linalg_support(
            LinalgCapabilityOp::Cholesky
        )
    );
    assert!(
        !<super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<num_complex::Complex64>>::has_linalg_support(
            LinalgCapabilityOp::ThinSvd
        )
    );
}

#[test]
fn cuda_backend_reports_ex_capabilities_only_when_wired() {
    let has_native_cuda = cfg!(feature = "cuda");
    assert!(
        !<super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<f64>>::has_linalg_support(
            LinalgCapabilityOp::SolveEx
        ),
        "CUDA should not report SolveEx before the native EX solve kernel exists",
    );
    assert!(
        <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<f64>>::has_linalg_support(
            LinalgCapabilityOp::CholeskyEx
        ) == has_native_cuda,
        "CUDA CholeskyEx capability should match whether native CUDA kernels are compiled in",
    );
    assert!(
        !<super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<num_complex::Complex64>>::has_linalg_support(
            LinalgCapabilityOp::CholeskyEx
        ),
        "CUDA should not report complex CholeskyEx capability",
    );
}

#[test]
fn cuda_runtime_rejects_cpu_tensor_pointer_lookup() {
    let cpu = tenferro_tensor::Tensor::from_slice(
        &[1.0_f64, 2.0, 3.0, 4.0],
        &[2, 2],
        tenferro_tensor::MemoryOrder::ColumnMajor,
    )
    .unwrap();

    let err = super::runtime::device_ptr(&cpu, "a").unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("a"));
    assert!(msg.contains("GPU"));
}

#[test]
fn cuda_runtime_status_helpers_preserve_op_names() {
    let cublas = super::runtime::check_cublas_status(1, "cublasCreate_v2").unwrap_err();
    assert!(format!("{cublas}").contains("cublasCreate_v2"));

    let cusolver = super::runtime::check_cusolver_status(7, "cusolverDnSetStream").unwrap_err();
    assert!(format!("{cusolver}").contains("cusolverDnSetStream"));
}

#[test]
fn cuda_runtime_library_candidates_include_unversioned_fallbacks() {
    let candidates = super::runtime::default_library_candidates();
    assert!(candidates.cublas.iter().any(|name| name == "libcublas.so"));
    assert!(candidates
        .cusolver
        .iter()
        .any(|name| name == "libcusolver.so"));
}

#[test]
fn cuda_wrappers_label_missing_library_errors() {
    let err = super::wrappers::load_first_library(
        &[String::from("/definitely/missing/libcublas.so")],
        "cuBLAS",
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("cuBLAS"));
    assert!(msg.contains("missing"));
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_runtime_loads_solver_handles_with_real_context() {
    if !cuda_runtime_available() {
        return;
    }

    let path = cutensor_path().expect("TENFERRO_TEST_CUDA is set but libcutensor.so was not found");
    let (_backend, ctx) = tenferro_prims::CudaBackend::load(path).unwrap();
    let runtime = super::runtime::load_runtime(&ctx).unwrap();

    assert!(!runtime.cublas_handle.raw.is_null());
    assert!(!runtime.cusolver_handle.raw.is_null());
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_solve_matches_cpu_for_small_real_matrix_f32() {
    cuda_solve_matches_cpu_for_small_real_matrix_generic::<f32>();
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_solve_matches_cpu_for_small_real_matrix_f64() {
    cuda_solve_matches_cpu_for_small_real_matrix_generic::<f64>();
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_solve_triangular_matches_cpu_for_small_upper_real_matrix_f32() {
    cuda_solve_triangular_matches_cpu_for_small_real_matrix_generic::<f32>(true);
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_solve_triangular_matches_cpu_for_small_upper_real_matrix_f64() {
    cuda_solve_triangular_matches_cpu_for_small_real_matrix_generic::<f64>(true);
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_solve_triangular_matches_cpu_for_small_lower_real_matrix_f32() {
    cuda_solve_triangular_matches_cpu_for_small_real_matrix_generic::<f32>(false);
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_solve_triangular_matches_cpu_for_small_lower_real_matrix_f64() {
    cuda_solve_triangular_matches_cpu_for_small_real_matrix_generic::<f64>(false);
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_solve_triangular_zero_diagonal_returns_error_f32() {
    cuda_solve_triangular_zero_diagonal_returns_error_generic::<f32>();
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_solve_triangular_zero_diagonal_returns_error_f64() {
    cuda_solve_triangular_zero_diagonal_returns_error_generic::<f64>();
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_lu_factor_matches_cpu_for_small_real_matrix_f32() {
    cuda_lu_factor_matches_cpu_for_small_real_matrix_generic::<f32>();
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_lu_factor_matches_cpu_for_small_real_matrix_f64() {
    cuda_lu_factor_matches_cpu_for_small_real_matrix_generic::<f64>();
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_lu_factor_ex_matches_cpu_for_mixed_batch_f32() {
    cuda_lu_factor_ex_matches_cpu_for_mixed_batch_generic::<f32>();
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_lu_factor_ex_matches_cpu_for_mixed_batch_f64() {
    cuda_lu_factor_ex_matches_cpu_for_mixed_batch_generic::<f64>();
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_qr_reconstructs_small_real_matrix_f32() {
    cuda_qr_reconstructs_small_real_matrix_generic::<f32>();
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_qr_reconstructs_small_real_matrix_f64() {
    cuda_qr_reconstructs_small_real_matrix_generic::<f64>();
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_lu_factor_ex_does_not_treat_small_nonzero_pivot_as_zero_f32() {
    cuda_lu_factor_ex_does_not_treat_small_nonzero_pivot_as_zero_generic::<f32>();
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_lu_factor_ex_does_not_treat_small_nonzero_pivot_as_zero_f64() {
    cuda_lu_factor_ex_does_not_treat_small_nonzero_pivot_as_zero_generic::<f64>();
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_cholesky_matches_cpu_for_small_real_matrix_f32() {
    cuda_cholesky_matches_cpu_for_small_real_matrix_generic::<f32>();
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_cholesky_matches_cpu_for_small_real_matrix_f64() {
    cuda_cholesky_matches_cpu_for_small_real_matrix_generic::<f64>();
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_cholesky_ex_matches_cpu_for_mixed_batch_f32() {
    cuda_cholesky_ex_matches_cpu_for_mixed_batch_generic::<f32>();
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_cholesky_ex_matches_cpu_for_mixed_batch_f64() {
    cuda_cholesky_ex_matches_cpu_for_mixed_batch_generic::<f64>();
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_cholesky_reports_minor_for_non_spd_matrix_f32() {
    cuda_cholesky_reports_minor_for_non_spd_matrix_generic::<f32>();
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_cholesky_reports_minor_for_non_spd_matrix_f64() {
    cuda_cholesky_reports_minor_for_non_spd_matrix_generic::<f64>();
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_svdvals_matches_cpu_for_small_tall_real_matrix_f32() {
    cuda_svdvals_matches_cpu_for_small_real_matrix_generic::<f32>(false);
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_svdvals_matches_cpu_for_small_tall_real_matrix_f64() {
    cuda_svdvals_matches_cpu_for_small_real_matrix_generic::<f64>(false);
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_svdvals_matches_cpu_for_small_wide_real_matrix_f32() {
    cuda_svdvals_matches_cpu_for_small_real_matrix_generic::<f32>(true);
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_svdvals_matches_cpu_for_small_wide_real_matrix_f64() {
    cuda_svdvals_matches_cpu_for_small_real_matrix_generic::<f64>(true);
}

#[cfg(feature = "cuda")]
fn cuda_thin_svd_matches_cpu_for_small_real_matrix_generic<T>(wide: bool)
where
    T: crate::KernelLinalgScalar<Real = T>
        + super::scalar_type::CudaLinalgScalar
        + Float
        + std::fmt::Debug,
{
    if !cuda_runtime_available() {
        return;
    }

    let path = cutensor_path().expect("TENFERRO_TEST_CUDA is set but libcutensor.so was not found");
    let (_backend, mut cuda_ctx) = tenferro_prims::CudaBackend::load(path).unwrap();
    let mut cpu_ctx = tenferro_prims::CpuContext::new(1);

    let a_cpu = if wide {
        Tensor::from_slice(
            &[
                cast::<T>(3.0),
                cast::<T>(1.0),
                cast::<T>(1.0),
                cast::<T>(2.0),
                cast::<T>(0.0),
                cast::<T>(1.0),
            ],
            &[2, 3],
            MemoryOrder::ColumnMajor,
        )
        .unwrap()
    } else {
        Tensor::from_slice(
            &[
                cast::<T>(3.0),
                cast::<T>(1.0),
                cast::<T>(0.0),
                cast::<T>(1.0),
                cast::<T>(2.0),
                cast::<T>(1.0),
            ],
            &[3, 2],
            MemoryOrder::ColumnMajor,
        )
        .unwrap()
    };

    let expected =
        <crate::backend::CpuTensorLinalgBackend as crate::TensorLinalgPrims<T>>::thin_svd(
            &mut cpu_ctx,
            &a_cpu,
        )
        .unwrap();

    let a_gpu = a_cpu
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let got = <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<T>>::thin_svd(
        &mut cuda_ctx,
        &a_gpu,
    )
    .unwrap();

    let k = a_cpu.dims()[0].min(a_cpu.dims()[1]);
    let expected_dims = vec![a_cpu.dims()[0], k];
    let expected_vt_dims = vec![k, a_cpu.dims()[1]];
    assert_eq!(got.u.dims(), expected_dims.as_slice());
    assert_eq!(got.s.dims(), &[k]);
    assert_eq!(got.vt.dims(), expected_vt_dims.as_slice());
    assert_eq!(
        got.u.logical_memory_space(),
        LogicalMemorySpace::GpuMemory { device_id: 0 }
    );
    assert_eq!(
        got.s.logical_memory_space(),
        LogicalMemorySpace::GpuMemory { device_id: 0 }
    );
    assert_eq!(
        got.vt.logical_memory_space(),
        LogicalMemorySpace::GpuMemory { device_id: 0 }
    );

    assert_close_slice(
        if wide {
            "cuda thin_svd singular values wide"
        } else {
            "cuda thin_svd singular values tall"
        },
        &tensor_data_on_cpu(&got.s),
        &tensor_data_on_cpu(&expected.s),
        cast::<T>(4096.0) * T::epsilon(),
    );

    let u = tensor_data_on_cpu(&got.u);
    let s = tensor_data_on_cpu(&got.s);
    let vt = tensor_data_on_cpu(&got.vt);
    let reconstructed =
        reconstruct_thin_svd_col_major(&u, a_cpu.dims()[0], k, &s, &vt, a_cpu.dims()[1]);
    assert_close_slice(
        if wide {
            "cuda thin_svd reconstruction wide"
        } else {
            "cuda thin_svd reconstruction tall"
        },
        &reconstructed,
        a_cpu.buffer().as_slice().unwrap(),
        cast::<T>(4096.0) * T::epsilon(),
    );
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_thin_svd_matches_cpu_for_small_tall_real_matrix_f32() {
    cuda_thin_svd_matches_cpu_for_small_real_matrix_generic::<f32>(false);
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_thin_svd_matches_cpu_for_small_tall_real_matrix_f64() {
    cuda_thin_svd_matches_cpu_for_small_real_matrix_generic::<f64>(false);
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_thin_svd_matches_cpu_for_small_wide_real_matrix_f32() {
    cuda_thin_svd_matches_cpu_for_small_real_matrix_generic::<f32>(true);
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_thin_svd_matches_cpu_for_small_wide_real_matrix_f64() {
    cuda_thin_svd_matches_cpu_for_small_real_matrix_generic::<f64>(true);
}
