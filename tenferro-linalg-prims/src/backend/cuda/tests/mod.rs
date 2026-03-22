use crate::LinalgCapabilityOp;
#[cfg(feature = "cuda")]
use num_complex::{Complex32, Complex64};
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

fn assert_close_complex_slice<T: Float + std::fmt::Debug>(
    label: &str,
    got: &[num_complex::Complex<T>],
    expected: &[num_complex::Complex<T>],
    atol: T,
) {
    assert_eq!(
        got.len(),
        expected.len(),
        "{label}: length mismatch {} vs {}",
        got.len(),
        expected.len()
    );
    for (index, (got_value, expected_value)) in got.iter().zip(expected.iter()).enumerate() {
        let re_diff = (got_value.re - expected_value.re).abs();
        let im_diff = (got_value.im - expected_value.im).abs();
        assert!(
            re_diff <= atol,
            "{label}[{index}].re diff {re_diff:?} exceeded tolerance {atol:?}; got {:?}, expected {:?}",
            got_value,
            expected_value
        );
        assert!(
            im_diff <= atol,
            "{label}[{index}].im diff {im_diff:?} exceeded tolerance {atol:?}; got {:?}, expected {:?}",
            got_value,
            expected_value
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

fn matmul_col_major_complex<T: Float>(
    lhs: &[num_complex::Complex<T>],
    m: usize,
    k: usize,
    rhs: &[num_complex::Complex<T>],
    n: usize,
) -> Vec<num_complex::Complex<T>> {
    let mut out = vec![num_complex::Complex::new(T::zero(), T::zero()); m * n];
    for col in 0..n {
        for row in 0..m {
            let mut acc = num_complex::Complex::new(T::zero(), T::zero());
            for inner in 0..k {
                acc = acc + lhs[row + inner * m] * rhs[inner + col * k];
            }
            out[row + col * m] = acc;
        }
    }
    out
}

fn gram_col_major_complex<T: Float>(
    q: &[num_complex::Complex<T>],
    m: usize,
    k: usize,
) -> Vec<num_complex::Complex<T>> {
    let mut out = vec![num_complex::Complex::new(T::zero(), T::zero()); k * k];
    for col in 0..k {
        for row in 0..k {
            let mut acc = num_complex::Complex::new(T::zero(), T::zero());
            for inner in 0..m {
                acc = acc + q[inner + row * m].conj() * q[inner + col * m];
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

fn reconstruct_thin_svd_col_major_complex<T: Float>(
    u: &[num_complex::Complex<T>],
    m: usize,
    k: usize,
    s: &[T],
    vt: &[num_complex::Complex<T>],
    n: usize,
) -> Vec<num_complex::Complex<T>> {
    let mut scaled_u = u.to_vec();
    for col in 0..k {
        for row in 0..m {
            scaled_u[row + col * m] = scaled_u[row + col * m] * s[col];
        }
    }
    matmul_col_major_complex(&scaled_u, m, k, vt, n)
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
fn cuda_solve_matches_cpu_for_small_complex32_matrix() {
    if !cuda_runtime_available() {
        return;
    }

    let path = cutensor_path().expect("TENFERRO_TEST_CUDA is set but libcutensor.so was not found");
    let (_backend, mut cuda_ctx) = tenferro_prims::CudaBackend::load(path).unwrap();
    let mut cpu_ctx = tenferro_prims::CpuContext::new(1);

    let a_cpu = Tensor::from_slice(
        &[
            Complex32::new(2.0, 1.0),
            Complex32::new(1.0, -1.0),
            Complex32::new(3.0, 0.5),
            Complex32::new(4.0, 2.0),
        ],
        &[2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let b_cpu = Tensor::from_slice(
        &[Complex32::new(5.0, -2.0), Complex32::new(7.0, 3.0)],
        &[2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let expected = <crate::backend::CpuTensorLinalgBackend as crate::TensorLinalgPrims<
        Complex32,
    >>::solve(&mut cpu_ctx, &a_cpu, &b_cpu)
    .unwrap();

    let a_gpu = a_cpu
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let b_gpu = b_cpu
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();

    let got = <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<Complex32>>::solve(
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
    assert_close_complex_slice(
        "cuda solve complex32",
        &tensor_data_on_cpu(&got),
        &tensor_data_on_cpu(&expected),
        256.0_f32 * f32::epsilon(),
    );
}

#[cfg(feature = "cuda")]
fn cuda_solve_matches_cpu_for_small_complex64_matrix() {
    if !cuda_runtime_available() {
        return;
    }

    let path = cutensor_path().expect("TENFERRO_TEST_CUDA is set but libcutensor.so was not found");
    let (_backend, mut cuda_ctx) = tenferro_prims::CudaBackend::load(path).unwrap();
    let mut cpu_ctx = tenferro_prims::CpuContext::new(1);

    let a_cpu = Tensor::from_slice(
        &[
            Complex64::new(2.0, 1.0),
            Complex64::new(1.0, -1.0),
            Complex64::new(3.0, 0.5),
            Complex64::new(4.0, 2.0),
        ],
        &[2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let b_cpu = Tensor::from_slice(
        &[Complex64::new(5.0, -2.0), Complex64::new(7.0, 3.0)],
        &[2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let expected = <crate::backend::CpuTensorLinalgBackend as crate::TensorLinalgPrims<
        Complex64,
    >>::solve(&mut cpu_ctx, &a_cpu, &b_cpu)
    .unwrap();

    let a_gpu = a_cpu
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let b_gpu = b_cpu
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();

    let got = <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<Complex64>>::solve(
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
    assert_close_complex_slice(
        "cuda solve complex64",
        &tensor_data_on_cpu(&got),
        &tensor_data_on_cpu(&expected),
        256.0_f64 * f64::epsilon(),
    );
}

#[cfg(feature = "cuda")]
fn cuda_solve_broadcasted_rhs_matches_cpu_generic<T>()
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
            cast::<T>(2.0), //
            cast::<T>(3.0),
            cast::<T>(0.0),
            cast::<T>(0.0),
            cast::<T>(4.0),
        ],
        &[2, 2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let b_cpu = Tensor::from_slice(
        &[cast::<T>(6.0), cast::<T>(8.0)],
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

    assert_eq!(got.dims(), &[2, 2]);
    assert_eq!(
        got.logical_memory_space(),
        LogicalMemorySpace::GpuMemory { device_id: 0 }
    );
    assert_close_slice(
        "cuda solve broadcast rhs",
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
fn cuda_solve_triangular_broadcasted_rhs_matches_cpu_generic<T>(upper: bool)
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
            cast::<T>(2.0), //
            cast::<T>(3.0),
            cast::<T>(0.0),
            cast::<T>(0.0),
            cast::<T>(4.0),
        ],
        &[2, 2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let b_cpu = Tensor::from_slice(
        &[cast::<T>(6.0), cast::<T>(8.0)],
        &[2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();

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
        "cuda solve_triangular broadcast rhs",
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
fn cuda_lu_factor_matches_cpu_for_small_complex32_matrix() {
    if !cuda_runtime_available() {
        return;
    }

    let path = cutensor_path().expect("TENFERRO_TEST_CUDA is set but libcutensor.so was not found");
    let (_backend, mut cuda_ctx) = tenferro_prims::CudaBackend::load(path).unwrap();
    let mut cpu_ctx = tenferro_prims::CpuContext::new(1);

    let a_cpu = Tensor::from_slice(
        &[
            Complex32::new(2.0, 1.0),
            Complex32::new(1.0, -1.0),
            Complex32::new(3.0, 0.5),
            Complex32::new(4.0, 2.0),
        ],
        &[2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let expected = <crate::backend::CpuTensorLinalgBackend as crate::TensorLinalgPrims<
        Complex32,
    >>::lu_factor(&mut cpu_ctx, &a_cpu)
    .unwrap();

    let a_gpu = a_cpu
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let got = <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<Complex32>>::lu_factor(
        &mut cuda_ctx,
        &a_gpu,
    )
    .unwrap();

    assert_close_complex_slice(
        "cuda lu_factor complex32 lower factor",
        &tensor_data_on_cpu(&got.l),
        &tensor_data_on_cpu(&expected.l),
        256.0_f32 * f32::epsilon(),
    );
    assert_close_complex_slice(
        "cuda lu_factor complex32 upper factor",
        &tensor_data_on_cpu(&got.u),
        &tensor_data_on_cpu(&expected.u),
        256.0_f32 * f32::epsilon(),
    );
    assert_eq!(got.pivots, expected.pivots);
}

#[cfg(feature = "cuda")]
fn cuda_lu_factor_matches_cpu_for_small_complex64_matrix() {
    if !cuda_runtime_available() {
        return;
    }

    let path = cutensor_path().expect("TENFERRO_TEST_CUDA is set but libcutensor.so was not found");
    let (_backend, mut cuda_ctx) = tenferro_prims::CudaBackend::load(path).unwrap();
    let mut cpu_ctx = tenferro_prims::CpuContext::new(1);

    let a_cpu = Tensor::from_slice(
        &[
            Complex64::new(2.0, 1.0),
            Complex64::new(1.0, -1.0),
            Complex64::new(3.0, 0.5),
            Complex64::new(4.0, 2.0),
        ],
        &[2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let expected = <crate::backend::CpuTensorLinalgBackend as crate::TensorLinalgPrims<
        Complex64,
    >>::lu_factor(&mut cpu_ctx, &a_cpu)
    .unwrap();

    let a_gpu = a_cpu
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let got = <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<Complex64>>::lu_factor(
        &mut cuda_ctx,
        &a_gpu,
    )
    .unwrap();

    assert_close_complex_slice(
        "cuda lu_factor complex64 lower factor",
        &tensor_data_on_cpu(&got.l),
        &tensor_data_on_cpu(&expected.l),
        256.0_f64 * f64::epsilon(),
    );
    assert_close_complex_slice(
        "cuda lu_factor complex64 upper factor",
        &tensor_data_on_cpu(&got.u),
        &tensor_data_on_cpu(&expected.u),
        256.0_f64 * f64::epsilon(),
    );
    assert_eq!(got.pivots, expected.pivots);
}

#[cfg(feature = "cuda")]
fn cuda_lu_factor_ex_matches_cpu_for_complex_mixed_batch_complex32() {
    if !cuda_runtime_available() {
        return;
    }

    let path = cutensor_path().expect("TENFERRO_TEST_CUDA is set but libcutensor.so was not found");
    let (_backend, mut cuda_ctx) = tenferro_prims::CudaBackend::load(path).unwrap();
    let mut cpu_ctx = tenferro_prims::CpuContext::new(1);

    let a_cpu = Tensor::from_slice(
        &[
            Complex32::new(1.0, 0.25),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(1.0, -0.5), //
            Complex32::new(1.0, 0.0),
            Complex32::new(2.0, 0.5),
            Complex32::new(2.0, -0.25),
            Complex32::new(4.0, 0.75),
        ],
        &[2, 2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();

    let expected = <crate::backend::CpuTensorLinalgBackend as crate::TensorLinalgPrims<
        Complex32,
    >>::lu_factor_ex(&mut cpu_ctx, &a_cpu)
    .unwrap();

    let a_gpu = a_cpu
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let got =
        <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<Complex32>>::lu_factor_ex(
            &mut cuda_ctx,
            &a_gpu,
        )
        .unwrap();

    assert_eq!(got.info, expected.info);
    assert_close_complex_slice(
        "cuda lu_factor_ex complex32 lower factor",
        &tensor_data_on_cpu(&got.l),
        &tensor_data_on_cpu(&expected.l),
        256.0_f32 * f32::epsilon(),
    );
    assert_close_complex_slice(
        "cuda lu_factor_ex complex32 upper factor",
        &tensor_data_on_cpu(&got.u),
        &tensor_data_on_cpu(&expected.u),
        256.0_f32 * f32::epsilon(),
    );
    assert_eq!(got.pivots, expected.pivots);
}

#[cfg(feature = "cuda")]
fn cuda_lu_factor_ex_matches_cpu_for_complex_mixed_batch_complex64() {
    if !cuda_runtime_available() {
        return;
    }

    let path = cutensor_path().expect("TENFERRO_TEST_CUDA is set but libcutensor.so was not found");
    let (_backend, mut cuda_ctx) = tenferro_prims::CudaBackend::load(path).unwrap();
    let mut cpu_ctx = tenferro_prims::CpuContext::new(1);

    let a_cpu = Tensor::from_slice(
        &[
            Complex64::new(1.0, 0.25),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, -0.5), //
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.5),
            Complex64::new(2.0, -0.25),
            Complex64::new(4.0, 0.75),
        ],
        &[2, 2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();

    let expected = <crate::backend::CpuTensorLinalgBackend as crate::TensorLinalgPrims<
        Complex64,
    >>::lu_factor_ex(&mut cpu_ctx, &a_cpu)
    .unwrap();

    let a_gpu = a_cpu
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let got =
        <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<Complex64>>::lu_factor_ex(
            &mut cuda_ctx,
            &a_gpu,
        )
        .unwrap();

    assert_eq!(got.info, expected.info);
    assert_close_complex_slice(
        "cuda lu_factor_ex complex64 lower factor",
        &tensor_data_on_cpu(&got.l),
        &tensor_data_on_cpu(&expected.l),
        256.0_f64 * f64::epsilon(),
    );
    assert_close_complex_slice(
        "cuda lu_factor_ex complex64 upper factor",
        &tensor_data_on_cpu(&got.u),
        &tensor_data_on_cpu(&expected.u),
        256.0_f64 * f64::epsilon(),
    );
    assert_eq!(got.pivots, expected.pivots);
}

#[cfg(feature = "cuda")]
fn cuda_lu_factor_ex_reports_zero_pivot_for_complex_mixed_batch_complex32() {
    if !cuda_runtime_available() {
        return;
    }

    let path = cutensor_path().expect("TENFERRO_TEST_CUDA is set but libcutensor.so was not found");
    let (_backend, mut cuda_ctx) = tenferro_prims::CudaBackend::load(path).unwrap();
    let mut cpu_ctx = tenferro_prims::CpuContext::new(1);

    let a_cpu = Tensor::from_slice(
        &[
            Complex32::new(1.0, 0.25),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(1.0, -0.5), //
            Complex32::new(1.0, 0.0),
            Complex32::new(2.0, 0.0),
            Complex32::new(2.0, 0.0),
            Complex32::new(4.0, 0.0),
        ],
        &[2, 2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();

    let expected = <crate::backend::CpuTensorLinalgBackend as crate::TensorLinalgPrims<
        Complex32,
    >>::lu_factor_ex(&mut cpu_ctx, &a_cpu)
    .unwrap();

    let a_gpu = a_cpu
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let got =
        <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<Complex32>>::lu_factor_ex(
            &mut cuda_ctx,
            &a_gpu,
        )
        .unwrap();

    assert_eq!(got.info, vec![0, 2]);
    assert_eq!(got.info, expected.info);
    assert_close_complex_slice(
        "cuda lu_factor_ex complex32 zero pivot lower factor",
        &tensor_data_on_cpu(&got.l),
        &tensor_data_on_cpu(&expected.l),
        256.0_f32 * f32::epsilon(),
    );
    assert_close_complex_slice(
        "cuda lu_factor_ex complex32 zero pivot upper factor",
        &tensor_data_on_cpu(&got.u),
        &tensor_data_on_cpu(&expected.u),
        256.0_f32 * f32::epsilon(),
    );
    assert_eq!(got.pivots, expected.pivots);
}

#[cfg(feature = "cuda")]
fn cuda_lu_factor_ex_reports_zero_pivot_for_complex_mixed_batch_complex64() {
    if !cuda_runtime_available() {
        return;
    }

    let path = cutensor_path().expect("TENFERRO_TEST_CUDA is set but libcutensor.so was not found");
    let (_backend, mut cuda_ctx) = tenferro_prims::CudaBackend::load(path).unwrap();
    let mut cpu_ctx = tenferro_prims::CpuContext::new(1);

    let a_cpu = Tensor::from_slice(
        &[
            Complex64::new(1.0, 0.25),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, -0.5), //
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(4.0, 0.0),
        ],
        &[2, 2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();

    let expected = <crate::backend::CpuTensorLinalgBackend as crate::TensorLinalgPrims<
        Complex64,
    >>::lu_factor_ex(&mut cpu_ctx, &a_cpu)
    .unwrap();

    let a_gpu = a_cpu
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let got =
        <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<Complex64>>::lu_factor_ex(
            &mut cuda_ctx,
            &a_gpu,
        )
        .unwrap();

    assert_eq!(got.info, vec![0, 2]);
    assert_eq!(got.info, expected.info);
    assert_close_complex_slice(
        "cuda lu_factor_ex complex64 zero pivot lower factor",
        &tensor_data_on_cpu(&got.l),
        &tensor_data_on_cpu(&expected.l),
        256.0_f64 * f64::epsilon(),
    );
    assert_close_complex_slice(
        "cuda lu_factor_ex complex64 zero pivot upper factor",
        &tensor_data_on_cpu(&got.u),
        &tensor_data_on_cpu(&expected.u),
        256.0_f64 * f64::epsilon(),
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
fn cuda_cholesky_matches_cpu_for_small_complex32_matrix() {
    if !cuda_runtime_available() {
        return;
    }

    let path = cutensor_path().expect("TENFERRO_TEST_CUDA is set but libcutensor.so was not found");
    let (_backend, mut cuda_ctx) = tenferro_prims::CudaBackend::load(path).unwrap();
    let mut cpu_ctx = tenferro_prims::CpuContext::new(1);

    let a_cpu = Tensor::from_slice(
        &[
            Complex32::new(4.0, 0.0),
            Complex32::new(1.0, -1.0),
            Complex32::new(1.0, 1.0),
            Complex32::new(3.0, 0.0),
        ],
        &[2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();

    let expected = <crate::backend::CpuTensorLinalgBackend as crate::TensorLinalgPrims<
        Complex32,
    >>::cholesky(&mut cpu_ctx, &a_cpu)
    .unwrap();

    let a_gpu = a_cpu
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let got = <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<Complex32>>::cholesky(
        &mut cuda_ctx,
        &a_gpu,
    )
    .unwrap();

    assert_eq!(
        got.logical_memory_space(),
        LogicalMemorySpace::GpuMemory { device_id: 0 }
    );
    assert_close_complex_slice(
        "cuda cholesky complex32",
        &tensor_data_on_cpu(&got),
        &tensor_data_on_cpu(&expected),
        256.0_f32 * f32::epsilon(),
    );
}

#[cfg(feature = "cuda")]
fn cuda_cholesky_matches_cpu_for_small_complex64_matrix() {
    if !cuda_runtime_available() {
        return;
    }

    let path = cutensor_path().expect("TENFERRO_TEST_CUDA is set but libcutensor.so was not found");
    let (_backend, mut cuda_ctx) = tenferro_prims::CudaBackend::load(path).unwrap();
    let mut cpu_ctx = tenferro_prims::CpuContext::new(1);

    let a_cpu = Tensor::from_slice(
        &[
            Complex64::new(4.0, 0.0),
            Complex64::new(1.0, -1.0),
            Complex64::new(1.0, 1.0),
            Complex64::new(3.0, 0.0),
        ],
        &[2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();

    let expected = <crate::backend::CpuTensorLinalgBackend as crate::TensorLinalgPrims<
        Complex64,
    >>::cholesky(&mut cpu_ctx, &a_cpu)
    .unwrap();

    let a_gpu = a_cpu
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let got = <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<Complex64>>::cholesky(
        &mut cuda_ctx,
        &a_gpu,
    )
    .unwrap();

    assert_eq!(
        got.logical_memory_space(),
        LogicalMemorySpace::GpuMemory { device_id: 0 }
    );
    assert_close_complex_slice(
        "cuda cholesky complex64",
        &tensor_data_on_cpu(&got),
        &tensor_data_on_cpu(&expected),
        256.0_f64 * f64::epsilon(),
    );
}

#[cfg(feature = "cuda")]
fn cuda_cholesky_ex_reports_minor_for_complex_non_spd_matrix_complex32() {
    if !cuda_runtime_available() {
        return;
    }

    let path = cutensor_path().expect("TENFERRO_TEST_CUDA is set but libcutensor.so was not found");
    let (_backend, mut cuda_ctx) = tenferro_prims::CudaBackend::load(path).unwrap();
    let mut cpu_ctx = tenferro_prims::CpuContext::new(1);

    let a_cpu = Tensor::from_slice(
        &[
            Complex32::new(1.0, 0.0),
            Complex32::new(2.0, 0.0),
            Complex32::new(2.0, 0.0),
            Complex32::new(1.0, 0.0),
        ],
        &[2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let expected = <crate::backend::CpuTensorLinalgBackend as crate::TensorLinalgPrims<
        Complex32,
    >>::cholesky_ex(&mut cpu_ctx, &a_cpu)
    .unwrap();

    let a_gpu = a_cpu
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let got = <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<Complex32>>::cholesky_ex(
        &mut cuda_ctx,
        &a_gpu,
    )
    .unwrap();

    assert_eq!(got.info, expected.info);
    assert_close_complex_slice(
        "cuda cholesky_ex complex32",
        &tensor_data_on_cpu(&got.l),
        &tensor_data_on_cpu(&expected.l),
        256.0_f32 * f32::epsilon(),
    );
}

#[cfg(feature = "cuda")]
fn cuda_cholesky_ex_reports_minor_for_complex_non_spd_matrix_complex64() {
    if !cuda_runtime_available() {
        return;
    }

    let path = cutensor_path().expect("TENFERRO_TEST_CUDA is set but libcutensor.so was not found");
    let (_backend, mut cuda_ctx) = tenferro_prims::CudaBackend::load(path).unwrap();
    let mut cpu_ctx = tenferro_prims::CpuContext::new(1);

    let a_cpu = Tensor::from_slice(
        &[
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(1.0, 0.0),
        ],
        &[2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let expected = <crate::backend::CpuTensorLinalgBackend as crate::TensorLinalgPrims<
        Complex64,
    >>::cholesky_ex(&mut cpu_ctx, &a_cpu)
    .unwrap();

    let a_gpu = a_cpu
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let got = <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<Complex64>>::cholesky_ex(
        &mut cuda_ctx,
        &a_gpu,
    )
    .unwrap();

    assert_eq!(got.info, expected.info);
    assert_close_complex_slice(
        "cuda cholesky_ex complex64",
        &tensor_data_on_cpu(&got.l),
        &tensor_data_on_cpu(&expected.l),
        256.0_f64 * f64::epsilon(),
    );
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
fn cuda_svdvals_matches_cpu_for_small_complex_matrix_generic<T>(
    wide: bool,
    data: &[num_complex::Complex<T>],
) where
    T: crate::KernelLinalgScalar<Real = T> + Float + std::fmt::Debug + Send + Sync,
    num_complex::Complex<T>: crate::KernelLinalgScalar<Real = T>
        + super::scalar_type::CudaLinalgScalar
        + std::fmt::Debug,
{
    if !cuda_runtime_available() {
        return;
    }

    let path = cutensor_path().expect("TENFERRO_TEST_CUDA is set but libcutensor.so was not found");
    let (_backend, mut cuda_ctx) = tenferro_prims::CudaBackend::load(path).unwrap();
    let mut cpu_ctx = tenferro_prims::CpuContext::new(1);

    let dims = if wide { [2, 3] } else { [3, 2] };
    let a_cpu = Tensor::from_slice(data, &dims, MemoryOrder::ColumnMajor).unwrap();
    let expected = <crate::backend::CpuTensorLinalgBackend as crate::TensorLinalgPrims<
        num_complex::Complex<T>,
    >>::thin_svd(&mut cpu_ctx, &a_cpu)
    .unwrap();

    let a_gpu = a_cpu
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let got = <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<
        num_complex::Complex<T>,
    >>::svdvals(&mut cuda_ctx, &a_gpu)
    .unwrap();

    assert_eq!(got.dims(), expected.s.dims());
    assert_eq!(
        got.logical_memory_space(),
        LogicalMemorySpace::GpuMemory { device_id: 0 }
    );
    assert_close_slice(
        if wide {
            "cuda complex svdvals wide"
        } else {
            "cuda complex svdvals tall"
        },
        &tensor_data_on_cpu(&got),
        &tensor_data_on_cpu(&expected.s),
        cast::<T>(4096.0) * T::epsilon(),
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
        <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<num_complex::Complex64>>::has_linalg_support(
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
        <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<num_complex::Complex64>>::has_linalg_support(
            LinalgCapabilityOp::LuFactor
        ) == has_native_cuda
    );
    assert!(
        <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<num_complex::Complex64>>::has_linalg_support(
            LinalgCapabilityOp::LuFactorEx
        ) == has_native_cuda
    );
    assert!(
        <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<num_complex::Complex32>>::has_linalg_support(
            LinalgCapabilityOp::MatrixPower
        ) == has_native_cuda
    );
    assert!(
        <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<num_complex::Complex64>>::has_linalg_support(
            LinalgCapabilityOp::MatrixPower
        ) == has_native_cuda
    );
    assert!(
        <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<f64>>::has_linalg_support(
            LinalgCapabilityOp::ThinSvd
        ) == has_native_cuda
    );
    assert!(
        <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<f32>>::has_linalg_support(
            LinalgCapabilityOp::MatrixPower
        ) == has_native_cuda
    );
    assert!(
        <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<f64>>::has_linalg_support(
            LinalgCapabilityOp::MatrixPower
        ) == has_native_cuda
    );
    assert!(
        <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<f32>>::has_linalg_support(
            LinalgCapabilityOp::Det
        ) == has_native_cuda
    );
    assert!(
        <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<f64>>::has_linalg_support(
            LinalgCapabilityOp::Det
        ) == has_native_cuda
    );
    assert!(
        <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<f32>>::has_linalg_support(
            LinalgCapabilityOp::Slogdet
        ) == has_native_cuda
    );
    assert!(
        <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<f64>>::has_linalg_support(
            LinalgCapabilityOp::Slogdet
        ) == has_native_cuda
    );
    assert!(
        <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<f32>>::has_linalg_support(
            LinalgCapabilityOp::ThinSvd
        ) == has_native_cuda
    );
    assert!(
        <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<f32>>::has_linalg_support(
            LinalgCapabilityOp::Norm
        ) == has_native_cuda
    );
    assert!(
        <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<f64>>::has_linalg_support(
            LinalgCapabilityOp::Norm
        ) == has_native_cuda
    );
    assert!(
        <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<f32>>::has_linalg_support(
            LinalgCapabilityOp::Pinv
        ) == has_native_cuda
    );
    assert!(
        <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<f64>>::has_linalg_support(
            LinalgCapabilityOp::Pinv
        ) == has_native_cuda
    );
    assert!(
        <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<f32>>::has_linalg_support(
            LinalgCapabilityOp::Inv
        ) == has_native_cuda
    );
    assert!(
        <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<f64>>::has_linalg_support(
            LinalgCapabilityOp::Inv
        ) == has_native_cuda
    );
    assert!(
        !<super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<f32>>::has_linalg_support(
            LinalgCapabilityOp::Lstsq
        )
    );
    assert!(
        !<super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<f64>>::has_linalg_support(
            LinalgCapabilityOp::Lstsq
        )
    );
    assert!(
        <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<num_complex::Complex64>>::has_linalg_support(
            LinalgCapabilityOp::Solve
        ) == has_native_cuda
    );
    assert!(
        <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<num_complex::Complex64>>::has_linalg_support(
            LinalgCapabilityOp::Cholesky
        ) == has_native_cuda
    );
    assert!(
        <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<num_complex::Complex64>>::has_linalg_support(
            LinalgCapabilityOp::CholeskyEx
        ) == has_native_cuda
    );
    assert!(
        <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<num_complex::Complex32>>::has_linalg_support(
            LinalgCapabilityOp::ThinSvd
        ) == has_native_cuda
    );
    assert!(
        <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<num_complex::Complex64>>::has_linalg_support(
            LinalgCapabilityOp::ThinSvd
        ) == has_native_cuda
    );
    assert!(
        <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<num_complex::Complex32>>::has_linalg_support(
            LinalgCapabilityOp::Qr
        ) == has_native_cuda
    );
    assert!(
        <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<num_complex::Complex64>>::has_linalg_support(
            LinalgCapabilityOp::Qr
        ) == has_native_cuda
    );
    assert!(
        !<super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<num_complex::Complex32>>::has_linalg_support(
            LinalgCapabilityOp::Det
        )
    );
    assert!(
        !<super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<num_complex::Complex64>>::has_linalg_support(
            LinalgCapabilityOp::Det
        )
    );
    assert!(
        !<super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<num_complex::Complex32>>::has_linalg_support(
            LinalgCapabilityOp::Norm
        )
    );
    assert!(
        !<super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<num_complex::Complex64>>::has_linalg_support(
            LinalgCapabilityOp::Norm
        )
    );
    assert!(
        !<super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<num_complex::Complex32>>::has_linalg_support(
            LinalgCapabilityOp::Pinv
        )
    );
    assert!(
        !<super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<num_complex::Complex64>>::has_linalg_support(
            LinalgCapabilityOp::Pinv
        )
    );
    assert!(
        !<super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<num_complex::Complex32>>::has_linalg_support(
            LinalgCapabilityOp::Lstsq
        )
    );
    assert!(
        !<super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<num_complex::Complex64>>::has_linalg_support(
            LinalgCapabilityOp::Lstsq
        )
    );
    assert!(
        !<super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<num_complex::Complex64>>::has_linalg_support(
            LinalgCapabilityOp::SolveTriangular
        )
    );
}

#[test]
fn cuda_backend_reports_ex_capabilities_only_when_wired() {
    let has_native_cuda = cfg!(feature = "cuda");
    assert!(
        <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<f64>>::has_linalg_support(
            LinalgCapabilityOp::SolveEx
        ) == has_native_cuda,
        "CUDA SolveEx capability should match whether native CUDA kernels are compiled in",
    );
    assert!(
        <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<num_complex::Complex64>>::has_linalg_support(
            LinalgCapabilityOp::SolveEx
        ) == has_native_cuda,
        "CUDA complex SolveEx capability should match whether native CUDA kernels are compiled in",
    );
    assert!(
        <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<f64>>::has_linalg_support(
            LinalgCapabilityOp::CholeskyEx
        ) == has_native_cuda,
        "CUDA CholeskyEx capability should match whether native CUDA kernels are compiled in",
    );
    assert!(
        <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<f32>>::has_linalg_support(
            LinalgCapabilityOp::SolveEx
        ) == has_native_cuda,
        "CUDA SolveEx capability should match whether native CUDA kernels are compiled in for f32",
    );
    assert!(
        <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<num_complex::Complex64>>::has_linalg_support(
            LinalgCapabilityOp::SolveEx
        ) == has_native_cuda,
        "CUDA should report complex SolveEx capability once native CUDA kernels are compiled in",
    );
    assert!(
        <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<num_complex::Complex64>>::has_linalg_support(
            LinalgCapabilityOp::CholeskyEx
        ) == has_native_cuda,
        "CUDA should report complex CholeskyEx capability once native CUDA kernels are compiled in",
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
fn cuda_solve_broadcasted_rhs_matches_cpu_f32() {
    cuda_solve_broadcasted_rhs_matches_cpu_generic::<f32>();
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_solve_broadcasted_rhs_matches_cpu_f64() {
    cuda_solve_broadcasted_rhs_matches_cpu_generic::<f64>();
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_solve_ex_matches_cpu_for_mixed_batch_f32() {
    cuda_solve_ex_matches_cpu_for_mixed_batch_generic::<f32>();
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_solve_ex_matches_cpu_for_mixed_batch_f64() {
    cuda_solve_ex_matches_cpu_for_mixed_batch_generic::<f64>();
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_solve_ex_broadcasted_rhs_matches_cpu_f32() {
    cuda_solve_ex_broadcasted_rhs_matches_cpu_generic::<f32>();
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_solve_ex_broadcasted_rhs_matches_cpu_f64() {
    cuda_solve_ex_broadcasted_rhs_matches_cpu_generic::<f64>();
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_solve_ex_does_not_treat_small_nonzero_pivot_as_zero_f32() {
    cuda_solve_ex_does_not_treat_small_nonzero_pivot_as_zero_generic::<f32>();
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_solve_ex_does_not_treat_small_nonzero_pivot_as_zero_f64() {
    cuda_solve_ex_does_not_treat_small_nonzero_pivot_as_zero_generic::<f64>();
}

#[cfg(feature = "cuda")]
fn cuda_solve_ex_matches_cpu_for_mixed_batch_generic<T>()
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
    let b_cpu = Tensor::from_slice(
        &[
            cast::<T>(3.0),
            cast::<T>(-1.0),
            cast::<T>(1.0),
            cast::<T>(1.0),
        ],
        &[2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();

    let expected =
        <crate::backend::CpuTensorLinalgBackend as crate::TensorLinalgPrims<T>>::solve_ex(
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
    let got = <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<T>>::solve_ex(
        &mut cuda_ctx,
        &a_gpu,
        &b_gpu,
    )
    .unwrap();

    assert_eq!(got.info, expected.info);
    assert_eq!(
        got.solution.logical_memory_space(),
        LogicalMemorySpace::GpuMemory { device_id: 0 }
    );
    assert_close_slice(
        "cuda solve_ex mixed batch",
        &tensor_data_on_cpu(&got.solution),
        &tensor_data_on_cpu(&expected.solution),
        cast::<T>(256.0) * T::epsilon(),
    );
}

#[cfg(feature = "cuda")]
fn cuda_solve_ex_broadcasted_rhs_matches_cpu_generic<T>()
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
    let b_cpu = Tensor::from_slice(
        &[cast::<T>(3.0), cast::<T>(-1.0)],
        &[2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();

    let expected =
        <crate::backend::CpuTensorLinalgBackend as crate::TensorLinalgPrims<T>>::solve_ex(
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
    let got = <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<T>>::solve_ex(
        &mut cuda_ctx,
        &a_gpu,
        &b_gpu,
    )
    .unwrap();

    assert_eq!(got.info, expected.info);
    assert_eq!(got.solution.dims(), &[2, 2]);
    assert_eq!(
        got.solution.logical_memory_space(),
        LogicalMemorySpace::GpuMemory { device_id: 0 }
    );
    assert_close_slice(
        "cuda solve_ex broadcast rhs",
        &tensor_data_on_cpu(&got.solution),
        &tensor_data_on_cpu(&expected.solution),
        cast::<T>(256.0) * T::epsilon(),
    );
}

#[cfg(feature = "cuda")]
fn cuda_solve_ex_does_not_treat_small_nonzero_pivot_as_zero_generic<T>()
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
    let b_cpu = Tensor::from_slice(
        &[cast::<T>(1.0), cast::<T>(2.0)],
        &[2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();

    let expected =
        <crate::backend::CpuTensorLinalgBackend as crate::TensorLinalgPrims<T>>::solve_ex(
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
    let got = <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<T>>::solve_ex(
        &mut cuda_ctx,
        &a_gpu,
        &b_gpu,
    )
    .unwrap();

    assert_eq!(got.info, expected.info);
    assert_close_slice(
        "cuda solve_ex small pivot",
        &tensor_data_on_cpu(&got.solution),
        &tensor_data_on_cpu(&expected.solution),
        cast::<T>(256.0) * T::epsilon(),
    );
}

#[cfg(feature = "cuda")]
fn cuda_solve_ex_preserves_complex_mixed_batch_complex32() {
    if !cuda_runtime_available() {
        return;
    }

    let path = cutensor_path().expect("TENFERRO_TEST_CUDA is set but libcutensor.so was not found");
    let (_backend, mut cuda_ctx) = tenferro_prims::CudaBackend::load(path).unwrap();
    let mut cpu_ctx = tenferro_prims::CpuContext::new(1);

    let a_cpu = Tensor::from_slice(
        &[
            Complex32::new(2.0, 1.0),
            Complex32::new(0.5, -0.25),
            Complex32::new(1.0, 0.0),
            Complex32::new(3.0, -1.0), //
            Complex32::new(1.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
        ],
        &[2, 2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let b_cpu = Tensor::from_slice(
        &[
            Complex32::new(5.0, 1.0),
            Complex32::new(4.0, -2.0), //
            Complex32::new(3.0, 0.0),
            Complex32::new(1.0, 0.5),
        ],
        &[2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let expected = <crate::backend::CpuTensorLinalgBackend as crate::TensorLinalgPrims<
        Complex32,
    >>::solve_ex(&mut cpu_ctx, &a_cpu, &b_cpu)
    .unwrap();

    let a_gpu = a_cpu
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let b_gpu = b_cpu
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let got = <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<Complex32>>::solve_ex(
        &mut cuda_ctx,
        &a_gpu,
        &b_gpu,
    )
    .unwrap();

    assert_eq!(got.info, expected.info);
    assert_eq!(
        got.solution.logical_memory_space(),
        LogicalMemorySpace::GpuMemory { device_id: 0 }
    );
    assert_close_complex_slice(
        "cuda solve_ex complex32",
        &tensor_data_on_cpu(&got.solution),
        &tensor_data_on_cpu(&expected.solution),
        256.0_f32 * f32::epsilon(),
    );
}

#[cfg(feature = "cuda")]
fn cuda_solve_ex_preserves_complex_mixed_batch_complex64() {
    if !cuda_runtime_available() {
        return;
    }

    let path = cutensor_path().expect("TENFERRO_TEST_CUDA is set but libcutensor.so was not found");
    let (_backend, mut cuda_ctx) = tenferro_prims::CudaBackend::load(path).unwrap();
    let mut cpu_ctx = tenferro_prims::CpuContext::new(1);

    let a_cpu = Tensor::from_slice(
        &[
            Complex64::new(2.0, 1.0),
            Complex64::new(0.5, -0.25),
            Complex64::new(1.0, 0.0),
            Complex64::new(3.0, -1.0), //
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
        &[2, 2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let b_cpu = Tensor::from_slice(
        &[
            Complex64::new(5.0, 1.0),
            Complex64::new(4.0, -2.0), //
            Complex64::new(3.0, 0.0),
            Complex64::new(1.0, 0.5),
        ],
        &[2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let expected = <crate::backend::CpuTensorLinalgBackend as crate::TensorLinalgPrims<
        Complex64,
    >>::solve_ex(&mut cpu_ctx, &a_cpu, &b_cpu)
    .unwrap();

    let a_gpu = a_cpu
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let b_gpu = b_cpu
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let got = <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<Complex64>>::solve_ex(
        &mut cuda_ctx,
        &a_gpu,
        &b_gpu,
    )
    .unwrap();

    assert_eq!(got.info, expected.info);
    assert_eq!(
        got.solution.logical_memory_space(),
        LogicalMemorySpace::GpuMemory { device_id: 0 }
    );
    assert_close_complex_slice(
        "cuda solve_ex complex64",
        &tensor_data_on_cpu(&got.solution),
        &tensor_data_on_cpu(&expected.solution),
        256.0_f64 * f64::epsilon(),
    );
}

#[cfg(feature = "cuda")]
fn cuda_solve_ex_reports_zero_pivot_for_complex_mixed_batch_complex32() {
    if !cuda_runtime_available() {
        return;
    }

    let path = cutensor_path().expect("TENFERRO_TEST_CUDA is set but libcutensor.so was not found");
    let (_backend, mut cuda_ctx) = tenferro_prims::CudaBackend::load(path).unwrap();
    let mut cpu_ctx = tenferro_prims::CpuContext::new(1);

    let a_cpu = Tensor::from_slice(
        &[
            Complex32::new(1.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(1.0, 0.0), //
            Complex32::new(1.0, 0.0),
            Complex32::new(2.0, 0.0),
            Complex32::new(2.0, 0.0),
            Complex32::new(4.0, 0.0),
        ],
        &[2, 2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let b_cpu = Tensor::from_slice(
        &[
            Complex32::new(3.0, -1.0),
            Complex32::new(1.0, 0.5), //
            Complex32::new(-1.0, -0.25),
            Complex32::new(-2.0, 0.5),
        ],
        &[2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let expected = <crate::backend::CpuTensorLinalgBackend as crate::TensorLinalgPrims<
        Complex32,
    >>::solve_ex(&mut cpu_ctx, &a_cpu, &b_cpu)
    .unwrap();

    let a_gpu = a_cpu
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let b_gpu = b_cpu
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let got = <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<Complex32>>::solve_ex(
        &mut cuda_ctx,
        &a_gpu,
        &b_gpu,
    )
    .unwrap();

    assert_eq!(expected.info, vec![0, 2]);
    assert_eq!(got.info, expected.info);
    let got_solution = tensor_data_on_cpu(&got.solution);
    let expected_solution = tensor_data_on_cpu(&expected.solution);
    assert_close_complex_slice(
        "cuda solve_ex complex32 zero pivot successful batch",
        &got_solution[..2],
        &expected_solution[..2],
        256.0_f32 * f32::epsilon(),
    );
}

#[cfg(feature = "cuda")]
fn cuda_solve_ex_reports_zero_pivot_for_complex_mixed_batch_complex64() {
    if !cuda_runtime_available() {
        return;
    }

    let path = cutensor_path().expect("TENFERRO_TEST_CUDA is set but libcutensor.so was not found");
    let (_backend, mut cuda_ctx) = tenferro_prims::CudaBackend::load(path).unwrap();
    let mut cpu_ctx = tenferro_prims::CpuContext::new(1);

    let a_cpu = Tensor::from_slice(
        &[
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0), //
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(4.0, 0.0),
        ],
        &[2, 2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let b_cpu = Tensor::from_slice(
        &[
            Complex64::new(3.0, -1.0),
            Complex64::new(1.0, 0.5), //
            Complex64::new(-1.0, -0.25),
            Complex64::new(-2.0, 0.5),
        ],
        &[2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let expected = <crate::backend::CpuTensorLinalgBackend as crate::TensorLinalgPrims<
        Complex64,
    >>::solve_ex(&mut cpu_ctx, &a_cpu, &b_cpu)
    .unwrap();

    let a_gpu = a_cpu
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let b_gpu = b_cpu
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let got = <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<Complex64>>::solve_ex(
        &mut cuda_ctx,
        &a_gpu,
        &b_gpu,
    )
    .unwrap();

    assert_eq!(expected.info, vec![0, 2]);
    assert_eq!(got.info, expected.info);
    let got_solution = tensor_data_on_cpu(&got.solution);
    let expected_solution = tensor_data_on_cpu(&expected.solution);
    assert_close_complex_slice(
        "cuda solve_ex complex64 zero pivot successful batch",
        &got_solution[..2],
        &expected_solution[..2],
        256.0_f64 * f64::epsilon(),
    );
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
fn cuda_solve_triangular_broadcasted_rhs_matches_cpu_upper_f32() {
    cuda_solve_triangular_broadcasted_rhs_matches_cpu_generic::<f32>(true);
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_solve_triangular_broadcasted_rhs_matches_cpu_upper_f64() {
    cuda_solve_triangular_broadcasted_rhs_matches_cpu_generic::<f64>(true);
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_solve_triangular_broadcasted_rhs_matches_cpu_lower_f32() {
    cuda_solve_triangular_broadcasted_rhs_matches_cpu_generic::<f32>(false);
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_solve_triangular_broadcasted_rhs_matches_cpu_lower_f64() {
    cuda_solve_triangular_broadcasted_rhs_matches_cpu_generic::<f64>(false);
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

#[cfg(feature = "cuda")]
#[test]
fn cuda_qr_reconstructs_small_complex32_matrix() {
    if !cuda_runtime_available() {
        return;
    }

    let path = cutensor_path().expect("TENFERRO_TEST_CUDA is set but libcutensor.so was not found");
    let (_backend, mut cuda_ctx) = tenferro_prims::CudaBackend::load(path).unwrap();

    let a_cpu = Tensor::from_slice(
        &[
            Complex32::new(cast::<f32>(1.0), cast::<f32>(0.5)),
            Complex32::new(cast::<f32>(2.0), cast::<f32>(-1.0)),
            Complex32::new(cast::<f32>(3.0), cast::<f32>(0.25)),
            Complex32::new(cast::<f32>(4.0), cast::<f32>(1.5)),
            Complex32::new(cast::<f32>(5.0), cast::<f32>(-0.75)),
            Complex32::new(cast::<f32>(6.0), cast::<f32>(0.0)),
        ],
        &[3, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let a_gpu = a_cpu
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();

    let got = <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<Complex32>>::qr(
        &mut cuda_ctx,
        &a_gpu,
    )
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
    let reconstructed = matmul_col_major_complex(&q, 3, 2, &r, 2);
    assert_close_complex_slice(
        "cuda complex qr reconstruction",
        &reconstructed,
        a_cpu.buffer().as_slice().unwrap(),
        cast::<f32>(2048.0) * f32::epsilon(),
    );

    let gram = gram_col_major_complex(&q, 3, 2);
    let mut identity = vec![Complex32::new(0.0, 0.0); 4];
    identity[0] = Complex32::new(1.0, 0.0);
    identity[3] = Complex32::new(1.0, 0.0);
    assert_close_complex_slice(
        "cuda complex qr orthogonality",
        &gram,
        &identity,
        cast::<f32>(2048.0) * f32::epsilon(),
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_qr_reconstructs_small_complex64_matrix() {
    if !cuda_runtime_available() {
        return;
    }

    let path = cutensor_path().expect("TENFERRO_TEST_CUDA is set but libcutensor.so was not found");
    let (_backend, mut cuda_ctx) = tenferro_prims::CudaBackend::load(path).unwrap();

    let a_cpu = Tensor::from_slice(
        &[
            Complex64::new(cast::<f64>(1.0), cast::<f64>(0.5)),
            Complex64::new(cast::<f64>(2.0), cast::<f64>(-1.0)),
            Complex64::new(cast::<f64>(3.0), cast::<f64>(0.25)),
            Complex64::new(cast::<f64>(4.0), cast::<f64>(1.5)),
            Complex64::new(cast::<f64>(5.0), cast::<f64>(-0.75)),
            Complex64::new(cast::<f64>(6.0), cast::<f64>(0.0)),
        ],
        &[3, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let a_gpu = a_cpu
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();

    let got = <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<Complex64>>::qr(
        &mut cuda_ctx,
        &a_gpu,
    )
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
    let reconstructed = matmul_col_major_complex(&q, 3, 2, &r, 2);
    assert_close_complex_slice(
        "cuda complex qr reconstruction",
        &reconstructed,
        a_cpu.buffer().as_slice().unwrap(),
        cast::<f64>(2048.0) * f64::epsilon(),
    );

    let gram = gram_col_major_complex(&q, 3, 2);
    let mut identity = vec![Complex64::new(0.0, 0.0); 4];
    identity[0] = Complex64::new(1.0, 0.0);
    identity[3] = Complex64::new(1.0, 0.0);
    assert_close_complex_slice(
        "cuda complex qr orthogonality",
        &gram,
        &identity,
        cast::<f64>(2048.0) * f64::epsilon(),
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_qr_reconstructs_wide_complex32_matrix() {
    if !cuda_runtime_available() {
        return;
    }

    let path = cutensor_path().expect("TENFERRO_TEST_CUDA is set but libcutensor.so was not found");
    let (_backend, mut cuda_ctx) = tenferro_prims::CudaBackend::load(path).unwrap();

    let a_cpu = Tensor::from_slice(
        &[
            Complex32::new(cast::<f32>(1.0), cast::<f32>(0.5)),
            Complex32::new(cast::<f32>(2.0), cast::<f32>(-1.0)),
            Complex32::new(cast::<f32>(3.0), cast::<f32>(0.25)),
            Complex32::new(cast::<f32>(4.0), cast::<f32>(1.5)),
            Complex32::new(cast::<f32>(5.0), cast::<f32>(-0.75)),
            Complex32::new(cast::<f32>(6.0), cast::<f32>(0.0)),
        ],
        &[2, 3],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let a_gpu = a_cpu
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();

    let got = <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<Complex32>>::qr(
        &mut cuda_ctx,
        &a_gpu,
    )
    .unwrap();

    assert_eq!(got.q.dims(), &[2, 2]);
    assert_eq!(got.r.dims(), &[2, 3]);
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
    let reconstructed = matmul_col_major_complex(&q, 2, 2, &r, 3);
    assert_close_complex_slice(
        "cuda complex qr wide reconstruction",
        &reconstructed,
        a_cpu.buffer().as_slice().unwrap(),
        cast::<f32>(2048.0) * f32::epsilon(),
    );

    let gram = gram_col_major_complex(&q, 2, 2);
    let identity = vec![
        Complex32::new(1.0, 0.0),
        Complex32::new(0.0, 0.0),
        Complex32::new(0.0, 0.0),
        Complex32::new(1.0, 0.0),
    ];
    assert_close_complex_slice(
        "cuda complex qr wide orthogonality",
        &gram,
        &identity,
        cast::<f32>(2048.0) * f32::epsilon(),
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_qr_reconstructs_wide_complex64_matrix() {
    if !cuda_runtime_available() {
        return;
    }

    let path = cutensor_path().expect("TENFERRO_TEST_CUDA is set but libcutensor.so was not found");
    let (_backend, mut cuda_ctx) = tenferro_prims::CudaBackend::load(path).unwrap();

    let a_cpu = Tensor::from_slice(
        &[
            Complex64::new(cast::<f64>(1.0), cast::<f64>(0.5)),
            Complex64::new(cast::<f64>(2.0), cast::<f64>(-1.0)),
            Complex64::new(cast::<f64>(3.0), cast::<f64>(0.25)),
            Complex64::new(cast::<f64>(4.0), cast::<f64>(1.5)),
            Complex64::new(cast::<f64>(5.0), cast::<f64>(-0.75)),
            Complex64::new(cast::<f64>(6.0), cast::<f64>(0.0)),
        ],
        &[2, 3],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let a_gpu = a_cpu
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();

    let got = <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<Complex64>>::qr(
        &mut cuda_ctx,
        &a_gpu,
    )
    .unwrap();

    assert_eq!(got.q.dims(), &[2, 2]);
    assert_eq!(got.r.dims(), &[2, 3]);
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
    let reconstructed = matmul_col_major_complex(&q, 2, 2, &r, 3);
    assert_close_complex_slice(
        "cuda complex qr wide reconstruction",
        &reconstructed,
        a_cpu.buffer().as_slice().unwrap(),
        cast::<f64>(2048.0) * f64::epsilon(),
    );

    let gram = gram_col_major_complex(&q, 2, 2);
    let identity = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
    ];
    assert_close_complex_slice(
        "cuda complex qr wide orthogonality",
        &gram,
        &identity,
        cast::<f64>(2048.0) * f64::epsilon(),
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_qr_reconstructs_batched_complex32_matrix() {
    if !cuda_runtime_available() {
        return;
    }

    let path = cutensor_path().expect("TENFERRO_TEST_CUDA is set but libcutensor.so was not found");
    let (_backend, mut cuda_ctx) = tenferro_prims::CudaBackend::load(path).unwrap();

    let a_cpu = Tensor::from_slice(
        &[
            Complex32::new(1.0, 0.5),
            Complex32::new(2.0, -1.0),
            Complex32::new(3.0, 0.25),
            Complex32::new(4.0, 1.5),
            Complex32::new(5.0, -0.75),
            Complex32::new(6.0, 0.0),
            Complex32::new(1.0, -0.25),
            Complex32::new(2.0, 0.75),
            Complex32::new(3.0, -1.5),
            Complex32::new(4.0, 0.5),
            Complex32::new(5.0, 0.0),
            Complex32::new(6.0, 0.25),
        ],
        &[3, 2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let a_gpu = a_cpu
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();

    let got = <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<Complex32>>::qr(
        &mut cuda_ctx,
        &a_gpu,
    )
    .unwrap();

    assert_eq!(got.q.dims(), &[3, 2, 2]);
    assert_eq!(got.r.dims(), &[2, 2, 2]);
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
    let input = tensor_data_on_cpu(&a_cpu);
    let matrix_stride = 3 * 2;
    let r_stride = 2 * 2;
    for batch in 0..2 {
        let q_slice = &q[batch * matrix_stride..(batch + 1) * matrix_stride];
        let r_slice = &r[batch * r_stride..(batch + 1) * r_stride];
        let input_slice = &input[batch * matrix_stride..(batch + 1) * matrix_stride];
        let reconstructed = matmul_col_major_complex(q_slice, 3, 2, r_slice, 2);
        assert_close_complex_slice(
            if batch == 0 {
                "cuda complex qr batched reconstruction batch0"
            } else {
                "cuda complex qr batched reconstruction batch1"
            },
            &reconstructed,
            input_slice,
            cast::<f32>(4096.0) * f32::epsilon(),
        );
        let gram = gram_col_major_complex(q_slice, 3, 2);
        let identity = vec![
            Complex32::new(1.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(1.0, 0.0),
        ];
        assert_close_complex_slice(
            if batch == 0 {
                "cuda complex qr batched orthogonality batch0"
            } else {
                "cuda complex qr batched orthogonality batch1"
            },
            &gram,
            &identity,
            cast::<f32>(4096.0) * f32::epsilon(),
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_qr_reconstructs_batched_complex64_matrix() {
    if !cuda_runtime_available() {
        return;
    }

    let path = cutensor_path().expect("TENFERRO_TEST_CUDA is set but libcutensor.so was not found");
    let (_backend, mut cuda_ctx) = tenferro_prims::CudaBackend::load(path).unwrap();

    let a_cpu = Tensor::from_slice(
        &[
            Complex64::new(1.0, 0.5),
            Complex64::new(2.0, -1.0),
            Complex64::new(3.0, 0.25),
            Complex64::new(4.0, 1.5),
            Complex64::new(5.0, -0.75),
            Complex64::new(6.0, 0.0),
            Complex64::new(1.0, -0.25),
            Complex64::new(2.0, 0.75),
            Complex64::new(3.0, -1.5),
            Complex64::new(4.0, 0.5),
            Complex64::new(5.0, 0.0),
            Complex64::new(6.0, 0.25),
        ],
        &[3, 2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let a_gpu = a_cpu
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();

    let got = <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<Complex64>>::qr(
        &mut cuda_ctx,
        &a_gpu,
    )
    .unwrap();

    assert_eq!(got.q.dims(), &[3, 2, 2]);
    assert_eq!(got.r.dims(), &[2, 2, 2]);
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
    let input = tensor_data_on_cpu(&a_cpu);
    let matrix_stride = 3 * 2;
    let r_stride = 2 * 2;
    for batch in 0..2 {
        let q_slice = &q[batch * matrix_stride..(batch + 1) * matrix_stride];
        let r_slice = &r[batch * r_stride..(batch + 1) * r_stride];
        let input_slice = &input[batch * matrix_stride..(batch + 1) * matrix_stride];
        let reconstructed = matmul_col_major_complex(q_slice, 3, 2, r_slice, 2);
        assert_close_complex_slice(
            if batch == 0 {
                "cuda complex qr batched reconstruction batch0"
            } else {
                "cuda complex qr batched reconstruction batch1"
            },
            &reconstructed,
            input_slice,
            cast::<f64>(4096.0) * f64::epsilon(),
        );
        let gram = gram_col_major_complex(q_slice, 3, 2);
        let identity = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ];
        assert_close_complex_slice(
            if batch == 0 {
                "cuda complex qr batched orthogonality batch0"
            } else {
                "cuda complex qr batched orthogonality batch1"
            },
            &gram,
            &identity,
            cast::<f64>(4096.0) * f64::epsilon(),
        );
    }
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
fn cuda_lu_factor_matches_cpu_for_small_complex_matrix_c32() {
    cuda_lu_factor_matches_cpu_for_small_complex32_matrix();
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_lu_factor_matches_cpu_for_small_complex_matrix_c64() {
    cuda_lu_factor_matches_cpu_for_small_complex64_matrix();
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_lu_factor_ex_matches_cpu_for_complex_mixed_batch_c32() {
    cuda_lu_factor_ex_matches_cpu_for_complex_mixed_batch_complex32();
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_lu_factor_ex_matches_cpu_for_complex_mixed_batch_c64() {
    cuda_lu_factor_ex_matches_cpu_for_complex_mixed_batch_complex64();
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_lu_factor_ex_reports_zero_pivot_for_complex_mixed_batch_c32() {
    cuda_lu_factor_ex_reports_zero_pivot_for_complex_mixed_batch_complex32();
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_lu_factor_ex_reports_zero_pivot_for_complex_mixed_batch_c64() {
    cuda_lu_factor_ex_reports_zero_pivot_for_complex_mixed_batch_complex64();
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_solve_matches_cpu_for_small_complex_matrix_c32() {
    cuda_solve_matches_cpu_for_small_complex32_matrix();
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_solve_matches_cpu_for_small_complex_matrix_c64() {
    cuda_solve_matches_cpu_for_small_complex64_matrix();
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_solve_ex_preserves_complex_mixed_batch_c32() {
    cuda_solve_ex_preserves_complex_mixed_batch_complex32();
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_solve_ex_preserves_complex_mixed_batch_c64() {
    cuda_solve_ex_preserves_complex_mixed_batch_complex64();
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_solve_ex_reports_zero_pivot_for_complex_mixed_batch_c32() {
    cuda_solve_ex_reports_zero_pivot_for_complex_mixed_batch_complex32();
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_solve_ex_reports_zero_pivot_for_complex_mixed_batch_c64() {
    cuda_solve_ex_reports_zero_pivot_for_complex_mixed_batch_complex64();
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
fn cuda_cholesky_matches_cpu_for_small_complex_matrix_c32() {
    cuda_cholesky_matches_cpu_for_small_complex32_matrix();
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_cholesky_matches_cpu_for_small_complex_matrix_c64() {
    cuda_cholesky_matches_cpu_for_small_complex64_matrix();
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_cholesky_ex_reports_minor_for_complex_non_spd_matrix_c32() {
    cuda_cholesky_ex_reports_minor_for_complex_non_spd_matrix_complex32();
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_cholesky_ex_reports_minor_for_complex_non_spd_matrix_c64() {
    cuda_cholesky_ex_reports_minor_for_complex_non_spd_matrix_complex64();
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

#[test]
#[cfg(feature = "cuda")]
fn cuda_svdvals_matches_cpu_for_small_tall_complex_matrix_c32() {
    cuda_svdvals_matches_cpu_for_small_complex_matrix_generic::<f32>(
        false,
        &[
            Complex32::new(3.0, 0.5),
            Complex32::new(1.0, -0.25),
            Complex32::new(0.0, 0.5),
            Complex32::new(1.0, 0.25),
            Complex32::new(2.0, -1.0),
            Complex32::new(1.0, 0.0),
        ],
    );
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_svdvals_matches_cpu_for_small_tall_complex_matrix_c64() {
    cuda_svdvals_matches_cpu_for_small_complex_matrix_generic::<f64>(
        false,
        &[
            Complex64::new(3.0, 0.5),
            Complex64::new(1.0, -0.25),
            Complex64::new(0.0, 0.5),
            Complex64::new(1.0, 0.25),
            Complex64::new(2.0, -1.0),
            Complex64::new(1.0, 0.0),
        ],
    );
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_svdvals_matches_cpu_for_small_wide_complex_matrix_c32() {
    cuda_svdvals_matches_cpu_for_small_complex_matrix_generic::<f32>(
        true,
        &[
            Complex32::new(3.0, 0.5),
            Complex32::new(1.0, -0.25),
            Complex32::new(1.0, 0.25),
            Complex32::new(2.0, -1.0),
            Complex32::new(0.0, 0.5),
            Complex32::new(1.0, 0.0),
        ],
    );
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_svdvals_matches_cpu_for_small_wide_complex_matrix_c64() {
    cuda_svdvals_matches_cpu_for_small_complex_matrix_generic::<f64>(
        true,
        &[
            Complex64::new(3.0, 0.5),
            Complex64::new(1.0, -0.25),
            Complex64::new(1.0, 0.25),
            Complex64::new(2.0, -1.0),
            Complex64::new(0.0, 0.5),
            Complex64::new(1.0, 0.0),
        ],
    );
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

#[cfg(feature = "cuda")]
fn cuda_thin_svd_matches_cpu_for_small_complex_matrix_generic<T>(
    wide: bool,
    data: &[num_complex::Complex<T>],
) where
    T: crate::KernelLinalgScalar<Real = T> + Float + std::fmt::Debug + Send + Sync,
    num_complex::Complex<T>: crate::KernelLinalgScalar<Real = T>
        + super::scalar_type::CudaLinalgScalar
        + std::fmt::Debug,
{
    if !cuda_runtime_available() {
        return;
    }

    let path = cutensor_path().expect("TENFERRO_TEST_CUDA is set but libcutensor.so was not found");
    let (_backend, mut cuda_ctx) = tenferro_prims::CudaBackend::load(path).unwrap();
    let mut cpu_ctx = tenferro_prims::CpuContext::new(1);

    let dims = if wide { [2, 3] } else { [3, 2] };
    let a_cpu = Tensor::from_slice(data, &dims, MemoryOrder::ColumnMajor).unwrap();
    let expected = <crate::backend::CpuTensorLinalgBackend as crate::TensorLinalgPrims<
        num_complex::Complex<T>,
    >>::thin_svd(&mut cpu_ctx, &a_cpu)
    .unwrap();

    let a_gpu = a_cpu
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let got = <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<
        num_complex::Complex<T>,
    >>::thin_svd(&mut cuda_ctx, &a_gpu)
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
            "cuda complex thin_svd singular values wide"
        } else {
            "cuda complex thin_svd singular values tall"
        },
        &tensor_data_on_cpu(&got.s),
        &tensor_data_on_cpu(&expected.s),
        cast::<T>(4096.0) * T::epsilon(),
    );

    let u = tensor_data_on_cpu(&got.u);
    let s = tensor_data_on_cpu(&got.s);
    let vt = tensor_data_on_cpu(&got.vt);
    let reconstructed =
        reconstruct_thin_svd_col_major_complex(&u, a_cpu.dims()[0], k, &s, &vt, a_cpu.dims()[1]);
    assert_close_complex_slice(
        if wide {
            "cuda complex thin_svd reconstruction wide"
        } else {
            "cuda complex thin_svd reconstruction tall"
        },
        &reconstructed,
        a_cpu.buffer().as_slice().unwrap(),
        cast::<T>(8192.0) * T::epsilon(),
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

#[test]
#[cfg(feature = "cuda")]
fn cuda_thin_svd_matches_cpu_for_small_tall_complex_matrix_c32() {
    cuda_thin_svd_matches_cpu_for_small_complex_matrix_generic::<f32>(
        false,
        &[
            Complex32::new(3.0, 0.5),
            Complex32::new(1.0, -0.25),
            Complex32::new(0.0, 0.5),
            Complex32::new(1.0, 0.25),
            Complex32::new(2.0, -1.0),
            Complex32::new(1.0, 0.0),
        ],
    );
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_thin_svd_matches_cpu_for_small_tall_complex_matrix_c64() {
    cuda_thin_svd_matches_cpu_for_small_complex_matrix_generic::<f64>(
        false,
        &[
            Complex64::new(3.0, 0.5),
            Complex64::new(1.0, -0.25),
            Complex64::new(0.0, 0.5),
            Complex64::new(1.0, 0.25),
            Complex64::new(2.0, -1.0),
            Complex64::new(1.0, 0.0),
        ],
    );
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_thin_svd_matches_cpu_for_small_wide_complex_matrix_c32() {
    cuda_thin_svd_matches_cpu_for_small_complex_matrix_generic::<f32>(
        true,
        &[
            Complex32::new(3.0, 0.5),
            Complex32::new(1.0, -0.25),
            Complex32::new(1.0, 0.25),
            Complex32::new(2.0, -1.0),
            Complex32::new(0.0, 0.5),
            Complex32::new(1.0, 0.0),
        ],
    );
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_thin_svd_matches_cpu_for_small_wide_complex_matrix_c64() {
    cuda_thin_svd_matches_cpu_for_small_complex_matrix_generic::<f64>(
        true,
        &[
            Complex64::new(3.0, 0.5),
            Complex64::new(1.0, -0.25),
            Complex64::new(1.0, 0.25),
            Complex64::new(2.0, -1.0),
            Complex64::new(0.0, 0.5),
            Complex64::new(1.0, 0.0),
        ],
    );
}
