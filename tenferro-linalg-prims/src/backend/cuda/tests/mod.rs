use crate::LinalgCapabilityOp;
use num_traits::{Float, NumCast};
use tenferro_device::LogicalMemorySpace;
use tenferro_tensor::{MemoryOrder, Tensor};

fn cuda_runtime_available() -> bool {
    std::env::var_os("TENFERRO_TEST_CUDA").is_some()
}

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
    assert!(
        <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<f64>>::has_linalg_support(
            LinalgCapabilityOp::Solve
        )
    );
    assert!(
        <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<f64>>::has_linalg_support(
            LinalgCapabilityOp::LuFactor
        )
    );
    assert!(
        <super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<f64>>::has_linalg_support(
            LinalgCapabilityOp::LuFactorEx
        )
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
        !<super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<f64>>::has_linalg_support(
            LinalgCapabilityOp::ThinSvd
        )
    );
    assert!(
        !<super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<num_complex::Complex64>>::has_linalg_support(
            LinalgCapabilityOp::Solve
        )
    );
}

#[test]
fn cuda_backend_reports_ex_capabilities_only_when_wired() {
    for op in [LinalgCapabilityOp::SolveEx, LinalgCapabilityOp::CholeskyEx] {
        assert!(
            !<super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<f64>>::has_linalg_support(
                op
            ),
            "CUDA should not report {op:?} before native EX kernels exist",
        );
    }
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
fn cuda_solve_matches_cpu_for_small_real_matrix_f32() {
    cuda_solve_matches_cpu_for_small_real_matrix_generic::<f32>();
}

#[test]
fn cuda_solve_matches_cpu_for_small_real_matrix_f64() {
    cuda_solve_matches_cpu_for_small_real_matrix_generic::<f64>();
}

#[test]
fn cuda_lu_factor_matches_cpu_for_small_real_matrix_f32() {
    cuda_lu_factor_matches_cpu_for_small_real_matrix_generic::<f32>();
}

#[test]
fn cuda_lu_factor_matches_cpu_for_small_real_matrix_f64() {
    cuda_lu_factor_matches_cpu_for_small_real_matrix_generic::<f64>();
}

#[test]
fn cuda_lu_factor_ex_matches_cpu_for_mixed_batch_f32() {
    cuda_lu_factor_ex_matches_cpu_for_mixed_batch_generic::<f32>();
}

#[test]
fn cuda_lu_factor_ex_matches_cpu_for_mixed_batch_f64() {
    cuda_lu_factor_ex_matches_cpu_for_mixed_batch_generic::<f64>();
}
