mod complex;
mod diagonal;
mod organization;

use crate::SemiringBinaryOp;
use num_complex::Complex64;
use tenferro_algebra::Standard;
use tenferro_device::LogicalMemorySpace;
use tenferro_device::Result;
use tenferro_tensor::MemoryOrder;
use tenferro_tensor::Tensor;

use super::*;
use crate::{
    ComplexScalePrimsDescriptor, CpuBackend, CpuContext, ScalarBinaryOp, ScalarPrimsDescriptor,
    ScalarReductionOp, ScalarUnaryOp, TensorComplexScalePrims, TensorScalarPrims,
};

#[test]
fn cuda_backend_feature_surface_matches_family_contracts() {
    let _core_plan_fn: fn(
        &mut CudaContext,
        &SemiringCoreDescriptor,
        &[&[usize]],
    ) -> Result<CudaPlan<f64>> = <CudaBackend as TensorSemiringCore<Standard<f64>>>::plan;
    let _fast_plan_fn: fn(
        &mut CudaContext,
        &SemiringFastPathDescriptor,
        &[&[usize]],
    ) -> Result<CudaPlan<f64>> = <CudaBackend as TensorSemiringFastPath<Standard<f64>>>::plan;
    let _execute_fn: fn(
        &mut CudaContext,
        &CudaPlan<f64>,
        f64,
        &[&Tensor<f64>],
        f64,
        &mut Tensor<f64>,
    ) -> Result<()> = <CudaBackend as TensorSemiringCore<Standard<f64>>>::execute;

    assert!(
        <CudaBackend as TensorSemiringFastPath<Standard<f64>>>::has_fast_path(
            SemiringFastPathDescriptor::Contract {
                modes_a: vec![0],
                modes_b: vec![0],
                modes_c: vec![0],
            }
        )
    );
    assert!(
        <CudaBackend as TensorSemiringFastPath<Standard<f64>>>::has_fast_path(
            SemiringFastPathDescriptor::ElementwiseBinary {
                op: SemiringBinaryOp::Mul,
            }
        )
    );
}

fn available_cutensor_library_path() -> Option<&'static str> {
    [
        "/usr/lib/x86_64-linux-gnu/libcutensor.so",
        "/usr/lib/x86_64-linux-gnu/libcutensor.so.2",
        "/usr/lib/x86_64-linux-gnu/libcutensor/12/libcutensor.so",
        "/usr/lib/x86_64-linux-gnu/libcutensor/12/libcutensor.so.2",
    ]
    .into_iter()
    .find(|path| std::path::Path::new(path).exists())
}

fn cuda_runtime_is_available() -> Option<&'static str> {
    let path = available_cutensor_library_path()?;
    if !cuda_device_zero_is_available() {
        return None;
    }
    Some(path)
}

fn cuda_device_zero_is_available() -> bool {
    std::panic::catch_unwind(|| {
        cudarc::runtime::result::device::get_count()
            .map(|count| count > 0)
            .unwrap_or(false)
    })
    .unwrap_or(false)
}

fn tensor_f64(data: &[f64], dims: &[usize]) -> Tensor<f64> {
    Tensor::from_slice(data, dims, MemoryOrder::ColumnMajor).unwrap()
}

fn tensor_f32(data: &[f32], dims: &[usize]) -> Tensor<f32> {
    Tensor::from_slice(data, dims, MemoryOrder::ColumnMajor).unwrap()
}

fn tensor_c64(data: &[Complex64], dims: &[usize]) -> Tensor<Complex64> {
    Tensor::from_slice(data, dims, MemoryOrder::ColumnMajor).unwrap()
}

fn tensor_f64_from_col_major_fn<F>(dims: &[usize], mut f: F) -> Tensor<f64>
where
    F: FnMut(&[usize]) -> f64,
{
    let len = dims.iter().copied().product::<usize>();
    let mut data = Vec::with_capacity(len);
    let mut idx = vec![0usize; dims.len()];

    if dims.is_empty() {
        data.push(f(&idx));
    } else {
        loop {
            data.push(f(&idx));

            let mut axis = 0usize;
            while axis < dims.len() {
                idx[axis] += 1;
                if idx[axis] < dims[axis] {
                    break;
                }
                idx[axis] = 0;
                axis += 1;
            }

            if axis == dims.len() {
                break;
            }
        }
    }

    Tensor::from_slice(&data, dims, MemoryOrder::ColumnMajor).unwrap()
}

fn load_cuda_backend() -> Option<(CudaBackend, CudaContext)> {
    let path = available_cutensor_library_path()?;
    if !cuda_device_zero_is_available() {
        return None;
    }
    Some(CudaBackend::load(path).unwrap())
}

#[test]
fn cuda_complex_scale_phase1_advertises_pointwise_mul_only_when_runtime_is_wired() {
    let desc = ComplexScalePrimsDescriptor::PointwiseMul;
    let supported =
        <CudaBackend as TensorComplexScalePrims<Complex64>>::has_complex_scale_support(desc);

    if cfg!(feature = "cuda") {
        assert!(supported);
    } else {
        assert!(!supported);
    }
}

#[test]
fn cuda_complex_scale_phase1_pointwise_mul_matches_cpu_when_runtime_is_available() {
    let Some((_backend, mut cuda_ctx)) = load_cuda_backend() else {
        return;
    };

    let desc = ComplexScalePrimsDescriptor::PointwiseMul;
    let lhs = tensor_c64(
        &[
            Complex64::new(1.0, -2.0),
            Complex64::new(-3.0, 4.0),
            Complex64::new(5.0, 0.5),
            Complex64::new(-7.0, -1.5),
        ],
        &[2, 2],
    );
    let rhs = tensor_f64(&[2.0_f64, -0.5, 3.0, 4.0], &[2, 2]);
    let plan = <CudaBackend as TensorComplexScalePrims<Complex64>>::plan(
        &mut cuda_ctx,
        &desc,
        &[lhs.dims(), rhs.dims(), lhs.dims()],
    )
    .unwrap();
    let lhs_gpu = lhs
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let rhs_gpu = rhs
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let mut output_gpu = Tensor::<Complex64>::zeros(
        lhs.dims(),
        LogicalMemorySpace::GpuMemory { device_id: 0 },
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    <CudaBackend as TensorComplexScalePrims<Complex64>>::execute(
        &mut cuda_ctx,
        &plan,
        Complex64::new(1.0, 0.0),
        &lhs_gpu,
        &rhs_gpu,
        Complex64::new(0.0, 0.0),
        &mut output_gpu,
    )
    .unwrap();

    let output = output_gpu
        .to_memory_space_async(LogicalMemorySpace::MainMemory)
        .unwrap();
    assert_eq!(
        output.buffer().as_slice(),
        Some(
            &[
                Complex64::new(2.0, -4.0),
                Complex64::new(1.5, -2.0),
                Complex64::new(15.0, 1.5),
                Complex64::new(-28.0, -6.0),
            ][..]
        )
    );
}

#[test]
fn cuda_batched_gemm_matches_cpu_for_small_real_batched_case() {
    let Some((_backend, mut cuda_ctx)) = load_cuda_backend() else {
        return;
    };

    let mut cpu_ctx = CpuContext::new(1);
    let desc = SemiringCoreDescriptor::BatchedGemm {
        batch_dims: vec![2],
        m: 2,
        n: 2,
        k: 3,
    };

    let a = tensor_f64_from_col_major_fn(&[2, 3, 2], |idx| {
        let m = idx[0] as f64;
        let k = idx[1] as f64;
        let batch = idx[2] as f64;
        1.0 + m + 10.0 * k + 100.0 * batch
    });
    let b = tensor_f64_from_col_major_fn(&[3, 2, 2], |idx| {
        let k = idx[0] as f64;
        let n = idx[1] as f64;
        let batch = idx[2] as f64;
        2.0 + n + 10.0 * k + 100.0 * batch
    });

    let cpu_plan = <CpuBackend as TensorSemiringCore<Standard<f64>>>::plan(
        &mut cpu_ctx,
        &desc,
        &[a.dims(), b.dims(), &[2, 2, 2]],
    )
    .unwrap();
    let cuda_plan = <CudaBackend as TensorSemiringCore<Standard<f64>>>::plan(
        &mut cuda_ctx,
        &desc,
        &[a.dims(), b.dims(), &[2, 2, 2]],
    )
    .unwrap();

    let mut c_cpu = Tensor::<f64>::zeros(
        &[2, 2, 2],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    <CpuBackend as TensorSemiringCore<Standard<f64>>>::execute(
        &mut cpu_ctx,
        &cpu_plan,
        1.0,
        &[&a, &b],
        0.0,
        &mut c_cpu,
    )
    .unwrap();

    let a_gpu = a
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let b_gpu = b
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let mut c_gpu = Tensor::<f64>::zeros(
        &[2, 2, 2],
        LogicalMemorySpace::GpuMemory { device_id: 0 },
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    <CudaBackend as TensorSemiringCore<Standard<f64>>>::execute(
        &mut cuda_ctx,
        &cuda_plan,
        1.0,
        &[&a_gpu, &b_gpu],
        0.0,
        &mut c_gpu,
    )
    .unwrap();

    let c_cuda = c_gpu
        .to_memory_space_async(LogicalMemorySpace::MainMemory)
        .unwrap();
    let cpu_slice = c_cpu.buffer().as_slice().unwrap();
    let cuda_slice = c_cuda.buffer().as_slice().unwrap();
    assert_eq!(cpu_slice.len(), cuda_slice.len());
    for (i, (lhs, rhs)) in cpu_slice.iter().zip(cuda_slice.iter()).enumerate() {
        assert!(
            (lhs - rhs).abs() <= 1.0e-9,
            "batched GEMM mismatch at flat index {i}: cpu={lhs:?}, cuda={rhs:?}"
        );
    }
}

#[test]
fn cuda_make_contiguous_smoke_runs_on_device_tensors_when_runtime_is_available() {
    let Some(path) = available_cutensor_library_path() else {
        return;
    };

    if !cuda_device_zero_is_available() {
        return;
    }

    let (_backend, mut ctx) = CudaBackend::load(path).unwrap();
    let base = Tensor::<f32>::from_slice(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[2, 3],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let input = base.permute(&[1, 0]).unwrap();
    let plan = <CudaBackend as TensorSemiringCore<Standard<f32>>>::plan(
        &mut ctx,
        &SemiringCoreDescriptor::MakeContiguous,
        &[&[3, 2], &[3, 2]],
    )
    .unwrap();
    let input_gpu = input
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let mut output_gpu = Tensor::<f32>::zeros(
        &[3, 2],
        LogicalMemorySpace::GpuMemory { device_id: 0 },
        MemoryOrder::ColumnMajor,
    )
    .unwrap();

    <CudaBackend as TensorSemiringCore<Standard<f32>>>::execute(
        &mut ctx,
        &plan,
        1.0,
        &[&input_gpu],
        0.0,
        &mut output_gpu,
    )
    .unwrap();

    let output_cpu = output_gpu
        .to_memory_space_async(LogicalMemorySpace::MainMemory)
        .unwrap();

    assert!(output_cpu.is_contiguous());
    assert_eq!(output_cpu.dims(), &[3, 2]);
    assert_eq!(
        output_cpu.buffer().as_slice(),
        Some(&[1.0, 3.0, 5.0, 2.0, 4.0, 6.0][..])
    );
}

#[test]
fn cuda_context_uses_shared_device_runtime() {
    let Some(path) = available_cutensor_library_path() else {
        return;
    };
    if !cuda_device_zero_is_available() {
        return;
    }

    let (_backend, ctx) = CudaBackend::load(path).unwrap();
    assert_eq!(ctx.device_id(), 0);
    assert_eq!(ctx.shared_runtime().device_id(), 0);
}

#[test]
fn cuda_resolve_conj_keeps_tensor_on_device_and_matches_cpu() {
    let Some(path) = available_cutensor_library_path() else {
        return;
    };
    if !cuda_device_zero_is_available() {
        return;
    }

    let (_backend, mut ctx) = CudaBackend::load(path).unwrap();
    let cpu = Tensor::from_slice(
        &[
            Complex64::new(1.0, 2.0),
            Complex64::new(3.0, -4.0),
            Complex64::new(-5.0, 6.0),
            Complex64::new(7.0, 8.0),
        ],
        &[2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let gpu = cpu
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap()
        .conj();

    let expected = Tensor::stack(&[&gpu], 0).unwrap().squeeze_dim(0).unwrap();
    assert_eq!(
        expected.logical_memory_space(),
        LogicalMemorySpace::GpuMemory { device_id: 0 }
    );
    let resolved = CudaBackend::resolve_conj(&mut ctx, &gpu);
    assert_eq!(
        resolved.logical_memory_space(),
        LogicalMemorySpace::GpuMemory { device_id: 0 }
    );
    assert!(!resolved.is_conjugated());

    let round_trip = resolved
        .to_memory_space_async(LogicalMemorySpace::MainMemory)
        .unwrap();
    let expected_round_trip = expected
        .to_memory_space_async(LogicalMemorySpace::MainMemory)
        .unwrap();
    assert_eq!(
        round_trip.buffer().as_slice(),
        expected_round_trip.buffer().as_slice()
    );
}

#[test]
fn cuda_scalar_add_and_abs_match_cpu() {
    let Some((_backend, mut cuda_ctx)) = load_cuda_backend() else {
        return;
    };

    let mut cpu_ctx = CpuContext::new(1);
    let lhs_base = tensor_f64(&[1.0, -2.0, 3.5, -4.5, 5.0, -6.0], &[2, 3]);
    let rhs_base = tensor_f64(&[-0.5, 1.5, -2.5, 3.5, -4.5, 5.5], &[2, 3]);
    let lhs = lhs_base.permute(&[1, 0]).unwrap();
    let rhs = rhs_base.permute(&[1, 0]).unwrap();

    let add_desc = ScalarPrimsDescriptor::PointwiseBinary {
        op: ScalarBinaryOp::Add,
    };
    let add_plan_cpu = <CpuBackend as TensorScalarPrims<Standard<f64>>>::plan(
        &mut cpu_ctx,
        &add_desc,
        &[lhs.dims(), rhs.dims(), lhs.dims()],
    )
    .unwrap();
    let add_plan_cuda = <CudaBackend as TensorScalarPrims<Standard<f64>>>::plan(
        &mut cuda_ctx,
        &add_desc,
        &[lhs.dims(), rhs.dims(), lhs.dims()],
    )
    .unwrap();

    let mut add_out_cpu = Tensor::<f64>::zeros(
        lhs.dims(),
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    <CpuBackend as TensorScalarPrims<Standard<f64>>>::execute(
        &mut cpu_ctx,
        &add_plan_cpu,
        1.0,
        &[&lhs, &rhs],
        0.0,
        &mut add_out_cpu,
    )
    .unwrap();

    let lhs_gpu = lhs
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let rhs_gpu = rhs
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let mut add_out_gpu = Tensor::<f64>::zeros(
        lhs.dims(),
        LogicalMemorySpace::GpuMemory { device_id: 0 },
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    <CudaBackend as TensorScalarPrims<Standard<f64>>>::execute(
        &mut cuda_ctx,
        &add_plan_cuda,
        1.0,
        &[&lhs_gpu, &rhs_gpu],
        0.0,
        &mut add_out_gpu,
    )
    .unwrap();

    let add_out_cuda = add_out_gpu
        .to_memory_space_async(LogicalMemorySpace::MainMemory)
        .unwrap();
    assert_eq!(
        add_out_cuda.buffer().as_slice(),
        add_out_cpu.buffer().as_slice()
    );

    let abs_desc = ScalarPrimsDescriptor::PointwiseUnary {
        op: ScalarUnaryOp::Abs,
    };
    let abs_plan_cpu = <CpuBackend as TensorScalarPrims<Standard<f64>>>::plan(
        &mut cpu_ctx,
        &abs_desc,
        &[add_out_cpu.dims(), add_out_cpu.dims()],
    )
    .unwrap();
    let abs_plan_cuda = <CudaBackend as TensorScalarPrims<Standard<f64>>>::plan(
        &mut cuda_ctx,
        &abs_desc,
        &[add_out_cpu.dims(), add_out_cpu.dims()],
    )
    .unwrap();

    let mut abs_out_cpu = Tensor::<f64>::zeros(
        add_out_cpu.dims(),
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    <CpuBackend as TensorScalarPrims<Standard<f64>>>::execute(
        &mut cpu_ctx,
        &abs_plan_cpu,
        1.0,
        &[&add_out_cpu],
        0.0,
        &mut abs_out_cpu,
    )
    .unwrap();

    let mut abs_out_gpu = Tensor::<f64>::zeros(
        add_out_cpu.dims(),
        LogicalMemorySpace::GpuMemory { device_id: 0 },
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    <CudaBackend as TensorScalarPrims<Standard<f64>>>::execute(
        &mut cuda_ctx,
        &abs_plan_cuda,
        1.0,
        &[&add_out_gpu],
        0.0,
        &mut abs_out_gpu,
    )
    .unwrap();

    let abs_out_cuda = abs_out_gpu
        .to_memory_space_async(LogicalMemorySpace::MainMemory)
        .unwrap();
    assert_eq!(
        abs_out_cuda.buffer().as_slice(),
        abs_out_cpu.buffer().as_slice()
    );
}

#[test]
fn cuda_scalar_sum_reduction_matches_cpu() {
    let Some((_backend, mut cuda_ctx)) = load_cuda_backend() else {
        return;
    };

    let mut cpu_ctx = CpuContext::new(1);
    let input_base = tensor_f64(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let input = input_base.permute(&[1, 0]).unwrap();
    let desc = ScalarPrimsDescriptor::Reduction {
        modes_a: vec![0, 1],
        modes_c: vec![1],
        op: ScalarReductionOp::Sum,
    };
    let plan_cpu = <CpuBackend as TensorScalarPrims<Standard<f64>>>::plan(
        &mut cpu_ctx,
        &desc,
        &[input.dims(), &[2]],
    )
    .unwrap();
    let plan_cuda = <CudaBackend as TensorScalarPrims<Standard<f64>>>::plan(
        &mut cuda_ctx,
        &desc,
        &[input.dims(), &[2]],
    )
    .unwrap();

    let mut out_cpu = Tensor::<f64>::zeros(
        &[2],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    <CpuBackend as TensorScalarPrims<Standard<f64>>>::execute(
        &mut cpu_ctx,
        &plan_cpu,
        1.0,
        &[&input],
        0.0,
        &mut out_cpu,
    )
    .unwrap();

    let input_gpu = input
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let mut out_gpu = Tensor::<f64>::zeros(
        &[2],
        LogicalMemorySpace::GpuMemory { device_id: 0 },
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    <CudaBackend as TensorScalarPrims<Standard<f64>>>::execute(
        &mut cuda_ctx,
        &plan_cuda,
        1.0,
        &[&input_gpu],
        0.0,
        &mut out_gpu,
    )
    .unwrap();

    let out_cuda = out_gpu
        .to_memory_space_async(LogicalMemorySpace::MainMemory)
        .unwrap();
    assert_eq!(out_cuda.buffer().as_slice(), out_cpu.buffer().as_slice());
}

#[test]
fn cuda_scalar_prod_reduction_matches_cpu() {
    let Some((_backend, mut cuda_ctx)) = load_cuda_backend() else {
        return;
    };

    let mut cpu_ctx = CpuContext::new(1);
    let input_base = tensor_f64(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let input = input_base.permute(&[1, 0]).unwrap();
    let desc = ScalarPrimsDescriptor::Reduction {
        modes_a: vec![0, 1],
        modes_c: vec![1],
        op: ScalarReductionOp::Prod,
    };
    let plan_cpu = <CpuBackend as TensorScalarPrims<Standard<f64>>>::plan(
        &mut cpu_ctx,
        &desc,
        &[input.dims(), &[2]],
    )
    .unwrap();
    let plan_cuda = <CudaBackend as TensorScalarPrims<Standard<f64>>>::plan(
        &mut cuda_ctx,
        &desc,
        &[input.dims(), &[2]],
    )
    .unwrap();

    let mut out_cpu = Tensor::<f64>::zeros(
        &[2],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    <CpuBackend as TensorScalarPrims<Standard<f64>>>::execute(
        &mut cpu_ctx,
        &plan_cpu,
        1.0,
        &[&input],
        0.0,
        &mut out_cpu,
    )
    .unwrap();

    let input_gpu = input
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let mut out_gpu = Tensor::<f64>::zeros(
        &[2],
        LogicalMemorySpace::GpuMemory { device_id: 0 },
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    <CudaBackend as TensorScalarPrims<Standard<f64>>>::execute(
        &mut cuda_ctx,
        &plan_cuda,
        1.0,
        &[&input_gpu],
        0.0,
        &mut out_gpu,
    )
    .unwrap();

    let out_cuda = out_gpu
        .to_memory_space_async(LogicalMemorySpace::MainMemory)
        .unwrap();
    assert_eq!(out_cuda.buffer().as_slice(), out_cpu.buffer().as_slice());
}

#[test]
fn cuda_scalar_threshold_and_mask_sum_match_cpu() {
    let Some((_backend, mut cuda_ctx)) = load_cuda_backend() else {
        return;
    };

    let mut cpu_ctx = CpuContext::new(1);
    let input_base = tensor_f32(&[0.5, 2.0, 3.5, 1.5, 2.5, -1.0], &[2, 3]);
    let threshold_base = tensor_f32(&[1.0, 1.0, 1.0, 2.0, 2.0, 2.0], &[2, 3]);
    let input = input_base.permute(&[1, 0]).unwrap();
    let threshold = threshold_base.permute(&[1, 0]).unwrap();

    let mask_desc = ScalarPrimsDescriptor::PointwiseBinary {
        op: ScalarBinaryOp::GreaterEqual,
    };
    let mask_plan_cpu = <CpuBackend as TensorScalarPrims<Standard<f32>>>::plan(
        &mut cpu_ctx,
        &mask_desc,
        &[input.dims(), threshold.dims(), input.dims()],
    )
    .unwrap();
    let mask_plan_cuda = <CudaBackend as TensorScalarPrims<Standard<f32>>>::plan(
        &mut cuda_ctx,
        &mask_desc,
        &[input.dims(), threshold.dims(), input.dims()],
    )
    .unwrap();

    let mut mask_cpu = Tensor::<f32>::zeros(
        input.dims(),
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    <CpuBackend as TensorScalarPrims<Standard<f32>>>::execute(
        &mut cpu_ctx,
        &mask_plan_cpu,
        1.0,
        &[&input, &threshold],
        0.0,
        &mut mask_cpu,
    )
    .unwrap();

    let input_gpu = input
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let threshold_gpu = threshold
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let mut mask_gpu = Tensor::<f32>::zeros(
        input.dims(),
        LogicalMemorySpace::GpuMemory { device_id: 0 },
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    <CudaBackend as TensorScalarPrims<Standard<f32>>>::execute(
        &mut cuda_ctx,
        &mask_plan_cuda,
        1.0,
        &[&input_gpu, &threshold_gpu],
        0.0,
        &mut mask_gpu,
    )
    .unwrap();

    let mask_cuda = mask_gpu
        .to_memory_space_async(LogicalMemorySpace::MainMemory)
        .unwrap();
    assert_eq!(mask_cuda.buffer().as_slice(), mask_cpu.buffer().as_slice());

    let reduce_desc = ScalarPrimsDescriptor::Reduction {
        modes_a: vec![0, 1],
        modes_c: vec![1],
        op: ScalarReductionOp::Sum,
    };
    let reduce_plan_cpu = <CpuBackend as TensorScalarPrims<Standard<f32>>>::plan(
        &mut cpu_ctx,
        &reduce_desc,
        &[mask_cpu.dims(), &[2]],
    )
    .unwrap();
    let reduce_plan_cuda = <CudaBackend as TensorScalarPrims<Standard<f32>>>::plan(
        &mut cuda_ctx,
        &reduce_desc,
        &[mask_gpu.dims(), &[2]],
    )
    .unwrap();

    let mut counts_cpu = Tensor::<f32>::zeros(
        &[2],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    <CpuBackend as TensorScalarPrims<Standard<f32>>>::execute(
        &mut cpu_ctx,
        &reduce_plan_cpu,
        1.0,
        &[&mask_cpu],
        0.0,
        &mut counts_cpu,
    )
    .unwrap();

    let mut counts_gpu = Tensor::<f32>::zeros(
        &[2],
        LogicalMemorySpace::GpuMemory { device_id: 0 },
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    <CudaBackend as TensorScalarPrims<Standard<f32>>>::execute(
        &mut cuda_ctx,
        &reduce_plan_cuda,
        1.0,
        &[&mask_gpu],
        0.0,
        &mut counts_gpu,
    )
    .unwrap();

    let counts_cuda = counts_gpu
        .to_memory_space_async(LogicalMemorySpace::MainMemory)
        .unwrap();
    assert_eq!(
        counts_cuda.buffer().as_slice(),
        counts_cpu.buffer().as_slice()
    );
    assert_eq!(counts_cpu.buffer().as_slice(), Some(&[2.0, 1.0][..]));
}
