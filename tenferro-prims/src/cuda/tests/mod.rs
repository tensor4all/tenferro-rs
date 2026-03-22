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
    CpuBackend, CpuContext, ScalarBinaryOp, ScalarPrimsDescriptor, ScalarReductionOp,
    ScalarUnaryOp, TensorScalarPrims,
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

fn load_cuda_backend() -> Option<(CudaBackend, CudaContext)> {
    let path = available_cutensor_library_path()?;
    if !cuda_device_zero_is_available() {
        return None;
    }
    Some(CudaBackend::load(path).unwrap())
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
    );

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

    let resolved = CudaBackend::resolve_conj(&mut ctx, &gpu);
    assert_eq!(
        resolved.logical_memory_space(),
        LogicalMemorySpace::GpuMemory { device_id: 0 }
    );
    assert!(!resolved.is_conjugated());

    let round_trip = resolved
        .to_memory_space_async(LogicalMemorySpace::MainMemory)
        .unwrap();
    assert_eq!(
        round_trip.buffer().as_slice(),
        Some(
            &[
                Complex64::new(1.0, -2.0),
                Complex64::new(3.0, 4.0),
                Complex64::new(-5.0, -6.0),
                Complex64::new(7.0, -8.0),
            ][..]
        )
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
    );
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
    );
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
    );
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
    );
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
    );
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
    );
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
    );
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
    );
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
    );
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
    );
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
    );
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
    );
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
